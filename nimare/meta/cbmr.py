"""Coordinate Based Meta Regression Methods."""

import copy
import logging
import re
import time
from functools import wraps

import nibabel as nib
import numpy as np
import pandas as pd
import scipy
import scipy.sparse
from nilearn.image import resample_to_img

try:
    import torch  # type: ignore[import-not-found]
except ImportError as e:
    raise ImportError(
        "Torch is required to use `CBMR` classes. Install with `pip install 'nimare[cbmr]'`."
    ) from e

from nimare import _version
from nimare.estimator import Estimator
from nimare.meta import models
from nimare.meta.utils import fit_voxelwise_cbmr_approximate
from nimare.results import MetaResult
from nimare.utils import (
    DEFAULT_FLOAT_DTYPE,
    _clip_p_values,
    _minimum_positive_float,
    b_spline_bases,
    dummy_encoding_moderators,
    get_masker,
    get_masker_mask_image,
    get_template,
    mm2vox,
    seed_torch,
    validate_coordinate_spaces,
)

LGR = logging.getLogger(__name__)
__version__ = _version.get_versions()["version"]
DEFAULT_GROUP_NAME = "Default"
DEFAULT_INCIDENCE_THRESHOLD = 0.001


def _uses_cuda(device):
    """Return whether the provided device string targets CUDA."""
    return str(device).startswith("cuda")


def _as_csr_matrix(value):
    """Return a sparse matrix in CSR format."""
    if scipy.sparse.isspmatrix_csr(value):
        return value
    return value.tocsr()


def _csr_row_indices(value):
    """Return row indices for each nonzero entry in a CSR matrix."""
    return np.repeat(np.arange(value.shape[0], dtype=value.indices.dtype), np.diff(value.indptr))


def _is_named_pairwise_contrast(contrast):
    """Return whether a contrast uses tuple shorthand like (A, B)."""
    return (
        isinstance(contrast, tuple)
        and len(contrast) == 2
        and all(isinstance(part, str) for part in contrast)
    )


def _validate_incidence_threshold(incidence_threshold):
    """Validate the empirical incidence threshold used for voxel filtering."""
    if incidence_threshold is None:
        return None
    incidence_threshold = float(incidence_threshold)
    if incidence_threshold < 0 or incidence_threshold >= 1:
        raise ValueError("incidence_threshold must be None or a value in [0, 1).")
    return incidence_threshold


def _normalize_named_pairwise_contrasts(contrasts):
    """Convert tuple shorthand like (A, B) into the legacy string form."""
    if contrasts is None:
        return None
    if isinstance(contrasts, str) or _is_named_pairwise_contrast(contrasts):
        contrasts = [contrasts]

    normalized = []
    for contrast in contrasts:
        if _is_named_pairwise_contrast(contrast):
            normalized.append(f"{contrast[0]}-{contrast[1]}")
        else:
            normalized.append(contrast)
    return normalized


def _clipped_stat_p_values(p_values, dtype=None):
    """Clip statistic p-values using a consistent dtype."""
    if dtype is None:
        dtype = np.asarray(p_values).dtype
    return _clip_p_values(p_values, dtype=dtype, copy=False)


def _normal_p_values(z_stats, two_sided=True, dtype=None):
    """Return clipped normal p-values for Wald statistics."""
    if two_sided:
        p_values = scipy.stats.norm.sf(np.abs(z_stats)) * 2
    else:
        p_values = scipy.stats.norm.sf(z_stats)
    return _clipped_stat_p_values(p_values, dtype=dtype)


def _chi_square_p_values(chi_square, df, dtype=None):
    """Return clipped chi-square p-values."""
    return _clipped_stat_p_values(scipy.stats.chi2.sf(chi_square, df=df), dtype=dtype)


def _two_sided_z_from_p_values(p_values):
    """Convert two-sided p-values to z-statistics with stable tail handling."""
    p_values = np.asarray(p_values)
    z_p_values = np.maximum(p_values, 2 * _minimum_positive_float(p_values.dtype))
    return scipy.stats.norm.isf(z_p_values / 2)


class CBMRResult(MetaResult):
    """Meta-analytic result for CBMR with result-centered inference helpers.

    The same result class is used for both standard global-moderator CBMR and voxelwise
    moderator-effect CBMR. Model-specific inference is selected from the fitted estimator's
    ``moderator_effect`` attribute.
    """

    @property
    def moderator_effect(self):
        """Return the fitted moderator-effect parameterization."""
        return getattr(self.estimator, "moderator_effect", "global")

    @property
    def groups(self):
        """Return fitted group names in display order."""
        return tuple(getattr(self.estimator, "groups", ()) or ())

    @property
    def moderators(self):
        """Return fitted moderator names in display order."""
        return tuple(getattr(self.estimator, "moderators", ()) or ())

    @property
    def voxelwise_moderator_effect_map_names(self):
        """Return voxelwise moderator-effect map names."""
        return tuple(name for name in self.maps if name.startswith("voxelwiseModeratorEffect_"))

    def copy(self):
        """Return a copy of the CBMR result object."""
        new = CBMRResult(
            estimator=self.estimator,
            corrector=self.corrector,
            diagnostics=self.diagnostics,
            mask=self.masker,
            maps=copy.deepcopy(self.maps),
            tables=copy.deepcopy(self.tables),
            description=self.description_,
        )
        new.metadata = copy.deepcopy(self.metadata)
        return new

    def describe_inference_inputs(self):
        """Summarize the fitted groups, moderators, and moderator-effect type."""
        return {
            "groups": self.groups,
            "moderators": self.moderators,
            "moderator_effect": self.moderator_effect,
        }

    def describe_voxelwise_moderator_effect_maps(self):
        """Return simple summaries for voxelwise moderator-effect maps."""
        return {
            name: (float(values.min()), float(values.mean()), float(values.max()))
            for name, values in self.maps.items()
            if name.startswith("voxelwiseModeratorEffect_")
        }

    def get_inference(self, device=None, method=None, **kwargs):
        """Return a fitted inference engine for advanced CBMR use cases.

        Parameters
        ----------
        device : str, optional
            Compute device to use for inference. Defaults to the device recorded on the fitted
            estimator.
        method : {"sandwich", "FI"}, optional
            Covariance estimator for voxelwise CBMR inference.
        **kwargs
            Additional keyword arguments passed to :class:`~nimare.meta.cbmr.CBMRInference`.
        """
        inference_device = device or getattr(self.estimator, "device", "cpu")
        inference = CBMRInference(
            device=inference_device,
            moderator_effect=self.moderator_effect,
            method=method,
            **kwargs,
        )
        inference.fit(self)
        return inference

    def infer(
        self,
        group_contrasts=False,
        moderator_contrasts=False,
        device=None,
        method=None,
        **kwargs,
    ):
        """Run CBMR inference from a fitted result.

        Parameters
        ----------
        group_contrasts : bool, dict, list, tuple, str, or None, optional
            Group homogeneity or comparison specification. Use ``False`` to skip group inference.
        moderator_contrasts : bool, dict, list, tuple, str, or None, optional
            Moderator effect or comparison specification. Use ``False`` to skip moderator
            inference.
        device : str, optional
            Compute device to use for inference. Defaults to the device recorded on the fitted
            estimator.
        method : {"sandwich", "FI"}, optional
            Covariance estimator for voxelwise CBMR inference.
        **kwargs
            Additional keyword arguments passed to :class:`~nimare.meta.cbmr.CBMRInference`.
        """
        inference = self.get_inference(device=device, method=method, **kwargs)
        return inference.transform(
            t_con_groups=group_contrasts,
            t_con_moderators=moderator_contrasts,
        )

    def _infer_named_effects(
        self,
        source,
        contrasts=None,
        pairwise=False,
        device=None,
        method=None,
        **kwargs,
    ):
        """Run inference for named group or moderator effects through one shared path."""
        if source == "groups":
            if contrasts is None:
                contrasts = list(self.groups)
            group_contrasts = (
                _normalize_named_pairwise_contrasts(contrasts) if pairwise else contrasts
            )
            moderator_contrasts = False
        elif source == "moderators":
            if not self.moderators:
                raise ValueError("This CBMR result does not include moderators.")
            if contrasts is None:
                contrasts = list(self.moderators)
            group_contrasts = False
            moderator_contrasts = (
                _normalize_named_pairwise_contrasts(contrasts) if pairwise else contrasts
            )
        else:
            raise ValueError("source must be either 'groups' or 'moderators'.")

        return self.infer(
            group_contrasts=group_contrasts,
            moderator_contrasts=moderator_contrasts,
            device=device,
            method=method,
            **kwargs,
        )

    def test_groups(self, groups=None, device=None, method=None, **kwargs):
        """Run one-group spatial homogeneity tests for the requested groups.

        Parameters
        ----------
        groups : list, tuple, str, or None, optional
            Group name or names to test. Defaults to all fitted groups.
        device : str, optional
            Compute device to use for inference. Defaults to the device recorded on the fitted
            estimator.
        method : {"sandwich", "FI"}, optional
            Covariance estimator for voxelwise CBMR inference.
        **kwargs
            Additional keyword arguments passed to :class:`~nimare.meta.cbmr.CBMRInference`.
        """
        return self._infer_named_effects(
            "groups",
            contrasts=groups,
            device=device,
            method=method,
            **kwargs,
        )

    def compare_groups(self, contrasts, device=None, method=None, **kwargs):
        """Run pairwise group-comparison tests using names or ``(group_a, group_b)`` tuples.

        Parameters
        ----------
        contrasts : list, tuple, or str
            Group comparison specification or specifications.
        device : str, optional
            Compute device to use for inference. Defaults to the device recorded on the fitted
            estimator.
        method : {"sandwich", "FI"}, optional
            Covariance estimator for voxelwise CBMR inference.
        **kwargs
            Additional keyword arguments passed to :class:`~nimare.meta.cbmr.CBMRInference`.
        """
        return self._infer_named_effects(
            "groups",
            contrasts=contrasts,
            pairwise=True,
            device=device,
            method=method,
            **kwargs,
        )

    def test_moderators(self, moderators=None, device=None, method=None, **kwargs):
        """Test whether the requested moderator effects differ from zero.

        Parameters
        ----------
        moderators : list, tuple, str, or None, optional
            Moderator name or names to test. Defaults to all fitted moderators.
        device : str, optional
            Compute device to use for inference. Defaults to the device recorded on the fitted
            estimator.
        method : {"sandwich", "FI"}, optional
            Covariance estimator for voxelwise CBMR inference.
        **kwargs
            Additional keyword arguments passed to :class:`~nimare.meta.cbmr.CBMRInference`.
        """
        return self._infer_named_effects(
            "moderators",
            contrasts=moderators,
            device=device,
            method=method,
            **kwargs,
        )

    def compare_moderators(self, contrasts, device=None, method=None, **kwargs):
        """Run pairwise moderator-comparison tests using names or tuples.

        Parameters
        ----------
        contrasts : list, tuple, or str
            Moderator comparison specification or specifications.
        device : str, optional
            Compute device to use for inference. Defaults to the device recorded on the fitted
            estimator.
        method : {"sandwich", "FI"}, optional
            Covariance estimator for voxelwise CBMR inference.
        **kwargs
            Additional keyword arguments passed to :class:`~nimare.meta.cbmr.CBMRInference`.
        """
        return self._infer_named_effects(
            "moderators",
            contrasts=contrasts,
            pairwise=True,
            device=device,
            method=method,
            **kwargs,
        )


class CBMREstimator(Estimator):
    """Coordinate-based meta-regression with a spatial model.

    .. warning::
        Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed in
        a future release. Prefer :class:`~nimare.nimads.Studyset`.

    .. versionadded:: 0.1.0

    Parameters
    ----------
    group_categories : :obj:`~str` or obj:`~list` or obj:`~None`, optional
        CBMR allows a collection to be categorized into multiple groups according to one or more
        group categories. Default is one-group CBMR.
    moderators : :obj:`~str` or obj:`~list` or obj:`~None`, optional
        CBMR can accommodate moderators (e.g. sample size, year of publication).
        Default is CBMR without moderators.
    moderator_effect : {"voxelwise", "global"}, optional
        How experiment-level moderator effects are parameterized. ``"global"`` fits the
        standard CBMR model with one coefficient per moderator and group. ``"voxelwise"`` fits
        the voxelwise moderator-effect CBMR backend, in which moderator effects vary smoothly over
        voxels.
        Default is ``"voxelwise"``.
    mask : :obj:`str`, :class:`~nibabel.nifti1.Nifti1Image`, or Nilearn masker, optional
        Region-of-interest mask. If None, CBMR uses the whole 2 mm MNI152 brain mask.
    incidence_threshold : :obj:`float` or None, optional
        Drop voxels with empirical focus incidence less than or equal to this threshold after
        applying ``mask``. Empirical incidence is the fraction of experiments with at least one
        focus in a voxel. Use None to retain all voxels in ``mask``. Default is 0.001.
    model : subclass of :class:`~nimare.meta.models.GeneralLinearModelEstimator`, optional
        Stochastic model class used by CBMR. Available options are:

        - :class:`~nimare.meta.models.PoissonEstimator` (default): the most efficient
          and widely used option, but slightly less accurate because it approximates
          low-rate binomial data, cannot account for over-dispersion in foci counts,
          and may underestimate standard errors.
        - :class:`~nimare.meta.models.NegativeBinomialEstimator`: slower and
          sometimes less stable, but slightly more accurate. This model allows
          anticipated excess variance relative to Poisson via a group-wise
          overdispersion parameter shared by all experiments and voxels.
        - :class:`~nimare.meta.models.ClusteredNegativeBinomialEstimator`: a
          random-effects Poisson variant that models experiment-level latent
          characteristics shared across the brain for a given experiment.
    penalty : :obj:`~bool`, optional
        Currently, the only available option is Firth-type penalty, which penalizes the
        likelihood function by Jeffreys' invariant prior and encourages convergence.
    spline_spacing : :obj:`~int`, optional
        Spatial structure of foci counts is parameterized by the coefficients of cubic
        B-spline bases in CBMR. Spatial smoothness in CBMR is determined by spline spacing,
        which is shared across the x, y, and z dimensions. Default is 10.
    n_iter : :obj:`int`, optional
        Number of iterations allowed in the log-likelihood optimization.
        Default is 2000.
    lr: :obj:`float`, optional
        Learning rate in optimization of log-likelihood function.
        Default is 1.
    lr_decay: :obj:`float`, optional
        Multiplicative factor of learning rate decay.
        Default is 0.999.
    tol: :obj:`float`, optional
        Stopping criterion based on the change in log-likelihood between two consecutive
        iterations. Default is 1e-9.
    device: :obj:`string`, optional
        Device type ('cpu' or 'cuda') representing where operations will be allocated.
        Default is 'cpu'.
    random_state : :obj:`int`, optional
        Random seed used for torch-based weight initialization. Default is None.
    **kwargs
        Keyword arguments. Arguments for the Estimator can be assigned here,
        Additional masking controls are exposed as ``mask`` and ``incidence_threshold``.

    Attributes
    ----------
    masker : :class:`~nilearn.maskers.NiftiMasker` or similar
        Masker object.
    inputs_ : :obj:`dict`
        Inputs to the Estimator. For CBMR estimators, this includes the following keys:
        coordinates,
        mask_img (brain mask image),
        id (experiment ids),
        ids_by_group (experiment ids categorized by groups),
        moderators_by_group (moderators categorized by groups, if present),
        coef_spline_bases (spatial matrix of cubic B-spline coefficients in x, y, and z),
        foci_by_experiment (experiment-by-voxel sparse focus-count matrices, categorized by
        groups),
        foci_per_voxel (voxelwise sum of foci counts across experiments, categorized by groups),
        foci_per_experiment (experiment-wise sum of foci counts across space, categorized by
        groups).

    Notes
    -----
    Follow-up inference is exposed through :class:`~nimare.meta.cbmr.CBMRResult` and
    :class:`~nimare.meta.cbmr.CBMRInference`.
    """

    _required_inputs = {"coordinates": ("coordinates", None)}
    _group_column = "_cbmr_group"
    _valid_moderator_effects = ("global", "voxelwise")
    _valid_backends = ("full", "approximate")

    @classmethod
    def _validate_moderator_effect(cls, moderator_effect):
        """Validate and normalize the public moderator-effect selector."""
        if isinstance(moderator_effect, str):
            moderator_effect = moderator_effect.lower()
        if moderator_effect not in cls._valid_moderator_effects:
            raise ValueError(
                "moderator_effect must be one of "
                f"{cls._valid_moderator_effects}. Got {moderator_effect!r}."
            )
        return moderator_effect

    def __init__(
        self,
        group_categories=None,
        moderators=None,
        moderator_effect="voxelwise",
        mask=None,
        incidence_threshold=DEFAULT_INCIDENCE_THRESHOLD,
        spline_spacing=10,
        model=models.PoissonEstimator,
        penalty=False,
        backend="full",
        n_iter=2000,
        lr=1,
        lr_decay=0.999,
        tol=1e-9,
        device="cpu",
        random_state=None,
        alpha=1.0,
        damping=1e-4,
        compute_nll=False,
        **kwargs,
    ):
        self._init_pipeline_state(
            group_categories=group_categories,
            moderators=moderators,
            moderator_effect=moderator_effect,
            mask=mask,
            incidence_threshold=incidence_threshold,
            spline_spacing=spline_spacing,
            model=model,
            penalty=penalty,
            backend=backend,
            n_iter=n_iter,
            lr=lr,
            lr_decay=lr_decay,
            tol=tol,
            device=device,
            random_state=random_state,
            alpha=alpha,
            damping=damping,
            compute_nll=compute_nll,
            **kwargs,
        )

    def _init_pipeline_state(
        self,
        group_categories=None,
        moderators=None,
        moderator_effect="voxelwise",
        mask=None,
        incidence_threshold=DEFAULT_INCIDENCE_THRESHOLD,
        spline_spacing=10,
        model=models.PoissonEstimator,
        penalty=False,
        backend="full",
        n_iter=2000,
        lr=1,
        lr_decay=0.999,
        tol=1e-9,
        device="cpu",
        random_state=None,
        alpha=1.0,
        damping=1e-4,
        compute_nll=False,
        **kwargs,
    ):
        """Initialize shared estimator state for the selected concrete CBMR pipeline."""
        super().__init__(**kwargs)
        self.moderator_effect = self._validate_moderator_effect(moderator_effect)
        self.incidence_threshold = _validate_incidence_threshold(incidence_threshold)
        if backend not in self._valid_backends:
            raise ValueError(f"backend must be one of {self._valid_backends}. Got {backend!r}.")
        if self.moderator_effect == "voxelwise" and model is not models.PoissonEstimator:
            raise ValueError("Voxelwise CBMR currently requires model=models.PoissonEstimator.")
        self.mask = mask
        self.masker = get_masker(mask) if mask is not None else None

        self.group_categories = group_categories
        self.moderators = moderators

        self.spline_spacing = spline_spacing
        model_class = models.PoissonEstimator if self.moderator_effect == "voxelwise" else model
        self.model = model_class(
            penalty=penalty, lr=lr, lr_decay=lr_decay, n_iter=n_iter, tol=tol, device=device
        )
        self.penalty = penalty
        self.backend = backend
        self.n_iter = n_iter
        self.lr = lr
        self.lr_decay = lr_decay
        self.tol = tol
        self.device = device
        self.random_state = random_state
        self.alpha = alpha
        self.damping = damping
        self.compute_nll = compute_nll
        self.voxelwise_model = None
        self.voxelwise_coef = None
        if _uses_cuda(self.device) and not torch.cuda.is_available():
            LGR.debug("cuda not found, use device cpu")
            self.device = "cpu"
        self.model.device = self.device

    def _generate_description(self):
        """Generate a description of the Estimator instance.

        Returns
        -------
        description : :obj:`str`
            Description of the Estimator instance.
        """
        description = """CBMR is a meta-regression framework that can explicitly model
                    group-wise spatial intensity function, and consider the effect of
                    moderators. It consists of two components: (1) a spatial
                    model that makes use of a spline parameterization to induce a smooth
                    response; (2) a generalized linear model (Poisson, Negative Binomial
                    (NB), Clustered NB) to model group-wise spatial intensity function).
                    CBMR is fitted via maximizing the log-likelihood function with L-BFGS
                    algorithm."""
        if self.moderators:
            moderators_str = f"""and accommodate the following moderators:
                            {', '.join(self.moderators)}"""
        else:
            moderators_str = ""
        moderator_effect_str = (
            " Moderator effects were modeled as global effects."
            if self.moderator_effect == "global"
            else " Moderator effects were modeled as voxelwise effects."
        )
        if self.model.penalty:
            penalty_str = " Firth-type penalty is applied to ensure convergence."
        else:
            penalty_str = ""

        if type(self.model).__name__ == "PoissonEstimator":
            model_str = (
                " Here, Poisson model \\citep{eisenberg1966general} is the most basic CBMR model. "
                "It's based on the assumption that foci arise from a realisation of a continuous "
                "inhomogeneous Poisson process, so that the (discrete) voxel-wise foci counts will"
                " be independently distributed as Poisson random variables, with rate equal to the"
                " integral of the true, unobserved, continuous intensity function over each voxel."
            )
        elif type(self.model).__name__ == "NegativeBinomialEstimator":
            model_str = (
                " Negative Binomial (NB) model \\citep{barndorff1969negative} is a generalized "
                "Poisson model with over-dispersion. "
                "It's a more flexible model, but more difficult to estimate. In practice, foci"
                "counts often display over-dispersion (the variance of response variable"
                " substantially exceeds the mean), which is not captured by the Poisson model."
            )
        elif type(self.model).__name__ == "ClusteredNegativeBinomialEstimator":
            model_str = (
                " Clustered NB model \\citep{geoffroy2001poisson} can also accommodate "
                "over-dispersion in foci counts. "
                "In the NB model, the latent random variable introduces independent variation "
                "at each voxel. While in the Clustered NB model, we assert the random effects "
                "are not independent voxelwise effects, but rather latent characteristics of "
                "each experiment, and represent a shared effect over the entire brain for a "
                "given experiment."
            )

        model_description = (
            f"CBMR is a meta-regression framework that was performed with NiMARE {__version__}. "
            f"{type(self.model).__name__} model was used to model group-wise spatial intensity "
            f"functions {moderators_str}." + moderator_effect_str + model_str
        )

        optimization_description = (
            "CBMR is fitted via maximizing the log-likelihood function with L-BFGS algorithm, with"
            f" learning rate {self.lr}, learning rate decay {self.lr_decay} and "
            f"tolerance {self.tol}." + penalty_str + f" The optimization is run on {self.device}."
            f" The input dataset included {self.inputs_['coordinates'].shape[0]} foci from "
            f"{len(self.inputs_['id'])} experiments. The analysis mask included "
            f"{self.inputs_['coef_spline_bases'].shape[0]} voxels after ROI and empirical "
            f"incidence filtering."
        )

        description = model_description + "\n" + optimization_description
        return description

    def _make_result(self, dataset, maps=None, tables=None, description=""):
        """Construct a CBMR-specific result object."""
        masker = self.masker or dataset.masker
        return CBMRResult(self, mask=masker, maps=maps, tables=tables, description=description)

    def _resolve_roi_masker(self, dataset):
        """Return the user-requested ROI masker or the default 2 mm MNI brain masker."""
        if self.masker is not None:
            return get_masker(self.masker)

        default_mask_img = get_template(space="mni152_2mm", mask="brain")
        return get_masker(default_mask_img)

    @staticmethod
    def _mask_image_from_data(mask_data, reference_img):
        """Create a binary mask image aligned to a reference image."""
        header = reference_img.header.copy()
        header.set_data_dtype(np.uint8)
        return nib.Nifti1Image(mask_data.astype(np.uint8), reference_img.affine, header)

    @staticmethod
    def _build_mask_lookup(mask_data):
        """Return a flat-index lookup from full image space to masked voxel space."""
        n_mask_voxels = int(mask_data.sum())
        mask_lookup = np.full(mask_data.size, -1, dtype=np.int32)
        mask_lookup[np.flatnonzero(mask_data.ravel())] = np.arange(n_mask_voxels, dtype=np.int32)
        return mask_lookup, n_mask_voxels

    def _initialize_spatial_inputs(self, masker, mask_img):
        """Build and cache mask-derived spatial inputs used by CBMR."""
        self.inputs_["mask_img"] = mask_img
        mask_data = np.asanyarray(mask_img.dataobj).astype(bool, copy=False)
        mask_lookup, n_mask_voxels = self._build_mask_lookup(mask_data)
        self.inputs_["coef_spline_bases"] = b_spline_bases(
            masker_voxels=mask_data,
            spacing=self.spline_spacing,
        )
        return mask_data, mask_lookup, n_mask_voxels

    def _compute_empirical_incidence(self, coordinates, ids_by_group, n_mask_voxels):
        """Return the empirical voxel incidence rate across experiments."""
        n_experiments = sum(len(group_ids) for group_ids in ids_by_group.values())
        if n_experiments == 0:
            raise ValueError("CBMR requires at least one experiment.")

        foci_by_experiment = self._build_group_foci_matrices(
            coordinates,
            ids_by_group,
            n_mask_voxels,
        )
        incidence_count = np.zeros(n_mask_voxels, dtype=np.float64)
        for group_matrix in foci_by_experiment.values():
            incidence_count += np.asarray((group_matrix > 0).sum(axis=0)).ravel()
        return incidence_count / float(n_experiments)

    def _threshold_mask_by_incidence(
        self,
        mask_img,
        mask_data,
        filtered_coordinates,
        ids_by_group,
        n_mask_voxels,
    ):
        """Apply empirical-incidence filtering to an ROI mask."""
        incidence_rate = self._compute_empirical_incidence(
            filtered_coordinates,
            ids_by_group,
            n_mask_voxels,
        )
        self.inputs_["empirical_incidence_rate_roi"] = incidence_rate

        if self.incidence_threshold is None:
            keep_voxels = np.ones(n_mask_voxels, dtype=bool)
        else:
            keep_voxels = incidence_rate > self.incidence_threshold

        if not np.any(keep_voxels):
            raise ValueError(
                "No voxels survived CBMR incidence filtering. Lower incidence_threshold or "
                "provide a less restrictive mask."
            )

        thresholded_mask_data = np.zeros(mask_data.size, dtype=bool)
        roi_flat_indices = np.flatnonzero(mask_data.ravel())
        thresholded_mask_data[roi_flat_indices[keep_voxels]] = True
        thresholded_mask_data = thresholded_mask_data.reshape(mask_data.shape)
        self.inputs_["empirical_incidence_rate"] = incidence_rate[keep_voxels]
        self.inputs_["incidence_threshold"] = self.incidence_threshold
        return self._mask_image_from_data(thresholded_mask_data, mask_img)

    def _filter_coordinates_to_mask(self, coordinates, mask_img, mask_data, mask_lookup):
        """Filter coordinates to the mask and attach masked-space voxel indices."""
        if coordinates.empty:
            filtered_coordinates = coordinates.copy()
            filtered_coordinates["_cbmr_mask_index"] = pd.Series(dtype=np.int32)
            return filtered_coordinates

        ijk = mm2vox(coordinates[["x", "y", "z"]].to_numpy(), mask_img.affine)
        shape = np.asarray(mask_data.shape, dtype=np.int64)
        in_bounds = np.all((ijk >= 0) & (ijk < shape), axis=1)

        keep_mask = np.zeros(coordinates.shape[0], dtype=bool)
        mask_indices = np.empty(0, dtype=np.int32)
        if np.any(in_bounds):
            bounded_idx = np.where(in_bounds)[0]
            bounded_ijk = ijk[bounded_idx]
            flat_voxel_index = np.ravel_multi_index(bounded_ijk.T, mask_data.shape)
            in_mask = mask_data.ravel()[flat_voxel_index]
            kept_idx = bounded_idx[in_mask]
            keep_mask[kept_idx] = True
            mask_indices = mask_lookup[flat_voxel_index[in_mask]]

        n_dropped = int(coordinates.shape[0] - keep_mask.sum())
        LGR.info(
            "%d/%d coordinates fall outside of the mask. Removing them.",
            n_dropped,
            coordinates.shape[0],
        )

        filtered_coordinates = coordinates.loc[keep_mask].copy()
        filtered_coordinates["_cbmr_mask_index"] = mask_indices
        return filtered_coordinates

    @staticmethod
    def _format_group_name(group_value):
        """Normalize a group label into the public map/table naming format."""
        if isinstance(group_value, (list, tuple, np.ndarray, pd.Series)):
            return "".join(str(value).capitalize() for value in group_value)
        return str(group_value).capitalize()

    def _collect_experiment_annotations(self, dataset):
        """Return one annotation row per experiment in the collected input order."""
        experiment_annotations = (
            dataset.annotations_df[dataset.annotations_df["id"].isin(self.inputs_["id"])]
            .drop_duplicates(subset=["id"])
            .set_index("id", drop=False)
            .reindex(self.inputs_["id"])
            .reset_index(drop=True)
        )
        experiment_annotations = experiment_annotations.copy()
        experiment_annotations["id"] = self.inputs_["id"]
        return experiment_annotations

    def _assign_group_labels(self, experiment_annotations):
        """Attach normalized group labels to the aligned experiment table."""
        if self.group_categories is None:
            experiment_annotations[self._group_column] = DEFAULT_GROUP_NAME
        elif isinstance(self.group_categories, str):
            if self.group_categories not in experiment_annotations.columns:
                raise ValueError(
                    f"Category_names: {self.group_categories} does not exist in the dataset"
                )
            experiment_annotations[self._group_column] = experiment_annotations[
                self.group_categories
            ].map(self._format_group_name)
        elif isinstance(self.group_categories, list):
            missing_categories = set(self.group_categories) - set(experiment_annotations.columns)
            if missing_categories:
                raise ValueError(
                    f"Category_names: {missing_categories} do/does not exist in the dataset."
                )
            experiment_annotations[self._group_column] = experiment_annotations[
                self.group_categories
            ].apply(lambda row: self._format_group_name(row.tolist()), axis=1)
        else:
            raise ValueError("group_categories must be None, a string, or a list of strings.")

        return experiment_annotations

    def _index_experiments_by_group(self, experiment_annotations):
        """Return experiment IDs grouped in the order used by downstream summaries."""
        ids_by_group = (
            experiment_annotations.groupby(self._group_column, sort=False)["id"]
            .agg(list)
            .to_dict()
        )
        return ids_by_group

    def _build_group_moderators(self, experiment_annotations):
        """Collect moderator arrays in the same experiment order used elsewhere in CBMR."""
        if not self.moderators:
            self.inputs_.pop("moderators_by_group", None)
            return experiment_annotations, None

        experiment_annotations, self.moderators = dummy_encoding_moderators(
            experiment_annotations, self.moderators
        )
        if isinstance(self.moderators, str):
            self.moderators = [self.moderators]

        moderators_by_group = {}
        for group, group_annotations in experiment_annotations.groupby(
            self._group_column, sort=False
        ):
            moderators_by_group[group] = group_annotations[self.moderators].to_numpy()

        return experiment_annotations, moderators_by_group

    def _build_experiment_group_inputs(self, dataset, filtered_coordinates, n_mask_voxels):
        """Assemble grouped experiment IDs, moderators, and focus summaries."""
        experiment_annotations = self._collect_experiment_annotations(dataset)
        experiment_annotations = self._assign_group_labels(experiment_annotations)
        ids_by_group = self._index_experiments_by_group(experiment_annotations)
        self.groups = list(ids_by_group.keys())

        experiment_annotations, moderators_by_group = self._build_group_moderators(
            experiment_annotations
        )
        foci_by_experiment, foci_per_voxel, foci_per_experiment = self._build_group_foci(
            filtered_coordinates,
            ids_by_group,
            n_mask_voxels,
        )
        inputs = {
            "ids_by_group": ids_by_group,
            "moderators_by_group": moderators_by_group,
            "foci_by_experiment": foci_by_experiment,
            "foci_per_voxel": foci_per_voxel,
            "foci_per_experiment": foci_per_experiment,
        }
        if self.moderator_effect == "voxelwise":
            inputs["foci_by_experiment_voxel"] = self._build_group_foci_matrices(
                filtered_coordinates,
                ids_by_group,
                n_mask_voxels,
            )
        return inputs

    @staticmethod
    def _build_group_foci_matrices(coordinates, ids_by_group, n_mask_voxels):
        """Return experiment-by-voxel foci count matrices for each group."""
        return CBMREstimator._build_group_sparse_foci_matrices(
            coordinates,
            ids_by_group,
            n_mask_voxels,
            dtype=np.float64,
        )

    @staticmethod
    def _build_group_sparse_foci_matrices(coordinates, ids_by_group, n_mask_voxels, dtype):
        """Return grouped experiment-by-voxel sparse focus-count matrices."""
        foci_by_experiment = {}
        if coordinates.empty:
            for group, group_ids in ids_by_group.items():
                foci_by_experiment[group] = scipy.sparse.csr_matrix(
                    (len(group_ids), n_mask_voxels),
                    dtype=dtype,
                )
            return foci_by_experiment

        coordinates = coordinates.loc[:, ["id", "_cbmr_mask_index"]].copy()
        for group, group_ids in ids_by_group.items():
            id_to_row = {exp_id: i for i, exp_id in enumerate(group_ids)}
            group_coordinates = coordinates.loc[coordinates["id"].isin(group_ids)]
            rows = group_coordinates["id"].map(id_to_row).to_numpy(dtype=np.int64, copy=False)
            cols = group_coordinates["_cbmr_mask_index"].to_numpy(dtype=np.int64, copy=False)
            data = np.ones(group_coordinates.shape[0], dtype=dtype)
            foci_by_experiment[group] = scipy.sparse.csr_matrix(
                (data, (rows, cols)),
                shape=(len(group_ids), n_mask_voxels),
                dtype=dtype,
            )
        return foci_by_experiment

    def _as_torch_tensor(self, value):
        """Convert an array-like object to a float64 tensor on the estimator device."""
        if scipy.sparse.issparse(value):
            value = _as_csr_matrix(value).toarray()
        return torch.as_tensor(value, dtype=torch.float64, device=self.device)

    def _prepare_torch_inputs(self):
        """Return tensorized voxelwise moderator-effect CBMR inputs."""
        bases = self._as_torch_tensor(self.inputs_["coef_spline_bases"])
        moderators_by_group = None
        if self.moderators:
            moderators_by_group = {
                group: self._as_torch_tensor(self.inputs_["moderators_by_group"][group])
                for group in self.groups
            }
        foci_by_experiment_voxel = {
            group: self._as_torch_tensor(self.inputs_["foci_by_experiment_voxel"][group])
            for group in self.groups
        }
        return bases, moderators_by_group, foci_by_experiment_voxel

    def _voxelwise_cbmr_description(self, backend):
        """Generate a NiMARE-style description for voxelwise moderator-effect CBMR."""
        if self.moderators:
            moderator_text = (
                "with voxelwise moderator effects for " f"{', '.join(self.moderators)}"
            )
        else:
            moderator_text = "without experiment-level moderators"
        return (
            f"Voxelwise moderator-effect CBMR was performed with the {backend} backend "
            f"{moderator_text}. The model used {len(self.groups)} group(s), "
            f"spline spacing {self.spline_spacing}, device {self.device}, and "
            f"{self.inputs_['coef_spline_bases'].shape[0]} analysis-mask voxels after ROI and "
            f"empirical incidence filtering."
        )

    def _build_group_foci(self, coordinates, ids_by_group, n_mask_voxels):
        """Summarize voxelwise and experiment-wise focus counts for each group."""
        foci_by_experiment = self._build_group_sparse_foci_matrices(
            coordinates,
            ids_by_group,
            n_mask_voxels,
            dtype=np.int32,
        )

        foci_per_voxel = {}
        foci_per_experiment = {}
        for group, group_foci_by_experiment in foci_by_experiment.items():
            group_foci_per_voxel = np.asarray(
                group_foci_by_experiment.sum(axis=0), dtype=np.int32
            ).reshape((-1, 1))
            group_foci_per_experiment = np.asarray(
                group_foci_by_experiment.sum(axis=1), dtype=np.int32
            ).reshape((-1, 1))

            foci_per_voxel[group] = group_foci_per_voxel
            foci_per_experiment[group] = group_foci_per_experiment

        return foci_by_experiment, foci_per_voxel, foci_per_experiment

    def _preprocess_input(self, dataset):
        """Mask required input images using either the Dataset's mask or the Estimator's.

        Also, categorize experiment id, voxelwise sum of foci counts across experiments,
        experiment-wise sum of foci counts across space into multiple groups. And summarize
        moderators into
        multiple groups (if exist).

        Parameters
        ----------
        dataset : :obj:`~nimare.nimads.Studyset` or :obj:`~nimare.dataset.Dataset`
            In this method, the collection is used to (1) select the appropriate mask image,
            (2) categorize experiments into multiple groups according to group categories in
            annotations,
            (3) summarize group-wise experiment id, moderators (if exist), foci per voxel, foci
            per experiment,
            (4) extract sample size metadata and use it as one of the moderators.

        Attributes
        ----------
        inputs_ : :obj:`dict`
            Specifically, (1) a "mask_img" key will be added (brain mask image),
            (2) an 'id' key will be added (id of all experiments in the dataset),
            (3) a 'coef_spline_bases' key will be added (spatial matrix of coefficient of cubic
            B-spline bases in x,y,z dimension),
            (4) an 'ids_by_group' key will be added (experiment id categorized by groups),
            (5) a 'moderators_by_group' key will be added (moderators categorized
            by groups) if moderators are considered,
            (6) a 'foci_by_experiment' key will be added (experiment-by-voxel sparse focus-count
            matrices, categorized by groups),
            (7) an 'foci_per_voxel' key will be added (voxelwise sum of foci count across
            experiments, categorized by groups),
            (8) an 'foci_per_experiment' key will be added (experiment-wise sum of
            foci count across space, categorized by groups).

        .. warning::
            Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed
            in a future release. Prefer :class:`~nimare.nimads.Studyset`.
        """
        validate_coordinate_spaces(self.inputs_["coordinates"])

        roi_masker = self._resolve_roi_masker(dataset)
        _, roi_mask_img = get_masker_mask_image(roi_masker)
        roi_mask_data = np.asanyarray(roi_mask_img.dataobj).astype(bool, copy=False)
        roi_mask_lookup, n_roi_voxels = self._build_mask_lookup(roi_mask_data)
        roi_filtered_coordinates = self._filter_coordinates_to_mask(
            self.inputs_["coordinates"],
            roi_mask_img,
            roi_mask_data,
            roi_mask_lookup,
        )

        experiment_annotations = self._collect_experiment_annotations(dataset)
        experiment_annotations = self._assign_group_labels(experiment_annotations)
        ids_by_group = self._index_experiments_by_group(experiment_annotations)

        analysis_mask_img = self._threshold_mask_by_incidence(
            roi_mask_img,
            roi_mask_data,
            roi_filtered_coordinates,
            ids_by_group,
            n_roi_voxels,
        )
        self.masker = get_masker(analysis_mask_img)
        _, mask_img = get_masker_mask_image(self.masker)
        mask_data, mask_lookup, n_mask_voxels = self._initialize_spatial_inputs(
            self.masker,
            mask_img,
        )
        filtered_coordinates = self._filter_coordinates_to_mask(
            self.inputs_["coordinates"],
            mask_img,
            mask_data,
            mask_lookup,
        )
        self.inputs_["coordinates"] = filtered_coordinates.drop(columns=["_cbmr_mask_index"])
        self.inputs_.update(
            self._build_experiment_group_inputs(
                dataset,
                filtered_coordinates,
                n_mask_voxels,
            )
        )

    def _fit(self, dataset):
        """Perform coordinate-based meta-regression (CBMR) on dataset.

        (1) Estimate group-wise spatial regression coefficients and its standard error via
        inverse of Fisher Information matrix; Similarly, estimate regression coefficient of
        moderators (if exist), as well as its standard error via inverse of
        Fisher Information matrix;
        (2) Estimate standard error of group-wise log intensity, group-wise intensity via delta
        method;
        (3) For NegativeBinomial or ClusteredNegativeBinomial model, estimate regression
        coefficient of overdispersion.s

        Parameters
        ----------
        dataset : :obj:`~nimare.nimads.Studyset` or :obj:`~nimare.dataset.Dataset`
            Collection to analyze.

        .. warning::
            Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed
            in a future release. Prefer :class:`~nimare.nimads.Studyset`.
        """
        if self.moderator_effect == "voxelwise":
            if self.backend == "approximate":
                return self._fit_approximate(dataset)
            return self._fit_full(dataset)

        init_weight_kwargs = {
            "groups": self.groups,
            "moderators": self.moderators,
            "spatial_coef_dim": self.inputs_["coef_spline_bases"].shape[1],
            "moderators_coef_dim": len(self.moderators) if self.moderators else None,
        }
        seed_torch(self.random_state, self.device)
        self.model.init_weights(**init_weight_kwargs)

        moderators_by_group = self.inputs_["moderators_by_group"] if self.moderators else None
        self.model.fit(
            self.inputs_["coef_spline_bases"],
            moderators_by_group,
            self.inputs_["foci_per_voxel"],
            self.inputs_["foci_per_experiment"],
        )

        maps, tables = self.model.summary()

        return maps, tables, self._description_text()

    def _fit_full(self, dataset):
        """Fit voxelwise moderator-effect CBMR with the full torch L-BFGS backend."""
        seed_torch(self.random_state, self.device)
        bases, moderators_by_group, foci_by_experiment_voxel = self._prepare_torch_inputs()
        moderators_coef_dim = len(self.moderators) if self.moderators else None
        self.voxelwise_model = models.SpatialCBMRModel(
            groups=self.groups,
            spatial_coef_dim=self.inputs_["coef_spline_bases"].shape[1],
            moderators_coef_dim=moderators_coef_dim,
            device=self.device,
        )
        optimizer = torch.optim.LBFGS(
            params=self.voxelwise_model.parameters(),
            lr=self.lr,
            max_iter=self.n_iter,
            tolerance_change=self.tol,
            line_search_fn="strong_wolfe",
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
        start_time = time.time()

        def closure():
            """Evaluate the L-BFGS closure required by torch."""
            optimizer.zero_grad()
            loss = self.voxelwise_model(
                bases,
                moderators_by_group,
                foci_by_experiment_voxel,
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        scheduler.step()
        LGR.info(
            "Voxelwise moderator-effect CBMR optimisation took %.1f s.",
            time.time() - start_time,
        )
        maps, tables = self._extract_torch_results(moderators_by_group)
        return maps, tables, self._voxelwise_cbmr_description("full L-BFGS")

    def _fit_approximate(self, dataset):
        """Fit voxelwise moderator-effect CBMR with the approximate backend."""
        bases = self.inputs_["coef_spline_bases"]
        maps = {}
        tables = {}
        self.voxelwise_coef = {}
        for group in self.groups:
            foci = self.inputs_["foci_by_experiment_voxel"][group]
            if self.moderators:
                moderators = self.inputs_["moderators_by_group"][group]
            else:
                moderators = np.empty((foci.shape[0], 0), dtype=np.float64)

            augmented_moderators = np.column_stack(
                [moderators, np.ones((foci.shape[0], 1), dtype=np.float64)]
            )
            coefficient = self._voxelwise_cbmr_approximate_solver(
                augmented_moderators,
                bases,
                foci,
                tol=self.tol,
                max_iter=self.n_iter,
                alpha=self.alpha,
                damping=self.damping,
                compute_nll=self.compute_nll,
            )
            self.voxelwise_coef[group] = coefficient
            self._add_approximate_results(
                maps,
                tables,
                group,
                moderators,
                coefficient,
            )
        return maps, tables, self._voxelwise_cbmr_description("approximate")

    @property
    def _voxelwise_cbmr_approximate_solver(self):
        """Return the approximate solver used by the voxelwise backend."""
        return fit_voxelwise_cbmr_approximate

    def _extract_torch_results(self, moderators_by_group):
        """Extract maps and coefficient tables from the fitted torch model."""
        bases = self.inputs_["coef_spline_bases"]
        maps = {}
        tables = {}
        for group in self.groups:
            spatial_coef = (
                self.voxelwise_model.spatial_coef_linears[group]
                .weight.detach()
                .cpu()
                .numpy()
                .ravel()
            )
            maps[f"spatialIntensity_group-{group}"] = np.exp(bases @ spatial_coef)
            self._add_spatial_coef_table(tables, group, spatial_coef)
            if self.moderators:
                moderator_coef = (
                    self.voxelwise_model.moderator_coef_linears[group]
                    .weight.detach()
                    .cpu()
                    .numpy()
                )
                moderators = moderators_by_group[group].detach().cpu().numpy()
                self._add_moderator_maps_and_tables(
                    maps,
                    tables,
                    group,
                    moderators,
                    moderator_coef,
                )
        return maps, tables

    @staticmethod
    def _add_spatial_coef_table(tables, group, spatial_coef):
        """Add one group to the CBMR-style spatial coefficient table."""
        columns = [f"basis_{i}" for i in range(spatial_coef.size)]
        spatial_coef_table = tables.get(
            "spatial_regression_coef",
            pd.DataFrame(columns=columns, dtype=np.float64),
        )
        spatial_coef_table = spatial_coef_table.reindex(columns=columns)
        spatial_coef_table.loc[group] = spatial_coef
        tables["spatial_regression_coef"] = spatial_coef_table

    @staticmethod
    def _append_group_moderator_table(tables, group, group_moderator_table):
        """Append one group to the aggregate voxelwise moderator-effect table."""
        indexed_table = group_moderator_table.copy()
        indexed_table.index = pd.MultiIndex.from_product(
            [[group], indexed_table.index],
            names=["group", "moderator"],
        )
        if "voxelwise_moderator_effects_regression_coef" in tables:
            tables["voxelwise_moderator_effects_regression_coef"] = pd.concat(
                [tables["voxelwise_moderator_effects_regression_coef"], indexed_table]
            )
        else:
            tables["voxelwise_moderator_effects_regression_coef"] = indexed_table

    @staticmethod
    def _add_moderator_table(tables, group, moderator_names, moderator_coef):
        """Add voxelwise moderator-effect coefficient tables for one group."""
        group_moderator_table = pd.DataFrame(
            moderator_coef,
            index=moderator_names,
            columns=[f"basis_{i}" for i in range(moderator_coef.shape[1])],
        )
        tables[f"voxelwise_moderator_effect_regression_coef_group-{group}"] = group_moderator_table
        CBMREstimator._append_group_moderator_table(
            tables,
            group,
            group_moderator_table,
        )

    def _add_moderator_maps_and_tables(self, maps, tables, group, moderators, moderator_coef):
        """Add voxelwise moderator-effect maps and tables for one group."""
        bases = self.inputs_["coef_spline_bases"]
        for index, moderator_name in enumerate(self.moderators):
            moderator_effect = moderators[:, index : index + 1] @ moderator_coef[index : index + 1]
            maps[f"voxelwiseModeratorEffect_{moderator_name}_group-{group}"] = (
                moderator_effect @ bases.T
            ).mean(axis=0)
        maps[f"voxelwiseModeratorEffectTotal_group-{group}"] = (
            moderators @ moderator_coef @ bases.T
        ).mean(axis=0)
        self._add_moderator_table(tables, group, self.moderators, moderator_coef)

    def _add_approximate_results(self, maps, tables, group, moderators, coefficient):
        """Add maps and coefficient tables for one approximate-backend group."""
        bases = self.inputs_["coef_spline_bases"]
        n_bases = bases.shape[1]
        coefficient = coefficient.reshape((-1, n_bases))
        moderator_coef = coefficient[:-1]
        spatial_coef = coefficient[-1]
        maps[f"spatialIntensity_group-{group}"] = np.exp(bases @ spatial_coef)
        self._add_spatial_coef_table(tables, group, spatial_coef)
        if self.moderators:
            self._add_moderator_maps_and_tables(
                maps,
                tables,
                group,
                moderators,
                moderator_coef,
            )


class CBMRInference(object):
    """Statistical inference on fitted CBMR results.

    Notes
    -----
        This is the public inference entry point.

    .. versionadded:: 0.1.0

    Parameters
    ----------
    device: :obj:`string`, optional
        Device type ('cpu' or 'cuda') represents the device on which operations will be allocated.
        Default is 'cpu'.
    moderator_effect : {"voxelwise", "global"}, optional
        Inference parameterization to use. ``"voxelwise"`` uses the integrated voxelwise CBMR
        inference backend with sandwich or inverse-Fisher covariance estimates. ``"global"`` uses
        the standard CBMR inference backend. Default is ``"voxelwise"``.
    method : {"sandwich", "FI"}, optional
        Covariance estimator for CBMR inference. The voxelwise default is ``"sandwich"`` because
        it uses empirical residual variation to provide standard errors that are more robust to
        model misspecification, study-level clustering, and departures from idealized Poisson
        assumptions common in coordinate-based meta-analysis. The global default is ``"FI"`` to
        preserve the historical inverse-Fisher behavior. ``"FI"`` can be more efficient when the
        likelihood, mean-variance relationship, and independence assumptions are correctly
        specified, but may be too optimistic otherwise.
    sandwich_meat : {"cluster", "iid"}, optional
        Meat estimator for voxelwise sandwich covariance. ``"cluster"`` aggregates scores by
        experiment, while ``"iid"`` treats experiment-voxel observations independently. For global
        CBMR the marginal spatial and moderator score rows are already aggregated, so both options
        use the same row-wise meat.
    sandwich_correction : {None, "hc0", "hc1", "hc3"}, optional
        Optional HC-style finite-sample/leverage correction for sandwich covariance.
    ridge : :obj:`float`, optional
        Nonnegative ridge added before inverting the Fisher information matrix.
    mask : :obj:`str`, :class:`~nibabel.nifti1.Nifti1Image`, or Nilearn masker, optional
        Optional ROI mask used to restrict voxelwise inference outputs from a fitted result.
        If None, the fitted result's analysis mask is used.
    incidence_threshold : :obj:`float` or None, optional
        Drop voxels with empirical focus incidence less than or equal to this threshold when
        incidence information is available on the fitted result. Use None to keep all fitted
        voxels. Default is 0.001.
    """

    _valid_methods = ("sandwich", "FI")
    _valid_sandwich_meats = ("cluster", "iid")
    _valid_sandwich_corrections = (None, "hc0", "hc1", "hc3")
    _voxelwise_default_method = "sandwich"
    _global_default_method = "FI"

    def __init__(
        self,
        device="cpu",
        moderator_effect="voxelwise",
        method=None,
        sandwich_meat="cluster",
        sandwich_correction="hc3",
        ridge=1e-6,
        mask=None,
        incidence_threshold=DEFAULT_INCIDENCE_THRESHOLD,
    ):
        self.moderator_effect = self._normalize_moderator_effect(moderator_effect)
        self.device = device
        self.mask = mask
        self.masker = get_masker(mask) if mask is not None else None
        self.incidence_threshold = _validate_incidence_threshold(incidence_threshold)
        # device check
        if _uses_cuda(self.device) and not torch.cuda.is_available():
            LGR.debug("cuda not found, use device 'cpu'")
            self.device = "cpu"

        self.result = None
        self.groups = None
        self.moderators = None

        if self.moderator_effect == "global" and method is None:
            method = self._global_default_method
        elif method is None:
            method = self._voxelwise_default_method

        self.method = self._validate_method(method)
        self.sandwich_meat = self._validate_sandwich_meat(sandwich_meat)
        self.sandwich_correction = self._validate_sandwich_correction(sandwich_correction)
        if ridge < 0:
            raise ValueError("ridge must be nonnegative.")
        self.ridge = ridge

        self._reset_inference_caches()

    @classmethod
    def _normalize_moderator_effect(cls, moderator_effect):
        """Normalize the shared public moderator-effect selector."""
        return CBMREstimator._validate_moderator_effect(moderator_effect)

    @staticmethod
    def _validate_choice(value, choices, error_message, case_map=None):
        """Validate a string-like option against allowed choices."""
        if isinstance(value, str):
            normalized = value.lower()
            value = case_map.get(normalized, normalized) if case_map else normalized
        if value not in choices:
            raise ValueError(error_message)
        return value

    @classmethod
    def _validate_method(cls, method):
        """Validate and normalize an inference standard-error method."""
        return cls._validate_choice(
            method,
            cls._valid_methods,
            "method must be one of {'sandwich', 'FI'}.",
            case_map={"fi": "FI", "sandwich": "sandwich"},
        )

    @classmethod
    def _validate_sandwich_meat(cls, sandwich_meat):
        """Validate and normalize the sandwich meat estimator."""
        return cls._validate_choice(
            sandwich_meat,
            cls._valid_sandwich_meats,
            "sandwich_meat must be either 'cluster' or 'iid'.",
        )

    @classmethod
    def _validate_sandwich_correction(cls, sandwich_correction):
        """Validate and normalize the sandwich leverage correction."""
        return cls._validate_choice(
            sandwich_correction,
            cls._valid_sandwich_corrections,
            "sandwich_correction must be None, 'hc0', 'hc1', or 'hc3'.",
        )

    def _validate_global_sandwich_model(self):
        """Validate model support for global robust covariance."""
        model = getattr(self, "estimator", None)
        model = getattr(model, "model", None)
        if model is not None and not isinstance(model, models.PoissonEstimator):
            raise ValueError(
                "Global sandwich inference is currently supported only for "
                "model=models.PoissonEstimator."
            )

    def _check_fit(fn):
        """Check if CBMRInference instance has been fit."""

        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            if self.result is None:
                raise ValueError("CBMRInference instance has not been fit.")
            return fn(self, *args, **kwargs)

        return wrapper

    @staticmethod
    def _copy_result_for_inference(result):
        """Create an inference result copy without deep-copying all stored arrays and tables."""
        copied_result = copy.copy(result)
        copied_result.estimator = copy.deepcopy(result.estimator)
        copied_result.maps = {
            map_name: np.array(map_, copy=True) for map_name, map_ in result.maps.items()
        }
        copied_result.tables = {
            table_name: table.copy(deep=True) for table_name, table in result.tables.items()
        }
        copied_result.metadata = copy.deepcopy(result.metadata)
        return copied_result

    def _reset_inference_caches(self):
        """Reset cached inference/covariance intermediates for the fitted result."""
        self._group_log_intensity_cache = {}
        self._group_null_log_intensity_cache = {}
        if self.moderator_effect == "global":
            self._group_spatial_covariance_cache = {}
            self._moderator_covariance = None
            self._moderator_variance = None
            self._moderator_coef_table = None
        else:  # voxelwise
            self._group_covariance_cache = {}
            self._group_coefficient_cache = {}

    def _get_group_log_intensity(self, group):
        """Return cached group log-intensity values."""
        group_log_intensity = self._group_log_intensity_cache.get(group)
        if group_log_intensity is None:
            if self.moderator_effect == "global":
                group_log_intensity = np.log(self.result.maps[f"spatialIntensity_group-{group}"])
            else:  # voxelwise
                bases = self.estimator.inputs_["coef_spline_bases"]
                coefficient = self._get_group_coefficient_matrix(group)
                group_log_intensity = bases @ coefficient[-1]
            self._group_log_intensity_cache[group] = group_log_intensity
        return group_log_intensity

    def _get_group_null_log_intensity(self, group):
        """Return cached null baseline log-intensity for homogeneity testing."""
        group_null_log_intensity = self._group_null_log_intensity_cache.get(group)
        if group_null_log_intensity is None:
            if self.moderator_effect == "global":
                group_foci_per_voxel = self.estimator.inputs_["foci_per_voxel"][group]
                group_foci_per_experiment = self.estimator.inputs_["foci_per_experiment"][group]
                n_voxels, n_experiments = (
                    group_foci_per_voxel.shape[0],
                    group_foci_per_experiment.shape[0],
                )
                group_null_log_intensity = np.log(
                    np.sum(group_foci_per_voxel) / (n_voxels * n_experiments)
                )
            else:  # voxelwise
                foci = self.estimator.inputs_["foci_by_experiment_voxel"][group]
                total_foci = foci.sum()
                n_experiments, n_voxels = foci.shape
                group_null_log_intensity = np.log(max(float(total_foci), np.finfo(float).tiny))
                group_null_log_intensity -= np.log(n_experiments * n_voxels)
            self._group_null_log_intensity_cache[group] = group_null_log_intensity
        return group_null_log_intensity

    def _get_group_spatial_covariance(self, involved_groups):
        """Return cached spatial covariance for the involved groups."""
        group_key = (tuple(involved_groups), self.method, self.sandwich_correction, self.ridge)
        cov_spatial_coef = self._group_spatial_covariance_cache.get(group_key)
        if cov_spatial_coef is None:
            if self.method == "FI":
                moderators_by_group = (
                    self.estimator.inputs_["moderators_by_group"] if self.moderators else None
                )
                f_spatial_coef = self.estimator.model.fisher_info_multiple_group_spatial(
                    involved_groups,
                    self.estimator.inputs_["coef_spline_bases"],
                    moderators_by_group,
                    self.estimator.inputs_["foci_per_voxel"],
                    self.estimator.inputs_["foci_per_experiment"],
                )
                cov_spatial_coef = np.linalg.inv(f_spatial_coef)
            else:
                cov_spatial_coef = self._compute_global_spatial_sandwich_covariance(
                    involved_groups
                )
            self._group_spatial_covariance_cache[group_key] = cov_spatial_coef
        return cov_spatial_coef

    def _get_moderator_covariance(self):
        """Return cached moderator covariance and marginal variances."""
        if self._moderator_covariance is None:
            if self.method == "FI":
                moderators_by_group = (
                    self.estimator.inputs_["moderators_by_group"] if self.moderators else None
                )
                f_moderator_coef = self.estimator.model.fisher_info_multiple_group_moderator(
                    self.estimator.inputs_["coef_spline_bases"],
                    moderators_by_group,
                    self.estimator.inputs_["foci_per_voxel"],
                    self.estimator.inputs_["foci_per_experiment"],
                )
                self._moderator_covariance = np.linalg.inv(f_moderator_coef)
            else:
                self._moderator_covariance = self._compute_global_moderator_sandwich_covariance()
            self._moderator_variance = np.diag(self._moderator_covariance)
        return self._moderator_covariance, self._moderator_variance

    @staticmethod
    def _resample_mask_to_reference(mask_img, reference_img):
        """Return mask data resampled into the reference mask grid."""
        same_grid = mask_img.shape == reference_img.shape and np.allclose(
            mask_img.affine, reference_img.affine
        )
        if not same_grid:
            mask_img = resample_to_img(mask_img, reference_img, interpolation="nearest")
        return np.asanyarray(mask_img.dataobj).astype(bool)

    def _voxel_selection_for_result(self, result):
        """Return a boolean voxel selector for optional inference masking."""
        reference_img = result.masker.mask_img
        reference_data = np.asanyarray(reference_img.dataobj).astype(bool)
        n_voxels = int(reference_data.sum())
        keep_voxels = np.ones(n_voxels, dtype=bool)

        if self.masker is not None:
            mask_img = self.masker.mask_img
            requested_data = self._resample_mask_to_reference(mask_img, reference_img)
            keep_voxels &= requested_data.ravel()[np.flatnonzero(reference_data.ravel())]

        incidence_rate = result.estimator.inputs_.get("empirical_incidence_rate")
        if self.incidence_threshold is not None and incidence_rate is not None:
            incidence_rate = np.asarray(incidence_rate)
            if incidence_rate.shape[0] != n_voxels:
                raise ValueError("Stored empirical incidence rates do not match the result mask.")
            keep_voxels &= incidence_rate > self.incidence_threshold

        if not np.any(keep_voxels):
            raise ValueError(
                "No voxels survived CBMR inference masking. Lower incidence_threshold or "
                "provide a less restrictive mask."
            )
        return keep_voxels

    @staticmethod
    def _subset_sparse_columns(matrix, keep_voxels):
        """Subset sparse or dense experiment-by-voxel matrices by voxel columns."""
        if scipy.sparse.issparse(matrix):
            return _as_csr_matrix(matrix)[:, keep_voxels]
        return np.asarray(matrix)[:, keep_voxels]

    def _restrict_result_voxels(self, result):
        """Restrict a fitted result to the requested inference ROI/incidence set."""
        keep_voxels = self._voxel_selection_for_result(result)
        if np.all(keep_voxels):
            return result

        reference_img = result.masker.mask_img
        reference_data = np.asanyarray(reference_img.dataobj).astype(bool)
        kept_flat_indices = np.flatnonzero(reference_data.ravel())[keep_voxels]
        restricted_mask_data = np.zeros(reference_data.size, dtype=bool)
        restricted_mask_data[kept_flat_indices] = True
        restricted_mask_data = restricted_mask_data.reshape(reference_data.shape)
        restricted_mask_img = CBMREstimator._mask_image_from_data(
            restricted_mask_data, reference_img
        )
        restricted_masker = get_masker(restricted_mask_img)

        n_voxels = int(reference_data.sum())
        result.maps = {
            map_name: map_[keep_voxels] if map_.shape[0] == n_voxels else map_
            for map_name, map_ in result.maps.items()
        }
        result.masker = restricted_masker
        result.estimator.masker = restricted_masker
        result.estimator.inputs_["mask_img"] = restricted_mask_img
        result.estimator.inputs_["coef_spline_bases"] = result.estimator.inputs_[
            "coef_spline_bases"
        ][keep_voxels]

        incidence_rate = result.estimator.inputs_.get("empirical_incidence_rate")
        if incidence_rate is not None:
            result.estimator.inputs_["empirical_incidence_rate"] = np.asarray(incidence_rate)[
                keep_voxels
            ]

        for key in ("foci_by_experiment", "foci_by_experiment_voxel"):
            if key not in result.estimator.inputs_:
                continue
            for group, matrix in list(result.estimator.inputs_[key].items()):
                result.estimator.inputs_[key][group] = self._subset_sparse_columns(
                    matrix,
                    keep_voxels,
                )

        if "foci_by_experiment" in result.estimator.inputs_:
            for group, matrix in result.estimator.inputs_["foci_by_experiment"].items():
                result.estimator.inputs_["foci_per_voxel"][group] = np.asarray(
                    matrix.sum(axis=0),
                    dtype=np.int32,
                ).reshape((-1, 1))
                result.estimator.inputs_["foci_per_experiment"][group] = np.asarray(
                    matrix.sum(axis=1),
                    dtype=np.int32,
                ).reshape((-1, 1))
        return result

    def fit(self, result):
        """Fit CBMRInference instance.

        Parameters
        ----------
        result : :obj:`~nimare.meta.cbmr.CBMRResult`
            Fitted CBMR result containing regression coefficient tables and spatial intensity
            maps.
        """
        if self.moderator_effect == "voxelwise":
            if not isinstance(result, CBMRResult) or result.moderator_effect != "voxelwise":
                raise TypeError(
                    "CBMRInference.fit with moderator_effect='voxelwise' requires a "
                    "CBMRResult with moderator_effect='voxelwise'."
                )
        else:
            if not isinstance(result, CBMRResult) or result.moderator_effect != "global":
                raise TypeError(
                    "CBMRInference.fit with moderator_effect='global' requires a "
                    "CBMRResult with moderator_effect='global'."
                )

        self.result = self._copy_result_for_inference(result)
        self.result = self._restrict_result_voxels(self.result)
        self._reset_inference_caches()
        self.estimator = self.result.estimator
        self.groups = list(self.result.groups)
        self.moderators = list(self.result.moderators)

        if self.moderator_effect == "global":
            self.estimator.device = self.device
            self.estimator.model.device = self.device
            self.estimator.model.to(self.device)
            self.estimator.model._invalidate_tensor_inputs_cache()
            if self.method == "sandwich":
                self._validate_global_sandwich_model()
            if self.moderators:
                self._moderator_coef_table = (
                    self.result.tables["moderators_regression_coef"].to_numpy().T
                )

        self._create_regular_expressions()

        self.group_reference_dict = {
            group_name: index for index, group_name in enumerate(self.groups)
        }
        self.moderator_reference_dict = {}
        if self.moderators:
            self.moderator_reference_dict = {
                moderator_name: index for index, moderator_name in enumerate(self.moderators)
            }
            for moderator_name, index in self.moderator_reference_dict.items():
                LGR.info(f"{moderator_name} = index_{index}")

    @_check_fit
    def display(self):
        """Display Groups and Moderator names and order."""
        # visialize group/moderator names and their indices in contrast array
        LGR.info("Group Reference in contrast array")
        for group, index in self.group_reference_dict.items():
            LGR.info(f"{group} = index_{index}")
        if self.moderators:
            LGR.info("Moderator Reference in contrast array")
            for moderator, index in self.moderator_reference_dict.items():
                LGR.info(f"{moderator} = index_{index}")

    def _create_regular_expressions(self):
        """Create regular expressions for parsing contrast names.

        creates the following attributes:
        self.groups_regular_expression: regular expression for parsing group names
        self.moderators_regular_expression: regular expression for parsing moderator names

        usage:
        >>> self.groups_regular_expression.match("group1 - group2").groupdict()
        """
        operator = "(\\ ?(?P<operator>[+-]?)\\ ??)"
        for attr in ["groups", "moderators"]:
            groups = getattr(self, attr)
            if groups:
                first_group, second_group = [
                    f"(?P<{order}>{'|'.join([re.escape(g) for g in groups])})"
                    for order in ["first", "second"]
                ]
                reg_expr = re.compile(first_group + "(" + operator + second_group + "?)")
            else:
                reg_expr = None

            setattr(self, "{}_regular_expression".format(attr), reg_expr)

    def _contrast_source_context(self, source):
        """Return the fitted names, parser, and index lookup for one contrast source."""
        if source == "groups":
            return self.groups, self.groups_regular_expression, self.group_reference_dict
        if source == "moderators":
            return (
                self.moderators,
                self.moderators_regular_expression,
                self.moderator_reference_dict,
            )
        return None, None, None

    def _create_named_contrast_vector(self, contrast, source):
        """Create one named contrast vector for groups or moderators."""
        regressors, regular_expression, reference_dict = self._contrast_source_context(source)
        contrast_vector = np.zeros(len(regressors))
        contrast_match = regular_expression.match(contrast)
        if contrast_match is None:
            raise ValueError(f"{contrast} is not a valid contrast.")

        contrast_parts = contrast_match.groupdict()
        if all(contrast_parts.values()):
            contrast_vector[reference_dict[contrast_parts["first"]]] = 1
            contrast_vector[reference_dict[contrast_parts["second"]]] = int(
                contrast_parts["operator"] + "1"
            )
        else:
            contrast_vector[reference_dict[contrast]] = 1
        return contrast_vector

    @_check_fit
    def create_contrast(self, contrast_name, source="groups"):
        """Create contrast matrix for generalized hypothesis testing (GLH).

        Named group contrasts may refer to a single group (for a homogeneity test) or a pairwise
        comparison such as ``group_a-group_b``. Named moderator contrasts follow the same pattern.

        Parameters
        ----------
        contrast_name : :obj:`~string` or sequence of :obj:`~string`
            Name or names of the contrasts to construct.
        source : {"groups", "moderators"}, optional
            Whether to build group or moderator contrasts.
        """
        contrast_name = _normalize_named_pairwise_contrasts(contrast_name)
        contrast_matrix = {}
        if source in ("groups", "moderators"):
            for contrast in contrast_name:
                contrast_matrix[contrast] = self._create_named_contrast_vector(contrast, source)

        return contrast_matrix

    @_check_fit
    def transform(self, t_con_groups=None, t_con_moderators=None, method=None):
        """Conduct generalized linear hypothesis (GLH) testing on CBMR estimates.

        Parameters
        ----------
        t_con_groups : bool, dict, list, tuple, str, or None, optional
            Group homogeneity or comparison specification. Use ``False`` to skip group inference.
        t_con_moderators : bool, dict, list, tuple, str, or None, optional
            Moderator effect or comparison specification. Use ``False`` to skip moderator
            inference.
        method : {"sandwich", "FI"}, optional
            Covariance estimator for CBMR inference.
        """
        if method is not None:
            validated_method = self._validate_method(method)
            if self.moderator_effect == "global" and validated_method == "sandwich":
                self._validate_global_sandwich_model()
            if validated_method != self.method:
                self.method = validated_method
                self._reset_inference_caches()

        if self.moderator_effect == "voxelwise":
            self.result.metadata["voxelwise_cbmr_inference_method"] = self.method
            if self.method == "sandwich":
                self.result.metadata["voxelwise_cbmr_sandwich_meat"] = self.sandwich_meat
                self.result.metadata["voxelwise_cbmr_sandwich_correction"] = (
                    self.sandwich_correction
                )
            else:
                self.result.metadata.pop("voxelwise_cbmr_sandwich_meat", None)
                self.result.metadata.pop("voxelwise_cbmr_sandwich_correction", None)
        else:
            self.result.metadata["global_cbmr_inference_method"] = self.method
            if self.method == "sandwich":
                self.result.metadata["global_cbmr_sandwich_meat"] = self.sandwich_meat
                self.result.metadata["global_cbmr_sandwich_correction"] = self.sandwich_correction
            else:
                self.result.metadata.pop("global_cbmr_sandwich_meat", None)
                self.result.metadata.pop("global_cbmr_sandwich_correction", None)

        self.t_con_groups = t_con_groups
        self.t_con_moderators = t_con_moderators

        prepared_groups = self._preprocess_t_con_regressor(source="groups")
        if prepared_groups is not None:
            self.t_con_groups, self.t_con_groups_name = prepared_groups
            self._run_group_inference()
        prepared_moderators = self._preprocess_t_con_regressor(source="moderators")
        if prepared_moderators is not None:
            self.t_con_moderators, self.t_con_moderators_name = prepared_moderators
            self._run_moderator_inference()

        return self.result

    def fit_transform(self, result, t_con_groups=None, t_con_moderators=None, method=None):
        """Fit the inference engine and conduct GLH testing on CBMR estimates.

        Parameters
        ----------
        result : :obj:`~nimare.meta.cbmr.CBMRResult`
            Fitted CBMR result containing regression coefficient tables and spatial intensity
            maps.
        t_con_groups : bool, dict, list, tuple, str, or None, optional
            Group homogeneity or comparison specification. Use ``False`` to skip group inference.
        t_con_moderators : bool, dict, list, tuple, str, or None, optional
            Moderator effect or comparison specification. Use ``False`` to skip moderator
            inference.
        method : {"sandwich", "FI"}, optional
            Covariance estimator for voxelwise CBMR inference.
        """
        self.fit(result)
        return self.transform(t_con_groups, t_con_moderators, method=method)

    @_check_fit
    def _preprocess_t_con_regressor(self, source):
        """Preprocess contrast vector/matrix for GLH testing.

        Follow the steps below:
        (1) Remove groups not involved in contrast;
        (2) Standardize contrast matrix (row sum to 1);
        (3) Remove duplicate rows in contrast matrix.

        Parameters
        ----------
        source : :obj:`~string`
            Source of contrast matrix, either "groups" or "moderators".

        Returns
        -------
        t_con_regressor : :obj:`~list`
            Preprocessed contrast vector/matrix for inference on
            spatial intensity or moderators.
        t_con_regressor_name : :obj:`~list`
            Name of contrast vector/matrix for spatial intensity
        """
        t_con_regressor = getattr(self, f"t_con_{source}")
        regressors = getattr(self, f"{source}")
        if not regressors:
            if t_con_regressor in (None, False):
                return None
            raise ValueError(f"No {source} are available for CBMR inference.")

        if t_con_regressor is False:
            return None

        t_con_regressor, t_con_regressor_name = self._coerce_contrast_specification(
            t_con_regressor,
            regressors,
            source,
        )
        t_con_regressor = self._ensure_2d_contrasts(t_con_regressor)
        self._validate_contrast_shapes(t_con_regressor, len(regressors), source)
        t_con_regressor = self._remove_zero_rows(t_con_regressor, source)
        t_con_regressor = self._standardize_contrasts(t_con_regressor)
        t_con_regressor, t_con_regressor_name = self._deduplicate_contrasts(
            t_con_regressor,
            t_con_regressor_name,
        )

        return t_con_regressor, t_con_regressor_name

    def _coerce_contrast_specification(self, t_con_regressor, regressors, source):
        """Normalize user-provided contrast specifications into contrast matrices."""
        if t_con_regressor is None or t_con_regressor is True:
            default_contrasts = self.create_contrast(regressors, source=source)
            return list(default_contrasts.values()), list(default_contrasts.keys())

        if isinstance(t_con_regressor, dict):
            return list(t_con_regressor.values()), list(t_con_regressor.keys())

        if isinstance(t_con_regressor, str) or _is_named_pairwise_contrast(t_con_regressor):
            named_contrasts = self.create_contrast(t_con_regressor, source=source)
            return list(named_contrasts.values()), list(named_contrasts.keys())

        if isinstance(t_con_regressor, (list, np.ndarray)):
            if self._uses_named_contrast_list(t_con_regressor):
                named_contrasts = self.create_contrast(t_con_regressor, source=source)
                return list(named_contrasts.values()), list(named_contrasts.keys())

            contrast_matrices = [np.array(con_regressor) for con_regressor in t_con_regressor]
            for i, contrast in enumerate(contrast_matrices):
                self.result.metadata[f"GLH_{source}_{i}"] = contrast
            return contrast_matrices, None

        raise ValueError(
            f"Unsupported {source} contrast specification of type {type(t_con_regressor)}."
        )

    def _uses_named_contrast_list(self, t_con_regressor):
        """Return whether a contrast sequence is expressed entirely with names."""
        return (
            isinstance(t_con_regressor, list)
            and bool(t_con_regressor)
            and all(
                isinstance(con_regressor, str) or _is_named_pairwise_contrast(con_regressor)
                for con_regressor in t_con_regressor
            )
        )

    @staticmethod
    def _ensure_2d_contrasts(t_con_regressor):
        """Ensure each contrast array is two-dimensional."""
        return [
            con_regressor.reshape((1, -1)) if len(con_regressor.shape) == 1 else con_regressor
            for con_regressor in t_con_regressor
        ]

    @staticmethod
    def _validate_contrast_shapes(t_con_regressor, n_regressors, source):
        """Validate that each contrast has the expected regressor dimension."""
        wrong_shape = [con_regressor.shape[1] != n_regressors for con_regressor in t_con_regressor]
        if np.any(wrong_shape):
            wrong_con_regressor_idx = np.where(wrong_shape)[0].tolist()
            raise ValueError(
                f"""The shape of {str(wrong_con_regressor_idx)}th contrast vector(s) in contrast
                matrix doesn't match with {source}."""
            )

    @staticmethod
    def _remove_zero_rows(t_con_regressor, source):
        """Remove zero rows from each contrast matrix and reject empty contrasts."""
        con_regressor_zero_row = [
            np.where(np.sum(np.abs(con_regressor), axis=1) == 0)[0]
            for con_regressor in t_con_regressor
        ]
        if np.any([len(zero_row) > 0 for zero_row in con_regressor_zero_row]):
            t_con_regressor = [
                np.delete(t_con_regressor[i], con_regressor_zero_row[i], axis=0)
                for i in range(len(t_con_regressor))
            ]
            if np.any([con_regressor.shape[0] == 0 for con_regressor in t_con_regressor]):
                raise ValueError(
                    f"""One or more of contrast vector(s) in {source} contrast matrix are
                    all zeros."""
                )
        return t_con_regressor

    @staticmethod
    def _standardize_contrasts(t_con_regressor):
        """Scale each contrast row by its absolute row sum."""
        return [
            con_regressor / np.sum(np.abs(con_regressor), axis=1).reshape((-1, 1))
            for con_regressor in t_con_regressor
        ]

    @staticmethod
    def _deduplicate_contrasts(t_con_regressor, t_con_regressor_name):
        """Drop duplicate contrast matrices while preserving their original order."""
        deduplicated_contrasts = []
        deduplicated_names = [] if t_con_regressor_name is not None else None
        for index, con_regressor in enumerate(t_con_regressor):
            if any(np.array_equal(con_regressor, existing) for existing in deduplicated_contrasts):
                continue
            deduplicated_contrasts.append(con_regressor)
            if deduplicated_names is not None:
                deduplicated_names.append(t_con_regressor_name[index])
        return deduplicated_contrasts, deduplicated_names

    @_check_fit
    def _run_group_inference(self):
        """Evaluate all prepared group contrasts and write them into the result object."""
        for con_group_count, con_group in enumerate(self.t_con_groups):
            group_stats = self._evaluate_group_contrast(con_group)
            self._store_group_inference_result(con_group_count, group_stats)

    def _evaluate_group_contrast(self, con_group):
        """Compute statistics for one prepared group contrast."""
        X = self.estimator.inputs_["coef_spline_bases"]
        n_brain_voxel, spatial_coef_dim = X.shape
        involved_groups, simp_con_group, is_homogeneity_test = self._summarize_group_contrast(
            con_group
        )
        contrast_log_intensity = self._compute_group_contrast_log_intensity(
            simp_con_group,
            involved_groups,
            is_homogeneity_test,
        )
        if self.moderator_effect == "voxelwise":
            cov_spatial_coef = self._get_intercept_covariance_for_groups(involved_groups)
            spatial_coef_dim = None
            n_brain_voxel = None
        else:
            cov_spatial_coef = self._get_group_spatial_covariance(involved_groups)

        if con_group.shape[0] == 1:
            z_stats_spatial, p_vals_spatial = self._compute_group_wald_statistics(
                simp_con_group,
                involved_groups,
                cov_spatial_coef,
                contrast_log_intensity,
                X,
                spatial_coef_dim,
            )
            chi_sq_spatial = None
        else:
            chi_sq_spatial, z_stats_spatial, p_vals_spatial = self._compute_group_glh_statistics(
                simp_con_group,
                involved_groups,
                cov_spatial_coef,
                contrast_log_intensity,
                X,
                spatial_coef_dim=spatial_coef_dim,
                n_brain_voxel=n_brain_voxel,
                is_homogeneity_test=is_homogeneity_test,
            )

        return {
            "contrast_count": con_group.shape[0],
            "chi_square": chi_sq_spatial,
            "p": p_vals_spatial,
            "z": z_stats_spatial,
        }

    def _summarize_group_contrast(self, con_group):
        """Return the involved groups and simplified contrast matrix."""
        involved_index = np.where(np.any(con_group != 0, axis=0))[0].tolist()
        involved_groups = [self.groups[i] for i in involved_index]
        simp_con_group = con_group[:, ~np.all(con_group == 0, axis=0)]
        is_homogeneity_test = np.all(np.count_nonzero(con_group, axis=1) == 1)
        return involved_groups, simp_con_group, is_homogeneity_test

    def _compute_group_contrast_log_intensity(
        self,
        simp_con_group,
        involved_groups,
        is_homogeneity_test,
    ):
        """Project fitted group log-intensities through one contrast matrix."""
        involved_log_intensity_per_voxel = np.stack(
            [self._get_group_log_intensity(group) for group in involved_groups],
            axis=0,
        )
        if is_homogeneity_test:
            null_log_spatial_intensity = [
                self._get_group_null_log_intensity(group) for group in involved_groups
            ]
            involved_log_intensity_per_voxel -= np.asarray(null_log_spatial_intensity).reshape(
                -1, 1
            )
        return np.matmul(simp_con_group, involved_log_intensity_per_voxel)

    @staticmethod
    def _compute_group_wald_statistics(
        simp_con_group,
        involved_groups,
        cov_spatial_coef,
        contrast_log_intensity,
        X,
        spatial_coef_dim=None,
    ):
        """Compute one-contrast Wald statistics for group inference."""
        if spatial_coef_dim is None:
            spatial_coef_dim = X.shape[1]
        n_con_group_involved = len(involved_groups)
        var_log_intensity = []
        for k in range(n_con_group_involved):
            cov_spatial_coef_k = cov_spatial_coef[
                k * spatial_coef_dim : (k + 1) * spatial_coef_dim,
                k * spatial_coef_dim : (k + 1) * spatial_coef_dim,
            ]
            var_log_intensity_k = np.sum(np.multiply(X @ cov_spatial_coef_k, X), axis=1)
            var_log_intensity.append(var_log_intensity_k)
        var_log_intensity = np.stack(var_log_intensity, axis=0)
        involved_var_log_intensity = simp_con_group**2 @ var_log_intensity
        involved_std_log_intensity = np.sqrt(np.maximum(involved_var_log_intensity, 0.0))
        z_stats_spatial = contrast_log_intensity / np.where(
            involved_std_log_intensity > 0,
            involved_std_log_intensity,
            np.inf,
        )
        if n_con_group_involved == 1:
            p_vals_spatial = _normal_p_values(
                z_stats_spatial,
                two_sided=False,
                dtype=DEFAULT_FLOAT_DTYPE,
            )
        else:
            p_vals_spatial = _normal_p_values(z_stats_spatial, dtype=DEFAULT_FLOAT_DTYPE)
        return z_stats_spatial, p_vals_spatial

    def _compute_group_glh_statistics(
        self,
        simp_con_group,
        involved_groups,
        cov_spatial_coef,
        contrast_log_intensity,
        X,
        spatial_coef_dim=None,
        n_brain_voxel=None,
        is_homogeneity_test=False,
    ):
        """Compute multi-row GLH statistics for group inference."""
        n_con_group_involved = len(involved_groups)
        m = simp_con_group.shape[0]
        if spatial_coef_dim is None:
            spatial_coef_dim = X.shape[1]
        if n_brain_voxel is None:
            n_brain_voxel = X.shape[0]
        cov_spatial_coef = cov_spatial_coef.reshape(
            n_con_group_involved,
            spatial_coef_dim,
            n_con_group_involved,
            spatial_coef_dim,
        )
        cov_log_intensity = np.einsum(
            "vi,kisj,vj->ksv",
            X,
            cov_spatial_coef,
            X,
            optimize=True,
        )
        chi_sq_spatial = self._chi_square_log_intensity(
            m,
            n_brain_voxel,
            n_con_group_involved,
            simp_con_group,
            cov_log_intensity,
            contrast_log_intensity,
        )
        p_vals_spatial = _chi_square_p_values(chi_sq_spatial, df=m, dtype=DEFAULT_FLOAT_DTYPE)
        if is_homogeneity_test:
            z_stats_spatial = scipy.stats.norm.isf(p_vals_spatial)
            z_stats_spatial[z_stats_spatial < 0] = 0
        else:
            z_stats_spatial = _two_sided_z_from_p_values(p_vals_spatial)
            if simp_con_group.shape[0] == 1:
                z_stats_spatial *= np.sign(contrast_log_intensity.flatten())
        z_stats_spatial = np.clip(z_stats_spatial, a_min=-10, a_max=10)
        return chi_sq_spatial, z_stats_spatial, p_vals_spatial

    def _store_group_inference_result(self, con_group_count, group_stats):
        """Write one computed group-inference result into result maps."""
        contrast_name = self.t_con_groups_name[con_group_count] if self.t_con_groups_name else None
        if contrast_name:
            key_builder = lambda stat_name: f"{stat_name}_group-{contrast_name}"
        else:
            key_builder = lambda stat_name: f"{stat_name}_GLH_groups_{con_group_count}"
        self._store_stat_outputs(
            self.result.maps,
            group_stats,
            key_builder,
            chi_square_key="chiSquare",
        )

    @staticmethod
    def _store_stat_outputs(
        container,
        stats,
        key_builder,
        chi_square_key="chi_square",
        as_table=False,
    ):
        """Store chi-square, p, and z statistic outputs with shared naming logic."""
        stat_keys = []
        if stats["contrast_count"] > 1:
            stat_keys.append(("chi_square", chi_square_key))
        stat_keys.extend([("p", "p"), ("z", "z")])
        for stat_name, key_name in stat_keys:
            values = stats[stat_name]
            if as_table:
                values = pd.DataFrame(data=np.array(values), columns=[stat_name])
            container[key_builder(key_name)] = values

    @_check_fit
    def _glh_con_group(self):
        """Conduct GLH testing for group-wise spatial intensity estimation.

        GLH testing allows flexible hypothesis testings on spatial
        intensity, including group-wise spatial homogeneity test and
        group comparison test.
        """
        self._run_group_inference()

    @staticmethod
    def _chi_square_log_intensity(
        m=None,
        n_brain_voxel=None,
        n_con_group_involved=None,
        simp_con_group=None,
        cov_log_intensity=None,
        contrast_log_intensity=None,
        n_voxels=None,
        n_involved_groups=None,
    ):
        """
        Calculate chi-square statistics for GLH on group-wise log intensity function.

        It is an intermediate steps for GLH testings.

        Parameters
        ----------
        m : :obj:`int`
            Number of independent GLH tests.
        n_brain_voxel : :obj:`int`
            Number of voxels within the brain mask.
        n_con_group_involved : :obj:`int`
            Number of groups involved in the GLH test.
        simp_con_group : :obj:`numpy.ndarray`
            Simplified contrast matrix for the GLH test.
        cov_log_intensity : :obj:`numpy.ndarray`
            Covariance matrix of log intensity estimation.
        contrast_log_intensity : :obj:`numpy.ndarray`
            The product of contrast matrix and log intensity estimation.

        Returns
        -------
        chi_sq_spatial : :obj:`numpy.ndarray`
            Voxel-wise chi-square statistics for GLH tests on group-wise spatial
            intensity estimations.
        """
        if n_voxels is not None:
            n_brain_voxel = n_voxels
        if n_involved_groups is not None:
            n_con_group_involved = n_involved_groups
        if cov_log_intensity.ndim == 3:
            if cov_log_intensity.shape[:2] == (n_con_group_involved, n_con_group_involved):
                cov_by_voxel = np.moveaxis(cov_log_intensity, -1, 0)
            else:
                cov_by_voxel = cov_log_intensity
        else:
            cov_by_voxel = cov_log_intensity.reshape(
                n_con_group_involved, n_con_group_involved, n_brain_voxel
            ).transpose(2, 0, 1)
        contrast_cov = np.einsum(
            "ai,vij,bj->vab",
            simp_con_group,
            cov_by_voxel,
            simp_con_group,
            optimize=True,
        )
        contrast_values = contrast_log_intensity.T
        solved = np.linalg.solve(contrast_cov, contrast_values[..., np.newaxis])
        return np.einsum("va,va->v", contrast_values, solved[..., 0], optimize=True)

    @_check_fit
    def _run_moderator_inference(self):
        """Evaluate all prepared moderator contrasts and write them into the result object."""
        if self.moderator_effect == "voxelwise":
            for con_moderator_count, con_moderator in enumerate(self.t_con_moderators):
                for group in self.groups:
                    moderator_stats = self._evaluate_moderator_contrast(group, con_moderator)
                    self._store_moderator_inference_result(
                        con_moderator_count,
                        group,
                        moderator_stats,
                    )
            return

        cov_moderator_coef, var_moderator_coef = self._get_moderator_covariance()
        moderator_coef = self._moderator_coef_table
        for con_moderator_count, con_moderator in enumerate(self.t_con_moderators):
            moderator_stats = self._evaluate_moderator_contrast(
                con_moderator,
                cov_moderator_coef,
                var_moderator_coef,
                moderator_coef,
            )
            self._store_moderator_inference_result(con_moderator_count, moderator_stats)

    def _evaluate_moderator_contrast(
        self,
        con_moderator,
        cov_moderator_coef=None,
        var_moderator_coef=None,
        moderator_coef=None,
    ):
        """Compute statistics for one prepared moderator contrast."""
        if self.moderator_effect == "voxelwise":
            group = con_moderator
            con_moderator = cov_moderator_coef
            bases = self.estimator.inputs_["coef_spline_bases"]
            coefficient = self._get_group_coefficient_matrix(group)
            covariance = self._get_group_covariance(group)
            intercept_column = np.zeros((con_moderator.shape[0], 1))
            augmented_contrast = np.column_stack([con_moderator, intercept_column])
            return self._compute_spatial_coefficient_statistics(
                coefficient,
                covariance,
                augmented_contrast,
                bases,
            )

        m_con_moderator = con_moderator.shape[0]
        contrast_moderator_coef = np.matmul(con_moderator, moderator_coef)
        if m_con_moderator == 1:
            involved_var_moderator_coef = con_moderator**2 @ var_moderator_coef
            involved_std_moderator_coef = np.sqrt(involved_var_moderator_coef)
            z_stats_moderator = contrast_moderator_coef / involved_std_moderator_coef
            p_vals_moderator = _normal_p_values(z_stats_moderator)
            chi_sq_moderator = None
        else:
            contrast_covariance = con_moderator @ cov_moderator_coef @ con_moderator.T
            solved = np.linalg.solve(contrast_covariance, contrast_moderator_coef)
            chi_sq_moderator = contrast_moderator_coef.T @ solved
            p_vals_moderator = _chi_square_p_values(chi_sq_moderator, df=m_con_moderator)
            z_stats_moderator = _two_sided_z_from_p_values(p_vals_moderator)

        return {
            "contrast_count": m_con_moderator,
            "chi_square": chi_sq_moderator,
            "p": p_vals_moderator,
            "z": z_stats_moderator,
        }

    def _store_moderator_inference_result(
        self,
        con_moderator_count,
        moderator_stats_or_group,
        moderator_stats=None,
    ):
        """Write one computed moderator-inference result into result tables."""
        if moderator_stats is not None:
            group = moderator_stats_or_group
            contrast_name = (
                self.t_con_moderators_name[con_moderator_count]
                if self.t_con_moderators_name
                else None
            )
            if contrast_name:
                key_builder = (
                    lambda stat_name: f"{stat_name}_voxelwiseModeratorEffect_"
                    f"{contrast_name}_group-{group}"
                )
            else:
                key_builder = (
                    lambda stat_name: f"{stat_name}_GLH_voxelwiseModeratorEffects_"
                    f"{con_moderator_count}_group-{group}"
                )
            self._store_stat_outputs(
                self.result.maps,
                moderator_stats,
                key_builder,
                chi_square_key="chiSquare",
            )
            return

        moderator_stats = moderator_stats_or_group
        contrast_name = (
            self.t_con_moderators_name[con_moderator_count] if self.t_con_moderators_name else None
        )
        if contrast_name:
            key_builder = lambda stat_name: f"{stat_name}_{contrast_name}"
        else:
            key_builder = lambda stat_name: f"{stat_name}_GLH_moderators_{con_moderator_count}"
        self._store_stat_outputs(
            self.result.tables,
            moderator_stats,
            key_builder,
            as_table=True,
        )

    @_check_fit
    def _glh_con_moderator(self):
        """Conduct Generalized linear hypothesis (GLH) testing for moderators.

        GLH testing allows flexible hypothesis testings on regression
        coefficients of moderators, including testing for
        the existence of moderator effects and difference in moderator
        effects across multiple moderator effects.
        """
        self._run_moderator_inference()

    def _get_group_augmented_moderators(self, group):
        """Return the group moderator matrix with an intercept as the final column."""
        if self.moderators:
            moderators = np.asarray(self.estimator.inputs_["moderators_by_group"][group])
        else:
            n_experiments = self.estimator.inputs_["foci_by_experiment_voxel"][group].shape[0]
            moderators = np.empty((n_experiments, 0), dtype=np.float64)
        return np.column_stack([moderators, np.ones((moderators.shape[0], 1))])

    def _get_group_coefficient_matrix(self, group):
        """Return a ``(n_moderators + 1, n_bases)`` coefficient matrix."""
        coefficient = self._group_coefficient_cache.get(group)
        if coefficient is not None:
            return coefficient

        n_bases = self.estimator.inputs_["coef_spline_bases"].shape[1]
        if getattr(self.estimator, "voxelwise_coef", None) is not None:
            coefficient = np.asarray(self.estimator.voxelwise_coef[group]).reshape((-1, n_bases))
        elif getattr(self.estimator, "voxelwise_model", None) is not None:
            spatial_coef = (
                self.estimator.voxelwise_model.spatial_coef_linears[group]
                .weight.detach()
                .cpu()
                .numpy()
                .ravel()
            )
            if self.moderators:
                moderator_coef = (
                    self.estimator.voxelwise_model.moderator_coef_linears[group]
                    .weight.detach()
                    .cpu()
                    .numpy()
                )
                coefficient = np.vstack([moderator_coef, spatial_coef])
            else:
                coefficient = spatial_coef.reshape((1, -1))
        else:
            spatial_coef = self.result.tables["spatial_regression_coef"].loc[group].to_numpy()
            if self.moderators:
                moderator_coef = self.result.tables[
                    f"voxelwise_moderator_effect_regression_coef_group-{group}"
                ].to_numpy()
                coefficient = np.vstack([moderator_coef, spatial_coef])
            else:
                coefficient = spatial_coef.reshape((1, -1))

        self._group_coefficient_cache[group] = coefficient
        return coefficient

    @staticmethod
    def _compute_glm_sandwich_covariance(
        design,
        foci,
        mean,
        ridge=1e-6,
        correction="hc3",
    ):
        """Compute robust covariance for one log-link marginal Poisson GLM."""
        design = np.asarray(design, dtype=float)
        foci = np.asarray(foci, dtype=float).reshape(-1)
        mean = np.asarray(mean, dtype=float).reshape(-1)
        mean = np.nan_to_num(mean, nan=0.0, posinf=1e12, neginf=0.0)
        mean = np.clip(mean, 1e-12, 1e12)

        if design.ndim != 2:
            raise ValueError("design must be a two-dimensional array.")
        if design.shape[0] != foci.shape[0] or foci.shape != mean.shape:
            raise ValueError("design, foci, and mean must have matching rows.")

        fisher_info = design.T @ (design * mean[:, None])
        bread_inverse = CBMRInference._sandwich_bread_inverse(fisher_info, ridge)
        residuals = np.nan_to_num(foci - mean, nan=0.0, posinf=0.0, neginf=0.0)
        correction_factor = 1.0

        if correction == "hc1":
            n_observations, n_parameters = design.shape
            if n_observations <= n_parameters:
                raise ValueError(
                    "HC1 sandwich correction requires more observations than model columns. "
                    "Use sandwich_correction='hc0' or 'hc3' for this setting."
                )
            correction_factor = n_observations / float(n_observations - n_parameters)
        elif correction == "hc3":
            leverage = mean * np.einsum(
                "ij,jk,ik->i",
                design,
                bread_inverse,
                design,
                optimize=True,
            )
            leverage = np.nan_to_num(leverage, nan=0.0, posinf=1.0, neginf=0.0)
            leverage = np.clip(leverage, 0.0, 0.999)
            residuals = residuals / np.maximum(1.0 - leverage, 1e-6)

        meat_matrix = design.T @ (design * residuals[:, None] ** 2)
        covariance = correction_factor * bread_inverse @ meat_matrix @ bread_inverse
        return 0.5 * (covariance + covariance.T)

    def _global_group_moderator_sum(self, group):
        """Return the summed experiment-level moderator effect for one global-CBMR group."""
        if not self.moderators:
            return float(np.asarray(self.estimator.inputs_["foci_per_experiment"][group]).size)

        moderators = np.asarray(self.estimator.inputs_["moderators_by_group"][group], dtype=float)
        moderator_coef = np.asarray(self._moderator_coef_table, dtype=float)
        return float(np.exp(np.clip(moderators @ moderator_coef, -100, 100)).sum())

    def _compute_global_spatial_sandwich_covariance(self, involved_groups):
        """Compute block-diagonal robust covariance for global spatial coefficients."""
        bases = np.asarray(self.estimator.inputs_["coef_spline_bases"], dtype=float)
        n_bases = bases.shape[1]
        covariance = np.zeros((len(involved_groups) * n_bases, len(involved_groups) * n_bases))
        for group_index, group in enumerate(involved_groups):
            spatial_intensity = np.asarray(
                self.result.maps[f"spatialIntensity_group-{group}"],
                dtype=float,
            ).reshape(-1)
            mean = self._global_group_moderator_sum(group) * spatial_intensity
            group_covariance = self._compute_glm_sandwich_covariance(
                bases,
                self.estimator.inputs_["foci_per_voxel"][group],
                mean,
                ridge=self.ridge,
                correction=self.sandwich_correction,
            )
            group_slice = slice(group_index * n_bases, (group_index + 1) * n_bases)
            covariance[group_slice, group_slice] = group_covariance
        return covariance

    def _compute_global_moderator_sandwich_covariance(self):
        """Compute robust covariance for global experiment-level moderator coefficients."""
        if not self.moderators:
            return np.zeros((0, 0), dtype=np.float64)

        designs = []
        foci = []
        mean = []
        moderator_coef = np.asarray(self._moderator_coef_table, dtype=float)
        for group in self.groups:
            group_moderators = np.asarray(
                self.estimator.inputs_["moderators_by_group"][group],
                dtype=float,
            )
            spatial_sum = float(
                np.asarray(self.result.maps[f"spatialIntensity_group-{group}"], dtype=float).sum()
            )
            designs.append(group_moderators)
            foci.append(
                np.asarray(self.estimator.inputs_["foci_per_experiment"][group], dtype=float)
            )
            mean.append(
                spatial_sum * np.exp(np.clip(group_moderators @ moderator_coef, -100, 100))
            )

        return self._compute_glm_sandwich_covariance(
            np.vstack(designs),
            np.concatenate([group_foci.reshape(-1) for group_foci in foci]),
            np.concatenate([group_mean.reshape(-1) for group_mean in mean]),
            ridge=self.ridge,
            correction=self.sandwich_correction,
        )

    def _get_group_mean(self, group):
        """Return fitted Poisson mean for one group as experiments by voxels."""
        bases = self.estimator.inputs_["coef_spline_bases"]
        moderators = self._get_group_augmented_moderators(group)
        coefficient = self._get_group_coefficient_matrix(group)
        return np.exp(np.clip(moderators @ coefficient @ bases.T, -100, 100))

    @staticmethod
    def _compute_fisher_information(moderators, bases, mean):
        """Compute Fisher information for the Kronecker design without forming it."""
        n_moderators = moderators.shape[1]
        n_bases = bases.shape[1]
        fisher_info = np.zeros((n_moderators * n_bases, n_moderators * n_bases))
        moderator_weights = np.einsum("mr,ms,mv->rsv", moderators, moderators, mean)
        for row in range(n_moderators):
            row_slice = slice(row * n_bases, (row + 1) * n_bases)
            for col in range(row, n_moderators):
                col_slice = slice(col * n_bases, (col + 1) * n_bases)
                block = bases.T @ (bases * moderator_weights[row, col, :, None])
                fisher_info[row_slice, col_slice] = block
                if row != col:
                    fisher_info[col_slice, row_slice] = block.T
        return fisher_info

    @staticmethod
    def _sandwich_bread_inverse(fisher_info, ridge):
        """Invert the sandwich bread term with a ridge-stabilized pseudo-inverse."""
        regularized = fisher_info + ridge * np.eye(fisher_info.shape[0])
        bread_inverse = np.linalg.pinv(regularized)
        return 0.5 * (bread_inverse + bread_inverse.T)

    @staticmethod
    def _sandwich_meat_matrix(moderators, bases, residuals, meat):
        """Compute cluster- or iid-robust sandwich meat for the Kronecker design."""
        n_experiments, n_moderators = moderators.shape
        n_bases = bases.shape[1]
        n_parameters = n_moderators * n_bases
        if meat == "cluster":
            basis_residuals = bases.T @ residuals.T
            cluster_scores = np.zeros((n_parameters, n_experiments), dtype=float)
            for moderator_index in range(n_moderators):
                parameter_slice = slice(
                    moderator_index * n_bases,
                    (moderator_index + 1) * n_bases,
                )
                cluster_scores[parameter_slice, :] = (
                    basis_residuals * moderators[:, moderator_index][None, :]
                )
            return cluster_scores @ cluster_scores.T

        residual_weights = np.einsum("mr,ms,mv->rsv", moderators, moderators, residuals**2)
        meat_matrix = np.zeros((n_parameters, n_parameters), dtype=float)
        for row in range(n_moderators):
            row_slice = slice(row * n_bases, (row + 1) * n_bases)
            for col in range(row, n_moderators):
                col_slice = slice(col * n_bases, (col + 1) * n_bases)
                block = bases.T @ (bases * residual_weights[row, col, :, None])
                meat_matrix[row_slice, col_slice] = block
                if row != col:
                    meat_matrix[col_slice, row_slice] = block.T
        return meat_matrix

    @classmethod
    def _sandwich_meat_matrix_sparse_response(
        cls,
        moderators,
        bases,
        foci,
        mean,
        meat,
        residual_scale=None,
    ):
        """Compute sandwich meat from a sparse response without materializing dense foci."""
        foci = _as_csr_matrix(foci)
        if residual_scale is None:
            adjusted_mean = mean
            adjusted_foci = foci
        else:
            adjusted_mean = mean / residual_scale
            row_indices = _csr_row_indices(foci)
            adjusted_foci = scipy.sparse.csr_matrix(
                (
                    foci.data / residual_scale[row_indices, foci.indices],
                    foci.indices,
                    foci.indptr,
                ),
                shape=foci.shape,
            )

        if meat == "cluster":
            basis_residuals = (adjusted_foci @ bases).T - (bases.T @ adjusted_mean.T)
            n_bases = bases.shape[1]
            n_parameters = moderators.shape[1] * n_bases
            cluster_scores = np.zeros((n_parameters, moderators.shape[0]), dtype=float)
            for moderator_index in range(moderators.shape[1]):
                parameter_slice = slice(
                    moderator_index * n_bases,
                    (moderator_index + 1) * n_bases,
                )
                cluster_scores[parameter_slice, :] = (
                    basis_residuals * moderators[:, moderator_index][None, :]
                )
            return cluster_scores @ cluster_scores.T

        meat_matrix = cls._sandwich_meat_matrix(
            moderators,
            bases,
            adjusted_mean,
            meat="iid",
        )
        row_indices = _csr_row_indices(adjusted_foci)
        delta = (
            adjusted_foci.data**2
            - 2
            * adjusted_foci.data
            * adjusted_mean[
                row_indices,
                adjusted_foci.indices,
            ]
        )
        delta_matrix = scipy.sparse.csr_matrix(
            (delta, adjusted_foci.indices, adjusted_foci.indptr),
            shape=foci.shape,
        )
        n_bases = bases.shape[1]
        for row in range(moderators.shape[1]):
            row_slice = slice(row * n_bases, (row + 1) * n_bases)
            for col in range(row, moderators.shape[1]):
                col_slice = slice(col * n_bases, (col + 1) * n_bases)
                moderator_weight = moderators[:, row] * moderators[:, col]
                voxel_weight = np.asarray(
                    delta_matrix.multiply(moderator_weight[:, None]).sum(axis=0)
                ).ravel()
                block = bases.T @ (bases * voxel_weight[:, None])
                meat_matrix[row_slice, col_slice] += block
                if row != col:
                    meat_matrix[col_slice, row_slice] += block.T
        return meat_matrix

    @classmethod
    def _sandwich_correction_scale(cls, correction, bread_inverse, moderators, bases, mean):
        """Return residual scaling and finite-sample factor for sandwich corrections."""
        if correction is None or correction == "hc0":
            return None, 1.0

        n_experiments, n_moderators = moderators.shape
        if correction == "hc1":
            if n_experiments <= n_moderators:
                raise ValueError(
                    "HC1 sandwich correction requires more experiments than model columns. "
                    "Use sandwich_correction='hc0' or 'hc3' for this setting."
                )
            return None, n_experiments / float(n_experiments - n_moderators)

        n_bases = bases.shape[1]
        bread_inverse_blocks = bread_inverse.reshape(
            n_moderators,
            n_bases,
            n_moderators,
            n_bases,
        ).transpose(0, 2, 1, 3)
        leverage_basis = np.einsum(
            "vp,rspq,vq->rsv",
            bases,
            bread_inverse_blocks,
            bases,
            optimize=True,
        )
        leverage = mean * np.einsum(
            "mr,ms,rsv->mv",
            moderators,
            moderators,
            leverage_basis,
            optimize=True,
        )
        leverage = np.nan_to_num(leverage, nan=0.0, posinf=1.0, neginf=0.0)
        leverage = np.clip(leverage, 0.0, 0.999)
        return np.maximum(1.0 - leverage, 1e-6), 1.0

    @classmethod
    def _apply_sandwich_correction(
        cls,
        correction,
        bread_inverse,
        moderators,
        bases,
        mean,
        residuals,
    ):
        """Apply HC-style finite-sample corrections to sandwich residuals."""
        residual_scale, correction_factor = cls._sandwich_correction_scale(
            correction,
            bread_inverse,
            moderators,
            bases,
            mean,
        )
        if residual_scale is None:
            return residuals, correction_factor
        return residuals / residual_scale, correction_factor

    @classmethod
    def _compute_sandwich_covariance(
        cls,
        moderators,
        bases,
        foci,
        mean,
        ridge=1e-6,
        meat="cluster",
        correction="hc3",
    ):
        """Compute robust Poisson sandwich covariance for one voxelwise CBMR group.

        The voxelwise model has Kronecker rows ``moderator_experiment x basis_voxel``; this is
        distinct from the global-moderator covariance, where moderator effects are not spatially
        expanded over spline bases.
        """
        moderators = np.asarray(moderators, dtype=float)
        bases = np.asarray(bases, dtype=float)
        mean = np.asarray(mean, dtype=float)
        mean = np.nan_to_num(mean, nan=0.0, posinf=1e12, neginf=0.0)
        mean = np.clip(mean, 1e-12, 1e12)
        if scipy.sparse.issparse(foci):
            foci = _as_csr_matrix(foci)
            response_shape = foci.shape
        else:
            response_shape = np.asarray(foci).shape

        if response_shape != mean.shape:
            raise ValueError("foci and mean must have matching experiment-by-voxel shapes.")
        if response_shape != (moderators.shape[0], bases.shape[0]):
            raise ValueError("foci must have shape (n_experiments, n_voxels).")

        fisher_info = cls._compute_fisher_information(moderators, bases, mean)
        bread_inverse = cls._sandwich_bread_inverse(fisher_info, ridge)
        if scipy.sparse.issparse(foci):
            residual_scale, correction_factor = cls._sandwich_correction_scale(
                correction,
                bread_inverse,
                moderators,
                bases,
                mean,
            )
            meat_matrix = cls._sandwich_meat_matrix_sparse_response(
                moderators,
                bases,
                foci,
                mean,
                meat,
                residual_scale=residual_scale,
            )
        else:
            y = np.asarray(foci, dtype=float)
            residuals = np.nan_to_num(y - mean, nan=0.0, posinf=0.0, neginf=0.0)
            residuals, correction_factor = cls._apply_sandwich_correction(
                correction,
                bread_inverse,
                moderators,
                bases,
                mean,
                residuals,
            )
            meat_matrix = cls._sandwich_meat_matrix(moderators, bases, residuals, meat)
        covariance = correction_factor * bread_inverse @ meat_matrix @ bread_inverse
        return 0.5 * (covariance + covariance.T)

    def _get_group_covariance(self, group):
        """Return cached covariance of augmented coefficients for one group."""
        cache_key = (group, self.method, self.sandwich_meat, self.sandwich_correction, self.ridge)
        covariance = self._group_covariance_cache.get(cache_key)
        if covariance is not None:
            return covariance

        bases = self.estimator.inputs_["coef_spline_bases"]
        moderators = self._get_group_augmented_moderators(group)
        mean = self._get_group_mean(group)
        if self.method == "FI":
            fisher_info = self._compute_fisher_information(moderators, bases, mean)
            covariance = self._sandwich_bread_inverse(fisher_info, self.ridge)
        else:
            covariance = self._compute_sandwich_covariance(
                moderators,
                bases,
                self.estimator.inputs_["foci_by_experiment_voxel"][group],
                mean,
                ridge=self.ridge,
                meat=self.sandwich_meat,
                correction=self.sandwich_correction,
            )
        self._group_covariance_cache[cache_key] = covariance
        return covariance

    def _get_intercept_covariance_for_groups(self, involved_groups):
        """Return block-diagonal covariance for group spatial-intercept coefficients."""
        n_bases = self.estimator.inputs_["coef_spline_bases"].shape[1]
        covariance = np.zeros((len(involved_groups) * n_bases, len(involved_groups) * n_bases))
        for group_index, group in enumerate(involved_groups):
            group_covariance = self._get_group_covariance(group)
            block = group_covariance[-n_bases:, -n_bases:]
            group_slice = slice(group_index * n_bases, (group_index + 1) * n_bases)
            covariance[group_slice, group_slice] = block
        return covariance

    @staticmethod
    def _as_list(value, default, label):
        """Normalize a scalar-or-sequence selection to a list."""
        if value is None:
            return list(default)
        if isinstance(value, str):
            return [value]
        try:
            return list(value)
        except TypeError as exc:
            raise TypeError(f"{label} must be a string, sequence of strings, or None.") from exc

    @staticmethod
    def _validate_selected_names(selected, available, label):
        """Reject requested group or moderator names that are not available."""
        invalid = [name for name in selected if name not in available]
        if invalid:
            available_names = ", ".join(available)
            invalid_names = ", ".join(invalid)
            raise ValueError(
                f"Unknown {label}: {invalid_names}. Available {label}: {available_names}."
            )

    @staticmethod
    def _validate_unit_change(unit_change):
        """Return a finite scalar moderator-unit change."""
        unit_change = float(unit_change)
        if not np.isfinite(unit_change):
            raise ValueError("unit_change must be a finite scalar.")
        return unit_change

    @staticmethod
    def _unit_change_label(unit_change):
        """Format a unit change for stable map keys."""
        label = f"{unit_change:g}".replace("-", "neg").replace(".", "p")
        return label

    def _validate_voxelwise_moderator_diagnostic_inputs(self, moderators, groups, unit_change):
        """Validate voxelwise moderator-effect diagnostic selections."""
        if self.moderator_effect != "voxelwise":
            raise ValueError(
                "Voxelwise moderator-effect diagnostics require "
                "CBMRInference(moderator_effect='voxelwise')."
            )
        if not self.moderators:
            raise ValueError("This CBMR result does not include voxelwise moderators.")

        moderators = self._as_list(moderators, self.moderators, "moderators")
        groups = self._as_list(groups, self.groups, "groups")
        self._validate_selected_names(moderators, self.moderators, "moderators")
        self._validate_selected_names(groups, self.groups, "groups")
        unit_change = self._validate_unit_change(unit_change)
        return moderators, groups, unit_change

    def _compute_voxelwise_moderator_diagnostic_maps(self, moderator, group, unit_change):
        """Compute RI and ID maps for one moderator/group/unit-change combination."""
        bases = self.estimator.inputs_["coef_spline_bases"]
        coefficient = self._get_group_coefficient_matrix(group)
        moderator_index = self.moderator_reference_dict[moderator]
        moderator_log_intensity_change = bases @ coefficient[moderator_index]
        relative_intensity = np.exp(
            np.clip(unit_change * moderator_log_intensity_change, -100, 100)
        )
        baseline_intensity = np.asarray(
            self.result.maps[f"spatialIntensity_group-{group}"],
            dtype=float,
        )
        intensity_difference = baseline_intensity * (relative_intensity - 1.0)
        return relative_intensity, intensity_difference

    @staticmethod
    def _validate_id_threshold(id_threshold):
        """Return a finite non-negative ID threshold for ROI filtering."""
        if id_threshold is None:
            return None
        id_threshold = float(id_threshold)
        if not np.isfinite(id_threshold) or id_threshold < 0:
            raise ValueError("id_threshold must be None or a finite non-negative scalar.")
        return id_threshold

    @staticmethod
    def _mask_relative_intensity_to_id_roi(
        relative_intensity,
        intensity_difference,
        id_threshold=None,
    ):
        """Mask RI values to voxels whose absolute ID values pass a threshold."""
        id_threshold = CBMRInference._validate_id_threshold(id_threshold)
        relative_intensity = np.asarray(relative_intensity, dtype=float)
        intensity_difference = np.asarray(intensity_difference, dtype=float)
        if relative_intensity.shape != intensity_difference.shape:
            raise ValueError("relative_intensity and intensity_difference must have the same shape.")

        finite_difference = np.isfinite(intensity_difference)
        if not finite_difference.any():
            raise ValueError("intensity_difference must contain at least one finite value.")

        absolute_difference = np.abs(intensity_difference)
        if id_threshold is None:
            id_threshold = float(np.quantile(absolute_difference[finite_difference], 0.5))

        roi_mask = finite_difference & (absolute_difference >= id_threshold)
        masked_relative_intensity = np.where(
            roi_mask & np.isfinite(relative_intensity),
            relative_intensity,
            0.0,
        )
        return masked_relative_intensity, id_threshold

    @_check_fit
    def generate_voxelwise_moderator_effect_maps(
        self,
        moderators=None,
        groups=None,
        unit_change=1.0,
    ):
        """Generate diagnostic maps for voxelwise moderator effects.

        Parameters
        ----------
        moderators : :obj:`str`, sequence of :obj:`str`, or None, optional
            Spatially varying moderators to diagnose. If None, all fitted moderators are used.
        groups : :obj:`str`, sequence of :obj:`str`, or None, optional
            Groups for which to generate maps. If None, all fitted groups are used.
        unit_change : :obj:`float`, optional
            Moderator-unit increase to visualize. Default is 1.

        Returns
        -------
        :obj:`~nimare.meta.cbmr.CBMRResult`
            Result copy with added RI and ID maps. Relative Intensity (RI) is the multiplicative
            intensity ratio for ``unit_change`` moderator units, and Intensity Difference (ID) is
            the corresponding additive change from the fitted group baseline intensity.
        """
        moderators, groups, unit_change = self._validate_voxelwise_moderator_diagnostic_inputs(
            moderators,
            groups,
            unit_change,
        )
        unit_label = self._unit_change_label(unit_change)
        generated_maps = []

        for group in groups:
            for moderator in moderators:
                relative_intensity, intensity_difference = (
                    self._compute_voxelwise_moderator_diagnostic_maps(
                        moderator,
                        group,
                        unit_change,
                    )
                )
                relative_key = (
                    f"relativeIntensity_voxelwiseModeratorEffect_{moderator}_"
                    f"unit-{unit_label}_group-{group}"
                )
                difference_key = (
                    f"intensityDifference_voxelwiseModeratorEffect_{moderator}_"
                    f"unit-{unit_label}_group-{group}"
                )
                self.result.maps[relative_key] = relative_intensity
                self.result.maps[difference_key] = intensity_difference
                generated_maps.extend([relative_key, difference_key])

        self.result.metadata["voxelwise_moderator_effect_diagnostic_unit_change"] = unit_change
        self.result.metadata["voxelwise_moderator_effect_diagnostic_maps"] = tuple(generated_maps)
        return self.result

    @_check_fit
    def plot_voxelwise_moderator_effects(
        self,
        moderators=None,
        groups=None,
        unit_change=1.0,
        cut_coords=None,
        display_mode="ortho",
        id_threshold=None,
        threshold=None,
        figure=None,
        plot_kwargs=None,
    ):
        """Plot RI diagnostic maps within ID-defined regions of interest.

        Each requested moderator/group combination is plotted as one row. Intensity Difference
        (ID) values define the region of interest: voxels are retained when ``abs(ID)`` is greater
        than or equal to ``id_threshold``. If ``id_threshold`` is None, the 50% quantile of
        ``abs(ID)`` is used. Relative Intensity (RI) values are displayed only within that ROI.
        """
        id_threshold = self._validate_id_threshold(id_threshold)
        result = self.generate_voxelwise_moderator_effect_maps(
            moderators=moderators,
            groups=groups,
            unit_change=unit_change,
        )
        moderators, groups, unit_change = self._validate_voxelwise_moderator_diagnostic_inputs(
            moderators,
            groups,
            unit_change,
        )

        import matplotlib.pyplot as plt
        from nilearn.plotting import plot_stat_map

        plot_kwargs = {} if plot_kwargs is None else dict(plot_kwargs)
        unit_label = self._unit_change_label(unit_change)
        n_rows = len(moderators) * len(groups)
        if figure is None:
            figure = plt.figure(figsize=(5, 3.5 * n_rows))
        axes = figure.subplots(n_rows, 1, squeeze=False)
        plot_threshold = 1e-12 if threshold is None else threshold

        for row, (group, moderator) in enumerate(
            (group, moderator) for group in groups for moderator in moderators
        ):
            relative_key = (
                f"relativeIntensity_voxelwiseModeratorEffect_{moderator}_"
                f"unit-{unit_label}_group-{group}"
            )
            difference_key = (
                f"intensityDifference_voxelwiseModeratorEffect_{moderator}_"
                f"unit-{unit_label}_group-{group}"
            )
            masked_relative_intensity, resolved_id_threshold = (
                self._mask_relative_intensity_to_id_roi(
                    result.maps[relative_key],
                    result.maps[difference_key],
                    id_threshold=id_threshold,
                )
            )
            title_suffix = f"{moderator}, group={group}, unit={unit_change:g}"
            plot_stat_map(
                result.masker.inverse_transform(masked_relative_intensity),
                axes=axes[row, 0],
                figure=figure,
                cut_coords=cut_coords,
                display_mode=display_mode,
                threshold=plot_threshold,
                title=f"RI in ID ROI: {title_suffix}, |ID| >= {resolved_id_threshold:g}",
                **plot_kwargs,
            )

        return figure

    @staticmethod
    def _contrast_covariance_by_voxel(contrast, covariance, bases):
        """Project coefficient covariance into voxel-wise contrast covariance."""
        n_regressors = contrast.shape[1]
        n_bases = bases.shape[1]
        covariance_blocks = covariance.reshape(n_regressors, n_bases, n_regressors, n_bases)
        contrast_covariance_blocks = np.einsum(
            "sr,rpuq,tu->sptq",
            contrast,
            covariance_blocks,
            contrast,
            optimize=True,
        )
        return np.einsum(
            "np,sptq,nq->nst",
            bases,
            contrast_covariance_blocks,
            bases,
            optimize=True,
        )

    @classmethod
    def _compute_spatial_coefficient_statistics(cls, coefficient, covariance, contrast, bases):
        """Compute voxelwise Wald statistics for voxelwise moderator-effect coefficients."""
        contrast_eta = contrast @ coefficient @ bases.T
        contrast_cov = cls._contrast_covariance_by_voxel(contrast, covariance, bases)

        if contrast.shape[0] == 1:
            contrast_var = contrast_cov[:, 0, 0]
            contrast_std = np.sqrt(np.maximum(contrast_var, 0.0))
            z_stats = contrast_eta[0] / np.where(contrast_std > 0, contrast_std, np.inf)
            p_vals = _normal_p_values(z_stats)
            chi_square = None
        else:
            solved = np.linalg.solve(contrast_cov, contrast_eta.T[..., np.newaxis])
            chi_square = np.einsum("ns,ns->n", contrast_eta.T, solved[..., 0], optimize=True)
            p_vals = _chi_square_p_values(chi_square, df=contrast.shape[0])
            z_stats = _two_sided_z_from_p_values(p_vals)

        return {
            "contrast_count": contrast.shape[0],
            "chi_square": chi_square,
            "p": p_vals,
            "z": z_stats,
        }
