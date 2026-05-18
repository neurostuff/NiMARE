"""Spatially varying coordinate-based meta-regression estimators.

This module contains the estimator-facing API for spatially varying CBMR. The
low-level torch model lives in :mod:`nimare.meta.models`, and the numerical
solver used by the approximate backend lives in :mod:`nimare.meta.utils`. This
mirrors the organization of standard CBMR, where estimator logic and model
logic are kept separate.

Spatially varying CBMR extends standard CBMR by allowing experiment-level
moderator effects to vary smoothly across voxels. For experiment ``m`` and voxel
``v`` in group ``g``, the fitted linear predictor is

``log(lambda_gmv) = B(v) @ alpha_g + Z_gm @ beta_g @ B(v).T``

where ``B`` is the spatial B-spline basis matrix, ``alpha_g`` is the
group-specific spatial coefficient vector, ``Z_gm`` is one row of the
experiment-level moderator matrix, and ``beta_g`` is the moderator-by-basis
coefficient matrix. In this file, ``bases`` always refers to ``B`` and
``moderators`` always refers to ``Z``.
"""

import copy
import logging
import re
import time
from functools import wraps

import numpy as np
import pandas as pd
import scipy
import scipy.sparse

try:
    import torch  # type: ignore[import-not-found]
except ImportError as e:
    raise ImportError(
        "Torch is required to use `SpatialCBMR` classes. "
        "Install with `pip install 'nimare[cbmr]'`."
    ) from e

from nimare.meta import models
from nimare.meta.cbmr import (
    CBMREstimator,
    CBMRResult,
    _is_named_pairwise_contrast,
    _normalize_named_pairwise_contrasts,
)
from nimare.meta.utils import fit_spatial_cbmr_approximate
from nimare.utils import (
    DEFAULT_FLOAT_DTYPE,
    _clip_p_values,
    _minimum_positive_float,
    seed_torch,
)

LGR = logging.getLogger(__name__)


class SpatialCBMRResult(CBMRResult):
    """Meta-analytic result for spatially varying CBMR.

    This result class follows :class:`~nimare.meta.cbmr.CBMRResult`, but adds
    small convenience helpers for spatially varying moderator maps.

    Attributes
    ----------
    maps : :obj:`dict`
        Result maps. Spatially varying moderator maps are stored with names of
        the form ``"svModerator_{moderator}_group-{group}"``.
    tables : :obj:`dict`
        Result tables. As in standard CBMR, ``"spatial_regression_coef"``
        stores one row per group. Spatially varying moderator coefficients are
        stored in ``"sv_moderators_regression_coef"`` with a ``(group,
        moderator)`` index, with group-specific aliases named
        ``"sv_moderator_regression_coef_group-{group}"``.
    """

    def copy(self):
        """Return a copy of the spatially varying CBMR result object.

        This mirrors :meth:`~nimare.meta.cbmr.CBMRResult.copy`, but preserves
        the :class:`SpatialCBMRResult` subclass so that spatially varying helper
        methods remain available after copying.
        """
        new = SpatialCBMRResult(
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

    def get_inference(self, device=None, method="sandwich", **kwargs):
        """Return a fitted inference engine for spatially varying CBMR.

        Parameters
        ----------
        device : :obj:`str`, optional
            Compute device to record on the inference object. Spatially varying
            inference is currently NumPy/SciPy-based, but the argument is kept
            for consistency with :class:`~nimare.meta.cbmr.CBMRResult`.
        method : {"sandwich", "FI"}, optional
            Standard-error estimator used by the inference engine. The default
            sandwich estimator is robust to extra-Poisson experiment-level
            variation. ``"FI"`` uses the inverse Fisher information matrix.
        **kwargs
            Additional keyword arguments passed to :class:`SpatialCBMRInference`.

        Returns
        -------
        :class:`SpatialCBMRInference`
            Fitted spatially varying CBMR inference engine.
        """
        inference_device = device or getattr(self.estimator, "device", "cpu")
        inference = SpatialCBMRInference(device=inference_device, method=method, **kwargs)
        inference.fit(self)
        return inference

    def infer(
        self,
        group_contrasts=False,
        moderator_contrasts=False,
        device=None,
        method="sandwich",
        **kwargs,
    ):
        """Run spatially varying CBMR inference from a fitted result.

        Parameters
        ----------
        group_contrasts : bool, dict, list, tuple, str, or None, optional
            Group inference specification. Use ``False`` to skip group inference.
        moderator_contrasts : bool, dict, list, tuple, str, or None, optional
            Moderator inference specification. Use ``False`` to skip moderator inference.
        device : :obj:`str`, optional
            Compute device to record on the fitted inference object.
        method : {"sandwich", "FI"}, optional
            Standard-error estimator. Default is ``"sandwich"``. Use ``"FI"``
            for inverse-Fisher standard errors.
        **kwargs
            Additional keyword arguments passed to :class:`SpatialCBMRInference`.
        """
        inference = self.get_inference(device=device, method=method, **kwargs)
        return inference.transform(
            t_con_groups=group_contrasts,
            t_con_moderators=moderator_contrasts,
        )

    def test_groups(self, groups=None, device=None, method="sandwich", **kwargs):
        """Run one-group spatial homogeneity tests for the requested groups."""
        group_contrasts = list(self.groups) if groups is None else groups
        return self.infer(
            group_contrasts=group_contrasts,
            moderator_contrasts=False,
            device=device,
            method=method,
            **kwargs,
        )

    def compare_groups(self, contrasts, device=None, method="sandwich", **kwargs):
        """Run pairwise group-comparison tests using names or tuple shorthand."""
        group_contrasts = _normalize_named_pairwise_contrasts(contrasts)
        return self.infer(
            group_contrasts=group_contrasts,
            moderator_contrasts=False,
            device=device,
            method=method,
            **kwargs,
        )

    def test_moderators(self, moderators=None, device=None, method="sandwich", **kwargs):
        """Test whether requested spatially varying moderator effects differ from zero."""
        if not self.moderators:
            raise ValueError(
                "This spatial CBMR result does not include experiment-level moderators."
            )
        moderator_contrasts = list(self.moderators) if moderators is None else moderators
        return self.infer(
            group_contrasts=False,
            moderator_contrasts=moderator_contrasts,
            device=device,
            method=method,
            **kwargs,
        )

    def compare_moderators(self, contrasts, device=None, method="sandwich", **kwargs):
        """Run pairwise spatially varying moderator-comparison tests."""
        if not self.moderators:
            raise ValueError(
                "This spatial CBMR result does not include experiment-level moderators."
            )
        moderator_contrasts = _normalize_named_pairwise_contrasts(contrasts)
        return self.infer(
            group_contrasts=False,
            moderator_contrasts=moderator_contrasts,
            device=device,
            method=method,
            **kwargs,
        )

    @property
    def sv_moderator_names(self):
        """Return spatially varying moderator map names.

        Returns
        -------
        :obj:`tuple` of :obj:`str`
            Names of maps in ``self.maps`` that describe voxel-wise moderator
            effects.
        """
        return tuple(name for name in self.maps if name.startswith("svModerator_"))

    def describe_sv_effects(self):
        """Return simple summaries for spatially varying moderator maps.

        Returns
        -------
        :obj:`dict`
            Dictionary keyed by map name. Each value is a ``(min, mean, max)``
            tuple computed over in-mask voxels.
        """
        return {
            name: (float(values.min()), float(values.mean()), float(values.max()))
            for name, values in self.maps.items()
            if name.startswith("svModerator_")
        }


class _SpatialCBMRBase(CBMREstimator):
    """Shared preprocessing and result helpers for spatially varying CBMR estimators.

    The standard :class:`~nimare.meta.cbmr.CBMREstimator` preprocessing already
    creates group-wise foci summaries. Spatially varying CBMR additionally needs
    an experiment-by-voxel matrix, because the moderator design varies across
    experiments. This base class adds that matrix while preserving the standard
    CBMR input keys.
    """

    def _make_result(self, dataset, maps=None, tables=None, description=""):
        """Construct a spatially varying CBMR result object.

        Parameters
        ----------
        dataset : :obj:`~nimare.nimads.Studyset` or :obj:`~nimare.dataset.Dataset`
            Input collection used to determine the masker when no estimator
            masker was supplied.
        maps : :obj:`dict`, optional
            Voxel-wise output maps.
        tables : :obj:`dict`, optional
            Tabular output summaries.
        description : :obj:`str`, optional
            Human-readable fit description.

        Returns
        -------
        :class:`SpatialCBMRResult`
            Spatial CBMR result object.
        """
        masker = self.masker or dataset.masker
        return SpatialCBMRResult(
            self, mask=masker, maps=maps, tables=tables, description=description
        )

    def _build_experiment_group_inputs(self, dataset, filtered_coordinates, n_mask_voxels):
        """Extend CBMR preprocessing with experiment-by-voxel foci matrices.

        Parameters
        ----------
        dataset : :obj:`~nimare.nimads.Studyset` or :obj:`~nimare.dataset.Dataset`
            Input collection.
        filtered_coordinates : :class:`pandas.DataFrame`
            Coordinate table after removing foci outside the analysis mask. This
            table still contains ``"_cbmr_mask_index"``, which maps each focus
            to an in-mask voxel index.
        n_mask_voxels : :obj:`int`
            Number of voxels inside the analysis mask.

        Returns
        -------
        :obj:`dict`
            Standard CBMR group inputs plus ``"foci_by_experiment_voxel"``.
        """
        inputs = super()._build_experiment_group_inputs(
            dataset,
            filtered_coordinates,
            n_mask_voxels,
        )
        inputs["foci_by_experiment_voxel"] = self._build_group_foci_matrices(
            filtered_coordinates,
            inputs["ids_by_group"],
            n_mask_voxels,
        )
        return inputs

    @staticmethod
    def _build_group_foci_matrices(coordinates, ids_by_group, n_mask_voxels):
        """Return experiment-by-voxel foci count matrices for each group.

        Parameters
        ----------
        coordinates : :class:`pandas.DataFrame`
            Coordinate table with columns ``"id"`` and ``"_cbmr_mask_index"``.
            Each row is one focus.
        ids_by_group : :obj:`dict`
            Mapping from group name to ordered experiment IDs in that group.
            This order defines the row order of each output matrix.
        n_mask_voxels : :obj:`int`
            Number of in-mask voxels. This defines the column dimension.

        Returns
        -------
        :obj:`dict`
            Mapping group name to a sparse CSR matrix of shape
            ``(n_experiments_in_group, n_mask_voxels)``. Entry ``(i, j)`` is the
            number of foci from experiment ``i`` in masked voxel ``j``.
        """
        foci_by_experiment_voxel = {}
        if coordinates.empty:
            for group, group_ids in ids_by_group.items():
                foci_by_experiment_voxel[group] = scipy.sparse.csr_matrix(
                    (len(group_ids), n_mask_voxels),
                    dtype=np.float64,
                )
            return foci_by_experiment_voxel

        coordinates = coordinates.loc[:, ["id", "_cbmr_mask_index"]].copy()
        for group, group_ids in ids_by_group.items():
            id_to_row = {exp_id: i for i, exp_id in enumerate(group_ids)}
            group_coordinates = coordinates.loc[coordinates["id"].isin(group_ids)]
            rows = group_coordinates["id"].map(id_to_row).to_numpy(dtype=np.int64, copy=False)
            cols = group_coordinates["_cbmr_mask_index"].to_numpy(dtype=np.int64, copy=False)
            data = np.ones(group_coordinates.shape[0], dtype=np.float64)
            foci_by_experiment_voxel[group] = scipy.sparse.coo_matrix(
                (data, (rows, cols)),
                shape=(len(group_ids), n_mask_voxels),
                dtype=np.float64,
            ).tocsr()
        return foci_by_experiment_voxel

    def _as_torch_tensor(self, value):
        """Convert an array-like object to a float64 tensor on the estimator device.

        Parameters
        ----------
        value : array-like or sparse matrix
            Input data to convert. Sparse matrices are densified because the
            torch model currently expects dense tensors.

        Returns
        -------
        :class:`torch.Tensor`
            Float64 tensor allocated on ``self.device``.
        """
        if scipy.sparse.issparse(value):
            value = value.toarray()
        return torch.as_tensor(value, dtype=torch.float64, device=self.device)

    def _prepare_torch_inputs(self):
        """Return tensorized spatially varying CBMR inputs.

        Returns
        -------
        bases : :class:`torch.Tensor`
            Spatial B-spline basis matrix with shape ``(n_voxels, n_bases)``.
        moderators_by_group : :obj:`dict` or None
            Group-wise moderator matrices. Each array has shape
            ``(n_experiments_in_group, n_moderators)``.
        foci_by_experiment_voxel : :obj:`dict`
            Group-wise dense foci matrices. Each array has shape
            ``(n_experiments_in_group, n_voxels)``.
        """
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

    def _spatial_cbmr_description(self, backend):
        """Generate a NiMARE-style description for spatially varying CBMR.

        Parameters
        ----------
        backend : :obj:`str`
            Name of the backend used to fit the model.

        Returns
        -------
        :obj:`str`
            Human-readable result description.
        """
        if self.moderators:
            moderator_text = (
                "with spatially varying experiment-level moderator effects for "
                f"{', '.join(self.moderators)}"
            )
        else:
            moderator_text = "without experiment-level moderators"
        return (
            f"Spatially varying CBMR was performed with the {backend} backend "
            f"{moderator_text}. The model used {len(self.groups)} group(s), "
            f"spline spacing {self.spline_spacing}, and device {self.device}."
        )


class SpatialCBMREstimator(_SpatialCBMRBase):
    """Spatially varying coordinate-based meta-regression.

    This estimator follows :class:`~nimare.meta.cbmr.CBMREstimator` preprocessing,
    but replaces the standard CBMR model with a log-Poisson model in which moderator
    effects are smooth functions over voxels. Users can choose between the full torch
    L-BFGS backend and the approximate NumPy backend with ``backend``.

    Parameters
    ----------
    group_categories : :obj:`str` or :obj:`list` or None, optional
        Annotation column(s) used to categorize experiments into groups.
    moderators : :obj:`str` or :obj:`list` or None, optional
        Experiment-level moderators whose effects are allowed to vary spatially.
    mask : :obj:`str`, image-like, masker-like, or None, optional
        Mask used to define the analysis voxels.
    spline_spacing : :obj:`int`, optional
        Spacing for the B-spline basis used to smooth spatial effects.
    penalty : :obj:`bool`, optional
        Accepted for API consistency with :class:`~nimare.meta.cbmr.CBMREstimator`.
        Currently the spatially varying backend fits an unpenalized Poisson model.
    backend : {"full", "approximate"}, optional
        Fitting backend. ``"full"`` uses the torch L-BFGS model. ``"approximate"``
        uses the preconditioned-gradient solver in :mod:`nimare.meta.utils`.
        Default is ``"full"``.
    n_iter : :obj:`int`, optional
        Maximum number of iterations. For ``backend="full"``, this is the L-BFGS
        iteration limit. For ``backend="approximate"``, this is the approximate
        solver iteration limit.
    lr : :obj:`float`, optional
        L-BFGS learning rate.
    lr_decay : :obj:`float`, optional
        Learning-rate decay factor, retained for consistency with CBMR.
    tol : :obj:`float`, optional
        L-BFGS stopping tolerance.
    device : :obj:`str`, optional
        Torch device used for fitting.
    random_state : :obj:`int`, optional
        Random seed passed to torch before fitting.
    alpha : :obj:`float`, optional
        Step-size multiplier for the approximate backend. Ignored by the full backend.
    damping : :obj:`float`, optional
        Diagonal damping term for the approximate backend preconditioner. Ignored by the
        full backend.
    compute_nll : :obj:`bool`, optional
        Whether to compute and log the negative log-likelihood at each approximate
        backend iteration. Ignored by the full backend.
    **kwargs
        Additional keyword arguments passed to :class:`~nimare.estimator.Estimator`.

    Attributes
    ----------
    spatial_varying_model : :class:`~nimare.meta.models.SpatialCBMRModel`
        Fitted torch model. It stores group-level spatial coefficients in
        ``spatial_coef_linears`` and spatially varying moderator coefficients in
        ``moderator_coef_linears`` when ``backend="full"``.
    spatial_varying_coef : :obj:`dict`
        Mapping from group name to fitted augmented coefficient vectors when
        ``backend="approximate"``.
    """

    _valid_backends = ("full", "approximate")

    def __init__(
        self,
        group_categories=None,
        moderators=None,
        mask=None,
        spline_spacing=10,
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
        """Initialize the spatially varying CBMR estimator."""
        if backend not in self._valid_backends:
            raise ValueError(f"backend must be one of {self._valid_backends}. Got {backend!r}.")
        super().__init__(
            group_categories=group_categories,
            moderators=moderators,
            mask=mask,
            spline_spacing=spline_spacing,
            model=models.PoissonEstimator,
            penalty=penalty,
            n_iter=n_iter,
            lr=lr,
            lr_decay=lr_decay,
            tol=tol,
            device=device,
            random_state=random_state,
            **kwargs,
        )
        self.backend = backend
        self.alpha = alpha
        self.damping = damping
        self.compute_nll = compute_nll
        self.spatial_varying_model = None
        self.spatial_varying_coef = None

    def _fit(self, dataset):
        """Fit the spatially varying CBMR model with the selected backend.

        Parameters
        ----------
        dataset : :obj:`~nimare.nimads.Studyset` or :obj:`~nimare.dataset.Dataset`
            Collection to analyze. Preprocessing has already populated
            ``self.inputs_`` before this method is called.

        Returns
        -------
        maps : :obj:`dict`
            Voxel-wise result maps.
        tables : :obj:`dict`
            Coefficient summary tables.
        description : :obj:`str`
            Human-readable result description.
        """
        if self.backend == "approximate":
            return self._fit_approximate(dataset)
        return self._fit_full(dataset)

    def _fit_full(self, dataset):
        """Fit spatially varying CBMR with the full torch L-BFGS backend.

        Parameters
        ----------
        dataset : :obj:`~nimare.nimads.Studyset` or :obj:`~nimare.dataset.Dataset`
            Collection to analyze. This argument is accepted for consistency with
            :meth:`_fit`; all needed arrays are read from ``self.inputs_``.

        Returns
        -------
        maps : :obj:`dict`
            Voxel-wise result maps.
        tables : :obj:`dict`
            Coefficient summary tables.
        description : :obj:`str`
            Human-readable result description.
        """
        seed_torch(self.random_state, self.device)

        # bases: spatial B-spline design, shape (n_voxels, n_bases)
        # moderators_by_group: group -> moderator design matrix, shape (n_experiments, n_mods)
        # foci_by_experiment_voxel: group -> response matrix, shape (n_experiments, n_voxels)
        bases, moderators_by_group, foci_by_experiment_voxel = self._prepare_torch_inputs()
        moderators_coef_dim = len(self.moderators) if self.moderators else None
        self.spatial_varying_model = models.SpatialCBMRModel(
            groups=self.groups,
            spatial_coef_dim=self.inputs_["coef_spline_bases"].shape[1],
            moderators_coef_dim=moderators_coef_dim,
            device=self.device,
        )
        optimizer = torch.optim.LBFGS(
            params=self.spatial_varying_model.parameters(),
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
            loss = self.spatial_varying_model(
                bases,
                moderators_by_group,
                foci_by_experiment_voxel,
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        scheduler.step()
        LGR.info("Spatially varying CBMR optimisation took %.1f s.", time.time() - start_time)
        maps, tables = self._extract_torch_results(moderators_by_group)
        return maps, tables, self._spatial_cbmr_description("full L-BFGS")

    def _fit_approximate(self, dataset):
        """Fit spatially varying CBMR with the approximate backend.

        Parameters
        ----------
        dataset : :obj:`~nimare.nimads.Studyset` or :obj:`~nimare.dataset.Dataset`
            Collection to analyze. This argument is accepted for consistency with
            :meth:`_fit`; all needed arrays are read from ``self.inputs_``.

        Returns
        -------
        maps : :obj:`dict`
            Voxel-wise result maps.
        tables : :obj:`dict`
            Coefficient summary tables.
        description : :obj:`str`
            Human-readable result description.
        """
        # bases: spatial B-spline design, shape (n_voxels, n_bases)
        bases = self.inputs_["coef_spline_bases"]
        maps = {}
        tables = {}
        self.spatial_varying_coef = {}
        for group in self.groups:
            # foci: sparse response matrix, shape (n_experiments, n_voxels)
            foci = self.inputs_["foci_by_experiment_voxel"][group]
            if self.moderators:
                # moderators: experiment-level design, shape (n_experiments, n_moderators)
                moderators = self.inputs_["moderators_by_group"][group]
            else:
                moderators = np.empty((foci.shape[0], 0), dtype=np.float64)

            # The approximate solver expects a single design matrix. The final
            # intercept column estimates the group-level spatial intensity term.
            augmented_moderators = np.column_stack(
                [moderators, np.ones((foci.shape[0], 1), dtype=np.float64)]
            )
            coefficient = fit_spatial_cbmr_approximate(
                augmented_moderators,
                bases,
                foci,
                tol=self.tol,
                max_iter=self.n_iter,
                alpha=self.alpha,
                damping=self.damping,
                compute_nll=self.compute_nll,
            )
            self.spatial_varying_coef[group] = coefficient
            self._add_approximate_results(
                maps,
                tables,
                group,
                moderators,
                coefficient,
            )
        return maps, tables, self._spatial_cbmr_description("approximate")

    def _extract_torch_results(self, moderators_by_group):
        """Extract maps and coefficient tables from the fitted torch model.

        Parameters
        ----------
        moderators_by_group : :obj:`dict` or None
            Group-wise torch moderator matrices used during fitting.

        Returns
        -------
        maps : :obj:`dict`
            Spatial intensity and spatially varying moderator maps.
        tables : :obj:`dict`
            Spatial and moderator coefficient tables.
        """
        bases = self.inputs_["coef_spline_bases"]
        maps = {}
        tables = {}
        for group in self.groups:
            # spatial_coef: alpha_g, shape (n_bases,)
            spatial_coef = (
                self.spatial_varying_model.spatial_coef_linears[group]
                .weight.detach()
                .cpu()
                .numpy()
                .ravel()
            )
            maps[f"spatialIntensity_group-{group}"] = np.exp(bases @ spatial_coef)
            self._add_spatial_coef_table(tables, group, spatial_coef)
            if self.moderators:
                # moderator_coef: beta_g, shape (n_moderators, n_bases)
                moderator_coef = (
                    self.spatial_varying_model.moderator_coef_linears[group]
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
        """Add one group to the CBMR-style spatial coefficient table.

        Parameters
        ----------
        tables : :obj:`dict`
            Output table dictionary to update in place.
        group : :obj:`str`
            Group name.
        spatial_coef : :obj:`numpy.ndarray`
            Group-specific spatial coefficient vector with shape ``(n_bases,)``.
        """
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
        """Append one group to the aggregate spatially varying moderator table.

        Parameters
        ----------
        tables : :obj:`dict`
            Output table dictionary to update in place.
        group : :obj:`str`
            Group name.
        group_moderator_table : :class:`pandas.DataFrame`
            Group-specific moderator-by-basis coefficient table.
        """
        indexed_table = group_moderator_table.copy()
        indexed_table.index = pd.MultiIndex.from_product(
            [[group], indexed_table.index],
            names=["group", "moderator"],
        )
        if "sv_moderators_regression_coef" in tables:
            tables["sv_moderators_regression_coef"] = pd.concat(
                [tables["sv_moderators_regression_coef"], indexed_table]
            )
        else:
            tables["sv_moderators_regression_coef"] = indexed_table

    @staticmethod
    def _add_moderator_table(tables, group, moderator_names, moderator_coef):
        """Add spatially varying moderator coefficient tables for one group.

        Parameters
        ----------
        tables : :obj:`dict`
            Output table dictionary to update in place.
        group : :obj:`str`
            Group name.
        moderator_names : :obj:`list` of :obj:`str`
            Names of the experiment-level moderators.
        moderator_coef : :obj:`numpy.ndarray`
            Spatially varying moderator coefficient matrix with shape
            ``(n_moderators, n_bases)``.
        """
        group_moderator_table = pd.DataFrame(
            moderator_coef,
            index=moderator_names,
            columns=[f"basis_{i}" for i in range(moderator_coef.shape[1])],
        )
        tables[f"sv_moderator_regression_coef_group-{group}"] = group_moderator_table
        SpatialCBMREstimator._append_group_moderator_table(
            tables,
            group,
            group_moderator_table,
        )

    def _add_moderator_maps_and_tables(self, maps, tables, group, moderators, moderator_coef):
        """Add spatially varying moderator maps and tables for one group.

        Parameters
        ----------
        maps : :obj:`dict`
            Output map dictionary to update in place.
        tables : :obj:`dict`
            Output table dictionary to update in place.
        group : :obj:`str`
            Group name.
        moderators : :obj:`numpy.ndarray`
            Moderator design matrix with shape ``(n_experiments, n_moderators)``.
        moderator_coef : :obj:`numpy.ndarray`
            Fitted coefficient matrix with shape ``(n_moderators, n_bases)``.
        """
        bases = self.inputs_["coef_spline_bases"]
        for index, moderator_name in enumerate(self.moderators):
            # moderator_effect: experiment-by-basis contribution for one moderator.
            moderator_effect = moderators[:, index : index + 1] @ moderator_coef[index : index + 1]
            maps[f"svModerator_{moderator_name}_group-{group}"] = (
                moderator_effect @ bases.T
            ).mean(axis=0)
        maps[f"svModeratorTotal_group-{group}"] = (moderators @ moderator_coef @ bases.T).mean(
            axis=0
        )
        self._add_moderator_table(tables, group, self.moderators, moderator_coef)

    def _add_approximate_results(self, maps, tables, group, moderators, coefficient):
        """Add maps and coefficient tables for one approximate-backend group.

        Parameters
        ----------
        maps : :obj:`dict`
            Output map dictionary to update in place.
        tables : :obj:`dict`
            Output table dictionary to update in place.
        group : :obj:`str`
            Group name.
        moderators : :obj:`numpy.ndarray`
            Moderator design matrix with shape ``(n_experiments, n_moderators)``.
        coefficient : :obj:`numpy.ndarray`
            Flattened augmented coefficient vector returned by
            :func:`~nimare.meta.utils.fit_spatial_cbmr_approximate`.
        """
        bases = self.inputs_["coef_spline_bases"]
        n_bases = bases.shape[1]
        coefficient = coefficient.reshape((-1, n_bases))
        # coefficient[:-1]: spatially varying moderator coefficients, shape
        # (n_moderators, n_bases). coefficient[-1]: spatial intercept alpha_g.
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


class SpatialCBMRInference(object):
    """Statistical inference on fitted spatially varying CBMR results.

    This class mirrors the public interface of
    :class:`~nimare.meta.cbmr.CBMRInference`: call :meth:`fit` with a fitted
    :class:`SpatialCBMRResult`, then call :meth:`transform` with group and/or
    moderator contrasts. By default, standard errors are estimated with a
    robust Poisson sandwich covariance, following the spatial GLM inference
    strategy in ``_UKBInferenceBackend``. Users may request inverse-Fisher
    standard errors with ``method="FI"``. The key design matrix is never
    materialized explicitly; instead, covariance terms and contrast variances
    are computed from the Kronecker structure induced by the experiment-level
    design matrix ``Z`` and the spatial basis matrix ``B``.

    Parameters
    ----------
    device : :obj:`str`, optional
        Device label retained for API consistency with CBMR inference. The
        current implementation performs inference with NumPy/SciPy arrays.
        Default is ``"cpu"``.
    method : {"sandwich", "FI"}, optional
        Standard-error estimator. ``"sandwich"`` is the default and uses a
        robust Poisson sandwich covariance for the Kronecker design, following
        the implementation strategy used by ``_UKBInferenceBackend``.
        ``"FI"`` uses the inverse Fisher information matrix.
    sandwich_meat : {"cluster", "iid"}, optional
        Meat term used by the sandwich covariance. ``"cluster"`` aggregates
        scores by experiment, while ``"iid"`` treats experiment-by-voxel cells
        as independent. Default is ``"cluster"``.
    sandwich_correction : {None, "hc0", "hc1", "hc3"}, optional
        Small-sample leverage correction for sandwich residuals. Default is
        ``"hc3"``.
    ridge : :obj:`float`, optional
        Nonnegative diagonal ridge added to Fisher-information matrices before
        inversion. This stabilizes inference when the spatial basis design is
        nearly singular. Default is ``1e-6``.

    Notes
    -----
    Group inference tests baseline spatial intensity coefficients, i.e., the
    spatial intercept row of the augmented coefficient matrix. Moderator
    inference tests voxel-wise spatially varying moderator effects within each
    group. Inference maps are appended to a copy of the fitted result, leaving
    the input result unchanged.
    """

    _valid_methods = ("sandwich", "FI")
    _valid_sandwich_meats = ("cluster", "iid")
    _valid_sandwich_corrections = (None, "hc0", "hc1", "hc3")

    def __init__(
        self,
        device="cpu",
        method="sandwich",
        sandwich_meat="cluster",
        sandwich_correction="hc3",
        ridge=1e-6,
    ):
        """Initialize the spatially varying CBMR inference engine."""
        self.device = device
        self.method = self._validate_method(method)
        self.sandwich_meat = self._validate_sandwich_meat(sandwich_meat)
        self.sandwich_correction = self._validate_sandwich_correction(sandwich_correction)
        if ridge < 0:
            raise ValueError("ridge must be nonnegative.")
        self.ridge = ridge
        self.result = None
        self.estimator = None
        self.groups = None
        self.moderators = None
        self._reset_inference_caches()

    @classmethod
    def _validate_method(cls, method):
        """Validate and normalize an inference standard-error method."""
        if isinstance(method, str):
            if method.lower() == "fi":
                return "FI"
            if method.lower() == "sandwich":
                return "sandwich"
        raise ValueError("method must be one of {'sandwich', 'FI'}.")

    @classmethod
    def _validate_sandwich_meat(cls, sandwich_meat):
        """Validate and normalize the sandwich meat estimator."""
        if isinstance(sandwich_meat, str):
            sandwich_meat = sandwich_meat.lower()
        if sandwich_meat not in cls._valid_sandwich_meats:
            raise ValueError("sandwich_meat must be either 'cluster' or 'iid'.")
        return sandwich_meat

    @classmethod
    def _validate_sandwich_correction(cls, sandwich_correction):
        """Validate and normalize the sandwich leverage correction."""
        if isinstance(sandwich_correction, str):
            sandwich_correction = sandwich_correction.lower()
        if sandwich_correction not in cls._valid_sandwich_corrections:
            raise ValueError("sandwich_correction must be None, 'hc0', 'hc1', or 'hc3'.")
        return sandwich_correction

    def _check_fit(fn):
        """Check if SpatialCBMRInference instance has been fit."""

        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            if self.result is None:
                raise ValueError("SpatialCBMRInference instance has not been fit.")
            return fn(self, *args, **kwargs)

        return wrapper

    @staticmethod
    def _copy_result_for_inference(result):
        """Create an inference result copy without mutating fitted outputs."""
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
        """Reset cached covariance and coefficient arrays for the fitted result."""
        self._group_covariance_cache = {}
        self._group_coefficient_cache = {}
        self._group_log_intensity_cache = {}
        self._group_null_log_intensity_cache = {}

    def fit(self, result):
        """Fit the inference engine to a spatially varying CBMR result.

        Parameters
        ----------
        result : :class:`SpatialCBMRResult`
            Fitted spatially varying CBMR result containing coefficient tables,
            fitted maps, and the estimator inputs needed to reconstruct Fisher
            information.
        """
        if not isinstance(result, SpatialCBMRResult):
            raise TypeError("SpatialCBMRInference.fit requires a SpatialCBMRResult.")

        self.result = self._copy_result_for_inference(result)
        self._reset_inference_caches()
        self.estimator = self.result.estimator
        self.groups = list(self.result.groups)
        self.moderators = list(self.result.moderators)
        self.create_regular_expressions()
        self.group_reference_dict = {
            group_name: index for index, group_name in enumerate(self.groups)
        }
        self.moderator_reference_dict = {
            moderator_name: index for index, moderator_name in enumerate(self.moderators)
        }

    @_check_fit
    def display(self):
        """Log groups and moderators with their contrast-vector indices."""
        LGR.info("Group Reference in contrast array")
        for group, index in self.group_reference_dict.items():
            LGR.info("%s = index_%d", group, index)
        if self.moderators:
            LGR.info("Moderator Reference in contrast array")
            for moderator, index in self.moderator_reference_dict.items():
                LGR.info("%s = index_%d", moderator, index)

    def create_regular_expressions(self):
        """Create regular expressions for parsing named contrasts."""
        operator = "(\\ ?(?P<operator>[+-]?)\\ ??)"
        for attr in ["groups", "moderators"]:
            regressors = getattr(self, attr)
            if regressors:
                first_regressor, second_regressor = [
                    f"(?P<{order}>{'|'.join([re.escape(regressor) for regressor in regressors])})"
                    for order in ["first", "second"]
                ]
                reg_expr = re.compile(first_regressor + "(" + operator + second_regressor + "?)")
            else:
                reg_expr = None
            setattr(self, f"{attr}_regular_expression", reg_expr)

    @_check_fit
    def create_contrast(self, contrast_name, source="groups"):
        """Create a named contrast matrix for spatial CBMR inference.

        Parameters
        ----------
        contrast_name : :obj:`str`, :obj:`tuple`, or sequence
            Contrast name(s). Tuple shorthand such as ``("A", "B")`` is
            converted to the pairwise contrast ``"A-B"``.
        source : {"groups", "moderators"}, optional
            Whether to build contrasts over groups or moderators.

        Returns
        -------
        :obj:`dict`
            Mapping contrast names to one-dimensional contrast vectors.
        """
        contrast_name = _normalize_named_pairwise_contrasts(contrast_name)
        contrast_matrix = {}
        if source not in {"groups", "moderators"}:
            raise ValueError("source must be either 'groups' or 'moderators'.")

        regressors = getattr(self, source)
        reference = getattr(
            self, f"{source[:-1] if source.endswith('s') else source}_reference_dict"
        )
        regular_expression = getattr(self, f"{source}_regular_expression")
        for contrast in contrast_name:
            contrast_vector = np.zeros(len(regressors))
            contrast_match = regular_expression.match(contrast)
            if contrast_match is None:
                raise ValueError(f"{contrast} is not a valid contrast.")
            contrast_parts = contrast_match.groupdict()
            if all(contrast_parts.values()):
                contrast_vector[reference[contrast_parts["first"]]] = 1
                contrast_vector[reference[contrast_parts["second"]]] = int(
                    contrast_match["operator"] + "1"
                )
            else:
                contrast_vector[reference[contrast]] = 1
            contrast_matrix[contrast] = contrast_vector

        return contrast_matrix

    @_check_fit
    def transform(self, t_con_groups=None, t_con_moderators=None, method=None):
        """Run spatially varying CBMR inference.

        Parameters
        ----------
        t_con_groups : bool, dict, list, tuple, str, or None, optional
            Group inference specification. Use ``None`` or ``True`` to test all
            groups, ``False`` to skip group inference, named contrasts such as
            ``"group_a-group_b"`` or ``("group_a", "group_b")`` for pairwise
            tests, a dict mapping names to contrast arrays, or raw contrast
            arrays.
        t_con_moderators : bool, dict, list, tuple, str, or None, optional
            Moderator inference specification with the same accepted forms as
            ``t_con_groups``. Spatially varying moderator tests are computed
            separately within each fitted group.
        method : {"sandwich", "FI"}, optional
            Standard-error estimator to use for this transform call. If None,
            uses the method recorded on the inference object. The default object
            method is ``"sandwich"``.

        Returns
        -------
        :class:`SpatialCBMRResult`
            Copy of the fitted result with inference maps appended.
        """
        if method is not None:
            validated_method = self._validate_method(method)
            if validated_method != self.method:
                self.method = validated_method
                self._reset_inference_caches()
        self.result.metadata["spatial_cbmr_inference_method"] = self.method
        if self.method == "sandwich":
            self.result.metadata["spatial_cbmr_sandwich_meat"] = self.sandwich_meat
            self.result.metadata["spatial_cbmr_sandwich_correction"] = self.sandwich_correction
        else:
            self.result.metadata.pop("spatial_cbmr_sandwich_meat", None)
            self.result.metadata.pop("spatial_cbmr_sandwich_correction", None)

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
        """Fit and transform in one call."""
        self.fit(result)
        return self.transform(t_con_groups, t_con_moderators, method=method)

    @_check_fit
    def _preprocess_t_con_regressor(self, source):
        """Normalize and validate contrast specifications."""
        t_con_regressor = getattr(self, f"t_con_{source}")
        regressors = getattr(self, source)
        if not regressors:
            if t_con_regressor in (None, False):
                return None
            raise ValueError(f"No {source} are available for spatial CBMR inference.")

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
        """Normalize user-provided contrast specifications into arrays."""
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

    @staticmethod
    def _uses_named_contrast_list(t_con_regressor):
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
                f"The shape of {wrong_con_regressor_idx}th contrast vector(s) in contrast "
                f"matrix doesn't match with {source}."
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
                    f"One or more contrast vector(s) in {source} contrast matrix are all zeros."
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
        if getattr(self.estimator, "spatial_varying_coef", None) is not None:
            coefficient = np.asarray(self.estimator.spatial_varying_coef[group]).reshape(
                (-1, n_bases)
            )
        elif getattr(self.estimator, "spatial_varying_model", None) is not None:
            spatial_coef = (
                self.estimator.spatial_varying_model.spatial_coef_linears[group]
                .weight.detach()
                .cpu()
                .numpy()
                .ravel()
            )
            if self.moderators:
                moderator_coef = (
                    self.estimator.spatial_varying_model.moderator_coef_linears[group]
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
                    f"sv_moderator_regression_coef_group-{group}"
                ].to_numpy()
                coefficient = np.vstack([moderator_coef, spatial_coef])
            else:
                coefficient = spatial_coef.reshape((1, -1))

        self._group_coefficient_cache[group] = coefficient
        return coefficient

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
    def _as_dense_response(foci):
        """Return a dense experiment-by-voxel response matrix."""
        if scipy.sparse.issparse(foci):
            return foci.toarray()
        return np.asarray(foci, dtype=float)

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
        if correction is None or correction == "hc0":
            return residuals, 1.0

        n_experiments, n_moderators = moderators.shape
        if correction == "hc1":
            if n_experiments <= n_moderators:
                raise ValueError(
                    "HC1 sandwich correction requires more experiments than model columns. "
                    "Use sandwich_correction='hc0' or 'hc3' for this setting."
                )
            return residuals, n_experiments / float(n_experiments - n_moderators)

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
        return residuals / np.maximum(1.0 - leverage, 1e-6), 1.0

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
        """Compute robust Poisson sandwich covariance for one spatial CBMR group.

        The implementation follows ``_UKBInferenceBackend``: the bread is the
        Fisher information of the Kronecker design, while the meat is computed
        either from experiment-clustered scores or iid experiment-by-voxel
        scores. The full Kronecker design is never materialized.
        """
        moderators = np.asarray(moderators, dtype=float)
        bases = np.asarray(bases, dtype=float)
        y = cls._as_dense_response(foci)
        mean = np.asarray(mean, dtype=float)
        mean = np.nan_to_num(mean, nan=0.0, posinf=1e12, neginf=0.0)
        mean = np.clip(mean, 1e-12, 1e12)

        if y.shape != mean.shape:
            raise ValueError("foci and mean must have matching experiment-by-voxel shapes.")
        if y.shape != (moderators.shape[0], bases.shape[0]):
            raise ValueError("foci must have shape (n_experiments, n_voxels).")

        fisher_info = cls._compute_fisher_information(moderators, bases, mean)
        bread_inverse = cls._sandwich_bread_inverse(fisher_info, ridge)
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

    def _get_group_log_intensity(self, group):
        """Return cached baseline group log-intensity values."""
        group_log_intensity = self._group_log_intensity_cache.get(group)
        if group_log_intensity is None:
            bases = self.estimator.inputs_["coef_spline_bases"]
            coefficient = self._get_group_coefficient_matrix(group)
            group_log_intensity = bases @ coefficient[-1]
            self._group_log_intensity_cache[group] = group_log_intensity
        return group_log_intensity

    def _get_group_null_log_intensity(self, group):
        """Return cached null baseline log-intensity for homogeneity testing."""
        group_null_log_intensity = self._group_null_log_intensity_cache.get(group)
        if group_null_log_intensity is None:
            foci = self.estimator.inputs_["foci_by_experiment_voxel"][group]
            total_foci = foci.sum()
            n_experiments, n_voxels = foci.shape
            group_null_log_intensity = np.log(max(float(total_foci), np.finfo(float).tiny))
            group_null_log_intensity -= np.log(n_experiments * n_voxels)
            self._group_null_log_intensity_cache[group] = group_null_log_intensity
        return group_null_log_intensity

    def _run_group_inference(self):
        """Evaluate all prepared group contrasts and write maps into the result."""
        for con_group_count, con_group in enumerate(self.t_con_groups):
            group_stats = self._evaluate_group_contrast(con_group)
            self._store_group_inference_result(con_group_count, group_stats)

    @_check_fit
    def _glh_con_group(self):
        """Conduct GLH testing for baseline group spatial intensity.

        This method mirrors :meth:`nimare.meta.cbmr.CBMRInference._glh_con_group`
        and remains available for advanced users who interact with the lower-level
        inference object directly.
        """
        self._run_group_inference()

    def _evaluate_group_contrast(self, con_group):
        """Compute baseline spatial-intensity statistics for one group contrast."""
        bases = self.estimator.inputs_["coef_spline_bases"]
        involved_groups, simp_con_group, is_homogeneity_test = self._summarize_group_contrast(
            con_group
        )
        contrast_log_intensity = self._compute_group_contrast_log_intensity(
            simp_con_group,
            involved_groups,
            is_homogeneity_test,
        )
        covariance = self._get_intercept_covariance_for_groups(involved_groups)
        if con_group.shape[0] == 1:
            z_stats, p_vals = self._compute_group_wald_statistics(
                simp_con_group,
                involved_groups,
                covariance,
                contrast_log_intensity,
                bases,
            )
            chi_square = None
        else:
            chi_square, z_stats, p_vals = self._compute_group_glh_statistics(
                simp_con_group,
                involved_groups,
                covariance,
                contrast_log_intensity,
                bases,
                is_homogeneity_test,
            )
        return {
            "contrast_count": con_group.shape[0],
            "chi_square": chi_square,
            "p": p_vals,
            "z": z_stats,
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
        """Project fitted baseline log-intensities through one contrast matrix."""
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
    def _compute_group_wald_statistics(
        simp_con_group,
        involved_groups,
        cov_spatial_coef,
        contrast_log_intensity,
        bases,
    ):
        """Compute one-contrast Wald statistics for baseline group inference."""
        n_bases = bases.shape[1]
        var_log_intensity = []
        for index in range(len(involved_groups)):
            cov_spatial_coef_k = cov_spatial_coef[
                index * n_bases : (index + 1) * n_bases,
                index * n_bases : (index + 1) * n_bases,
            ]
            var_log_intensity_k = np.sum(np.multiply(bases @ cov_spatial_coef_k, bases), axis=1)
            var_log_intensity.append(var_log_intensity_k)
        var_log_intensity = np.stack(var_log_intensity, axis=0)
        involved_var_log_intensity = simp_con_group**2 @ var_log_intensity
        involved_std_log_intensity = np.sqrt(np.maximum(involved_var_log_intensity, 0.0))
        z_stats_spatial = contrast_log_intensity / np.where(
            involved_std_log_intensity > 0,
            involved_std_log_intensity,
            np.inf,
        )
        if len(involved_groups) == 1:
            p_vals_spatial = scipy.stats.norm.sf(z_stats_spatial)
        else:
            p_vals_spatial = scipy.stats.norm.sf(np.abs(z_stats_spatial)) * 2
        p_vals_spatial = _clip_p_values(
            p_vals_spatial,
            dtype=DEFAULT_FLOAT_DTYPE,
            copy=False,
        )
        return z_stats_spatial.ravel(), p_vals_spatial.ravel()

    def _compute_group_glh_statistics(
        self,
        simp_con_group,
        involved_groups,
        cov_spatial_coef,
        contrast_log_intensity,
        bases,
        is_homogeneity_test,
    ):
        """Compute multi-row GLH statistics for baseline group inference."""
        n_involved_groups = len(involved_groups)
        n_voxels, n_bases = bases.shape
        cov_spatial_coef = cov_spatial_coef.reshape(
            n_involved_groups,
            n_bases,
            n_involved_groups,
            n_bases,
        )
        cov_log_intensity = np.einsum(
            "vi,kisj,vj->ksv",
            bases,
            cov_spatial_coef,
            bases,
            optimize=True,
        )
        chi_sq_spatial = self._chi_square_log_intensity(
            n_voxels,
            n_involved_groups,
            simp_con_group,
            cov_log_intensity,
            contrast_log_intensity,
        )
        p_vals_spatial = scipy.stats.chi2.sf(chi_sq_spatial, df=simp_con_group.shape[0])
        p_vals_spatial = _clip_p_values(
            p_vals_spatial,
            dtype=DEFAULT_FLOAT_DTYPE,
            copy=False,
        )
        if is_homogeneity_test:
            z_stats_spatial = scipy.stats.norm.isf(p_vals_spatial)
            z_stats_spatial[z_stats_spatial < 0] = 0
        else:
            z_p_values = np.maximum(
                p_vals_spatial,
                2 * _minimum_positive_float(p_vals_spatial.dtype),
            )
            z_stats_spatial = scipy.stats.norm.isf(z_p_values / 2)
        z_stats_spatial = np.clip(z_stats_spatial, a_min=-10, a_max=10)
        return chi_sq_spatial, z_stats_spatial, p_vals_spatial

    @staticmethod
    def _chi_square_log_intensity(
        n_voxels,
        n_involved_groups,
        simp_con_group,
        cov_log_intensity,
        contrast_log_intensity,
    ):
        """Calculate voxel-wise chi-square statistics for group GLH tests."""
        if cov_log_intensity.ndim == 3:
            if cov_log_intensity.shape[:2] == (n_involved_groups, n_involved_groups):
                cov_by_voxel = np.moveaxis(cov_log_intensity, -1, 0)
            else:
                cov_by_voxel = cov_log_intensity
        else:
            cov_by_voxel = cov_log_intensity.reshape(
                n_involved_groups, n_involved_groups, n_voxels
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

    def _store_group_inference_result(self, con_group_count, group_stats):
        """Write one computed baseline group-inference result into result maps."""
        contrast_name = self.t_con_groups_name[con_group_count] if self.t_con_groups_name else None
        if contrast_name:
            if group_stats["contrast_count"] > 1:
                self.result.maps[f"chiSquare_group-{contrast_name}"] = group_stats["chi_square"]
            self.result.maps[f"p_group-{contrast_name}"] = group_stats["p"]
            self.result.maps[f"z_group-{contrast_name}"] = group_stats["z"]
        else:
            if group_stats["contrast_count"] > 1:
                self.result.maps[f"chiSquare_GLH_groups_{con_group_count}"] = group_stats[
                    "chi_square"
                ]
            self.result.maps[f"p_GLH_groups_{con_group_count}"] = group_stats["p"]
            self.result.maps[f"z_GLH_groups_{con_group_count}"] = group_stats["z"]

    def _run_moderator_inference(self):
        """Evaluate all moderator contrasts within every fitted group."""
        for con_moderator_count, con_moderator in enumerate(self.t_con_moderators):
            for group in self.groups:
                moderator_stats = self._evaluate_moderator_contrast(group, con_moderator)
                self._store_moderator_inference_result(
                    con_moderator_count,
                    group,
                    moderator_stats,
                )

    @_check_fit
    def _glh_con_moderator(self):
        """Conduct GLH testing for spatially varying moderator effects.

        This method mirrors :meth:`nimare.meta.cbmr.CBMRInference._glh_con_moderator`,
        but writes voxel-wise moderator inference statistics into maps because
        spatially varying moderator effects are functions over voxels.
        """
        self._run_moderator_inference()

    def _evaluate_moderator_contrast(self, group, con_moderator):
        """Compute spatially varying moderator statistics for one group and contrast."""
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
        """Compute voxel-wise Wald statistics for spatially varying coefficients."""
        contrast_eta = contrast @ coefficient @ bases.T
        contrast_cov = cls._contrast_covariance_by_voxel(contrast, covariance, bases)

        if contrast.shape[0] == 1:
            contrast_var = contrast_cov[:, 0, 0]
            contrast_std = np.sqrt(np.maximum(contrast_var, 0.0))
            z_stats = contrast_eta[0] / np.where(contrast_std > 0, contrast_std, np.inf)
            p_vals = scipy.stats.norm.sf(np.abs(z_stats)) * 2
            p_vals = _clip_p_values(
                p_vals,
                dtype=np.asarray(p_vals).dtype,
                copy=False,
            )
            chi_square = None
        else:
            solved = np.linalg.solve(contrast_cov, contrast_eta.T[..., np.newaxis])
            chi_square = np.einsum("ns,ns->n", contrast_eta.T, solved[..., 0], optimize=True)
            p_vals = scipy.stats.chi2.sf(chi_square, df=contrast.shape[0])
            p_vals = _clip_p_values(
                p_vals,
                dtype=np.asarray(p_vals).dtype,
                copy=False,
            )
            z_p_values = np.maximum(
                p_vals,
                2 * _minimum_positive_float(np.asarray(p_vals).dtype),
            )
            z_stats = scipy.stats.norm.isf(z_p_values / 2)

        return {
            "contrast_count": contrast.shape[0],
            "chi_square": chi_square,
            "p": p_vals,
            "z": z_stats,
        }

    def _store_moderator_inference_result(self, con_moderator_count, group, moderator_stats):
        """Write one spatially varying moderator-inference result into result maps."""
        contrast_name = (
            self.t_con_moderators_name[con_moderator_count] if self.t_con_moderators_name else None
        )
        if contrast_name:
            if moderator_stats["contrast_count"] > 1:
                self.result.maps[f"chiSquare_svModerator_{contrast_name}_group-{group}"] = (
                    moderator_stats["chi_square"]
                )
            self.result.maps[f"p_svModerator_{contrast_name}_group-{group}"] = moderator_stats["p"]
            self.result.maps[f"z_svModerator_{contrast_name}_group-{group}"] = moderator_stats["z"]
        else:
            if moderator_stats["contrast_count"] > 1:
                self.result.maps[
                    f"chiSquare_GLH_svModerators_{con_moderator_count}_group-{group}"
                ] = moderator_stats["chi_square"]
            self.result.maps[f"p_GLH_svModerators_{con_moderator_count}_group-{group}"] = (
                moderator_stats["p"]
            )
            self.result.maps[f"z_GLH_svModerators_{con_moderator_count}_group-{group}"] = (
                moderator_stats["z"]
            )
