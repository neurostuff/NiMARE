"""Image-based meta-analysis estimators."""

from __future__ import division

import logging
from collections import Counter

import nibabel as nib
import numpy as np
import pandas as pd
import pymare
from joblib import Memory
from nilearn.image import concat_imgs, resample_to_img
from nilearn.maskers import NiftiMasker

try:
    # nilearn >= 0.13.0
    from nilearn.image.image import check_same_fov
except ImportError:
    # nilearn >= 0.12.0; nilearn <= 0.12.1
    from nilearn._utils.niimg_conversions import check_same_fov

from pymare.stats import estimate_null_correlation

from nimare import _version
from nimare.estimator import Estimator
from nimare.meta._dependence import DependenceModel, hashable_label
from nimare.meta._permutation import _empirical_max_p, _permuted_ols
from nimare.meta.utils import _apply_liberal_mask
from nimare.transforms import d_to_g, p_to_z, t_to_d, t_to_z
from nimare.utils import (
    _boolean_unmask,
    _check_ncores,
    get_masker,
    get_masker_mask_image,
)

LGR = logging.getLogger(__name__)
__version__ = _version.get_versions()["version"]

# Mirrored from PyMARE so a typo is rejected at construction, not after the caller has
# assembled and masked a whole dataset.
WEIGHT_SCHEMES = frozenset({"individual", "rescale", "collapse"})


class IBMAEstimator(Estimator):
    """Base class for meta-analysis methods in :mod:`~nimare.meta`.

    .. warning::
        Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed in
        a future release. Prefer :class:`~nimare.nimads.Studyset`.

    .. versionchanged:: 0.20.1

        - New parameter: ``groupby``, identifying which images are statistically dependent on
          each other because they come from the same participants.

    .. versionchanged:: 0.2.1

        - New parameters: ``memory`` and ``memory_level`` for memory caching.

    .. versionchanged:: 0.2.0

        * Remove `resample` and `memory_limit` arguments. Resampling is now
          performed only if shape/affines are different.

    .. versionadded:: 0.0.12

        * IBMA-specific elements of ``Estimator`` excised and used to create ``IBMAEstimator``.
        * Generic kwargs and args converted to named kwargs.
          All remaining kwargs are for resampling.

    """

    #: Whether this estimator reads ``inputs_["corr_matrix"]``. Only the combination tests
    #: do. Cluster-robust standard errors are distribution-free, so estimating a whole-brain
    #: correlation for the meta-regression estimators would produce something nothing reads.
    _requires_corr_matrix = False

    def __init__(
        self,
        aggressive_mask=True,
        memory=Memory(location=None, verbose=0),
        memory_level=0,
        *,
        mask=None,
        groupby=None,
        **kwargs,
    ):
        self.aggressive_mask = aggressive_mask
        self.groupby = groupby

        if isinstance(groupby, str):
            # Pull the labels out of the dataset the same way sample sizes are pulled, so
            # that they arrive aligned to inputs_["id"] and drop_invalid applies to them too.
            self._required_inputs = dict(self._required_inputs)
            self._required_inputs["dependence_groups"] = ("metadata", groupby)

        if mask is not None:
            mask = get_masker(mask, memory=memory, memory_level=memory_level)
        self.masker = mask

        super().__init__(memory=memory, memory_level=memory_level)

        # defaults for resampling images (nilearn's defaults do not work well)
        self._resample_kwargs = {
            "clip": True,
            "interpolation": "linear",
            "copy_header": True,
        }

        # Identify any kwargs
        resample_kwargs = {k: v for k, v in kwargs.items() if k.startswith("resample__")}

        # Flag any extraneous kwargs
        other_kwargs = dict(set(kwargs.items()) - set(resample_kwargs.items()))
        if other_kwargs:
            LGR.warn(f"Unused keyword arguments found: {tuple(other_kwargs.items())}")

        # Update the default resampling parameters
        resample_kwargs = {k.split("resample__")[1]: v for k, v in resample_kwargs.items()}
        self._resample_kwargs.update(resample_kwargs)

    def _preprocess_input(self, dataset):
        """Preprocess inputs to the Estimator from the Dataset as needed.

        .. warning::
            Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed
            in a future release. Prefer :class:`~nimare.nimads.Studyset`.
        """
        masker, mask_img = get_masker_mask_image(self.masker, dataset=dataset)

        # Reserve the key for the correlation matrix
        self.inputs_["corr_matrix"] = None

        if self.aggressive_mask:
            # Ensure that protected values are not included among _required_inputs
            assert (
                "aggressive_mask" not in self._required_inputs.keys()
            ), "This is a protected name."

            if "aggressive_mask" in self.inputs_.keys():
                LGR.warning("Removing existing 'aggressive_mask' from Estimator.")
                self.inputs_.pop("aggressive_mask")
        else:
            # A dictionary to collect data, to be further reduced by the liberal mask.
            self.inputs_["data_bags"] = {}

        for name, (type_, _) in self._required_inputs.items():
            if type_ == "image":
                # Resampling will only occur if shape/affines are different
                imgs = [
                    (
                        nib.load(img)
                        if check_same_fov(nib.load(img), reference_masker=mask_img)
                        else resample_to_img(nib.load(img), mask_img, **self._resample_kwargs)
                    )
                    for img in self.inputs_[name]
                ]

                # input to NiFtiLabelsMasker must be 4d
                img4d = concat_imgs(imgs, ensure_ndim=4)

                # Mask required input images using either the dataset's mask or the estimator's.
                temp_arr = masker.transform(img4d)
                if name == "varcope_maps":
                    min_varcope = 1.0 / np.sqrt(np.finfo(temp_arr.dtype).max)
                    invalid_mask = ~np.isfinite(temp_arr) | (temp_arr <= min_varcope)
                    if np.any(invalid_mask):
                        n_invalid = int(np.sum(invalid_mask))
                        LGR.warning(
                            "Found %d non-finite, non-positive, or tiny varcope values; "
                            "setting to NaN.",
                            n_invalid,
                        )
                        temp_arr = temp_arr.copy()
                        temp_arr[invalid_mask] = np.nan

                # To save memory, we only save the original image array and perform masking later
                # in the estimator if self.aggressive_mask is True.
                self.inputs_[name] = temp_arr

                if self.aggressive_mask:
                    # Determine the good voxels here
                    nonzero_voxels_bool = np.all(temp_arr != 0, axis=0)
                    nonnan_voxels_bool = np.all(~np.isnan(temp_arr), axis=0)
                    good_voxels_bool = np.logical_and(nonzero_voxels_bool, nonnan_voxels_bool)

                    if "aggressive_mask" not in self.inputs_.keys():
                        self.inputs_["aggressive_mask"] = good_voxels_bool
                    else:
                        # Require voxels to be valid across all image-based inputs.
                        self.inputs_["aggressive_mask"] = np.logical_and(
                            self.inputs_["aggressive_mask"],
                            good_voxels_bool,
                        )
                else:
                    data_bags = zip(*_apply_liberal_mask(temp_arr))

                    keys = ["values", "voxel_mask", "study_mask"]
                    self.inputs_["data_bags"][name] = [dict(zip(keys, bag)) for bag in data_bags]

        # Further reduce image-based inputs to remove "bad" voxels
        # (voxels with zeros or NaNs in any studies)
        if self.aggressive_mask:
            if n_bad_voxels := (
                self.inputs_["aggressive_mask"].size - self.inputs_["aggressive_mask"].sum()
            ):
                LGR.warning(f"Masking out {n_bad_voxels} additional voxels.")

        self._preprocess_dependence(dataset)

    def _dependence(self, study_mask=None):
        """Return the grouping of the images being fitted.

        Built per call rather than cached, because the liberal-mask path fits bags in which
        only some of a group's images are present, so the grouping has to describe the bag.

        Parameters
        ----------
        study_mask : None or :obj:`numpy.ndarray`, optional
            Indices of the images being fitted, as supplied to ``_fit_model`` by the
            liberal-mask path. None means all images.

        Returns
        -------
        :obj:`~nimare.meta._dependence.DependenceModel`
        """
        model = DependenceModel(
            self.inputs_["contrast_names"],
            enabled=self.groupby is not False,
        )
        return model if study_mask is None else model.for_images(study_mask)

    def _fe_dof_map(self, est_summary, study_mask, n_voxels):
        """Return the fixed-effect degrees of freedom, one value per voxel.

        PyMARE reports Satterthwaite degrees of freedom for the CR2
        cluster-robust standard errors whenever group labels were supplied
        :footcite:p:`tipton2015small`, which is the reference distribution its
        p-values were actually drawn from. These are non-integer and vary by
        voxel under the liberal mask. When no labels reached PyMARE there is no
        cluster-robust covariance and hence no ``fe_dof``, so fall back to the
        group count.

        See Also
        --------
        nimare.meta._dependence.DependenceModel.dof
        """
        dof = est_summary.fe_dof
        if dof is None:
            return np.full(n_voxels, self._dependence(study_mask).dof, dtype=float)

        # (p, d) with p == 1, since every IBMA model is intercept-only.
        return np.asarray(dof, dtype=float).reshape(-1)[:n_voxels]

    def _sample_sizes_for_mask(self, study_mask):
        """Return per-image sample sizes aligned to a fitted model.

        Validation is left to PyMARE, which rejects non-finite or non-positive
        weights and, for the grouped combination tests, requires repeated rows
        in a group to carry equal weights.
        """
        return np.asarray(
            [np.mean(self.inputs_["sample_sizes"][idx]) for idx in study_mask],
            dtype=float,
        )

    def _resolve_group_labels(self, dataset):
        """Return one group label per image, in ``inputs_["id"]`` order."""
        if isinstance(self.groupby, str):
            # Registered as a metadata requirement in __init__, so it is already
            # collected and aligned.
            return [hashable_label(v) for v in self.inputs_["dependence_groups"]]

        if self.groupby is not None and self.groupby is not False:
            labels = [hashable_label(v) for v in np.asarray(self.groupby).ravel()]
            if len(labels) != len(self.inputs_["id"]):
                raise ValueError(
                    f"groupby must contain one label per image: expected "
                    f"{len(self.inputs_['id'])}, got {len(labels)}."
                )
            return labels

        # Look the study up per image id rather than filtering the table, so
        # the labels line up with inputs_["id"] no matter what order the rows
        # of dataset.images happen to be in.
        study_by_image = dict(zip(dataset.images["id"], dataset.images["study_id"]))
        try:
            return [study_by_image[image_id] for image_id in self.inputs_["id"]]
        except KeyError as exc:
            raise ValueError(
                f"No study could be found for image {exc.args[0]!r}; the dataset's image "
                "table and the estimator's inputs are out of sync."
            ) from exc

    def _preprocess_dependence(self, dataset):
        """Identify groups of images that are statistically dependent.

        Populates ``contrast_names`` (an integer group index per image),
        ``num_contrasts`` (images per group) and, for the combination tests
        only, ``corr_matrix`` (the empirical null correlation between the input
        maps). Estimators pass these on so that their inference accounts for
        the dependence.
        """
        labels = self._resolve_group_labels(dataset)

        # Sorting keeps the code assignment stable across runs; plain set() iteration
        # depends on string hash randomization, which would make seeded permutations
        # irreproducible. str() first so a mix of label types still orders.
        label_to_int = {label: i for i, label in enumerate(sorted(set(labels), key=str))}
        label_counts = Counter(labels)

        self.inputs_["contrast_names"] = np.array([label_to_int[label] for label in labels])
        self.inputs_["num_contrasts"] = np.array([label_counts[label] for label in labels])

        n_studies = len(self.inputs_["id"])
        n_unique = np.unique(self.inputs_["contrast_names"]).size
        if n_studies == n_unique:
            # Every group contains exactly one image, so there is no
            # within-group dependence to correct for.
            return

        if self.groupby is False:
            LGR.warning(
                f"{n_studies - n_unique} image(s) share a group with another image, but "
                "groupby=False was requested, so they will be treated as independent. "
                "This inflates significance."
            )
            return

        if not self._requires_corr_matrix:
            LGR.info(
                "Accounting for dependence among %d image(s) from %d group(s).",
                n_studies,
                n_unique,
            )
            return

        # Calculate the correlation matrix on the first image-based input,
        # which is the map the dependence acts on.
        image_names = [
            name for name, (type_, _) in self._required_inputs.items() if type_ == "image"
        ]
        if not image_names:
            return
        maps = self.inputs_[image_names[0]]

        if self.aggressive_mask:
            maps = maps[:, self.inputs_["aggressive_mask"]]

        # Correlating the raw maps measures how much studies agree, not how dependent they
        # are: every map carries the same activation, so studies independent by construction
        # still come out correlated. estimate_null_correlation strips the shared signal
        # first, which is what Brown's method and Stouffer's inflation term require. Passing
        # the groups inverts the shrinkage centering induces exactly rather than rescaling
        # it, which matters when few images are combined.
        corr_matrix = estimate_null_correlation(maps, groups=self.inputs_["contrast_names"])
        self.inputs_["corr_matrix"] = corr_matrix

        off_diagonal = corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)]
        LGR.info(
            "Correcting for dependence among %d image(s) from %d group(s) "
            "(max off-diagonal correlation %.3f).",
            n_studies,
            n_unique,
            np.nanmax(np.abs(off_diagonal)) if off_diagonal.size else np.nan,
        )


class _PyMARERegressionEstimator(IBMAEstimator):
    """Base class for the IBMA estimators backed by a PyMARE meta-regression estimator.

    These all share the same two dependence parameters, because the PyMARE estimators
    underneath them do. The combination tests (:class:`Fishers`, :class:`Stouffers`) have a
    different parameterization in PyMARE and so do not inherit from this.
    """

    def __init__(self, weight_scheme="rescale", rho=0.8, **kwargs):
        if weight_scheme not in WEIGHT_SCHEMES:
            raise ValueError(
                f"Invalid weight_scheme '{weight_scheme}'; must be one of "
                f"{sorted(WEIGHT_SCHEMES)}."
            )
        if not isinstance(rho, (int, float, np.integer, np.floating)) or isinstance(rho, bool):
            raise ValueError(f"Invalid rho {rho!r}; must be a number.")
        if not 0.0 <= float(rho) <= 1.0:
            raise ValueError(f"Invalid rho {rho!r}; must lie in [0, 1].")

        self.weight_scheme = weight_scheme
        self.rho = float(rho)
        super().__init__(**kwargs)

    def _pymare_weighting_kwargs(self, study_mask):
        """Return the PyMARE weighting arguments for the current model.

        ``rho`` is omitted under ``weight_scheme="individual"``, which models no
        within-group correlation and warns if it is supplied anyway.
        """
        if self.weight_scheme == "individual" or self._dependence(study_mask).labels is None:
            return {"weight_scheme": self.weight_scheme}
        return {"weight_scheme": self.weight_scheme, "rho": self.rho}


class Fishers(IBMAEstimator):
    """An image-based meta-analytic test using t- or z-statistic images.

    .. versionchanged:: 0.20.1

        * New parameter: ``groupby``, identifying images contributed by the same participants.
        * The ``dof`` map now counts independent groups rather than images.

    Requires z-statistic images, but will be extended to work with t-statistic images as well.

    This method is described in :footcite:t:`fisher1946statistical`.

    .. versionchanged:: 0.3.0

        * New parameter: ``two_sided``, controls the type of test to be performed. In addition,
            the default is now set to True (two-sided), which differs from previous versions
            where only one-sided tests were performed.

    .. versionchanged:: 0.2.1

        * New parameter: ``aggressive_mask``, to control whether to use an aggressive mask.

    Parameters
    ----------
    aggressive_mask : :obj:`bool`, optional
        Voxels with a value of zero of NaN in any of the input maps will be removed
        from the analysis.
        If False, all voxels are included by running a separate analysis on bags
        of voxels that belong that have a valid value across the same studies.
        Default is True.
    groupby : None, :obj:`str`, array-like, or False, optional
        How to identify images that share participants and are therefore dependent. None (the
        default) groups by ``study_id``. A :obj:`str` names a metadata field to group by
        instead, for a paper contributing independent samples (e.g. patients and controls).
        An array supplies one label per image. False treats every image as independent, which
        inflates significance whenever that is untrue. Default is None.
    use_sample_size : :obj:`bool`, optional
        Whether to assign each study a total weighted-Fisher coefficient equal
        to its sample size. Repeated images divide that coefficient internally
        in PyMARE, so image multiplicity does not change the study's total
        weight. Default is False, preserving ordinary Fisher/Brown inference.
    two_sided : :obj:`bool`, optional
        If True, performs an unsigned t-test. Both positive and negative effects are considered;
        the null hypothesis is that the effect is zero. If False, only positive effects are
        considered as relevant. The null hypothesis is that the effect is zero or negative.
        Default is True.

    Notes
    -----
    Requires ``z`` images.

    :meth:`fit` produces a :class:`~nimare.results.MetaResult` object with the following maps:

    ============== ===============================================================================
    "z"            Z-statistic map from one-sample test.
    "p"            P-value map from one-sample test.
    "dof"          Degrees of freedom map from one-sample test.
    ============== ===============================================================================

    Warnings
    --------
    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will result in invalid results. It cannot be used with these types of maskers.

    By default, all image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis. Setting ``aggressive_mask=False`` will
    instead run tha analysis in bags of voxels that have a valid value across
    the same studies.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`pymare.estimators.FisherCombinationTest`:
        The PyMARE estimator called by this class.
    """

    # Brown's method needs the null correlation between the input maps; see
    # IBMAEstimator._requires_corr_matrix.
    _requires_corr_matrix = True

    _required_inputs = {"z_maps": ("image", "z")}

    def __init__(self, two_sided=True, use_sample_size=False, **kwargs):
        super().__init__(**kwargs)
        self.use_sample_size = use_sample_size
        if self.use_sample_size:
            self._required_inputs = dict(self._required_inputs)
            self._required_inputs["sample_sizes"] = ("metadata", "sample_sizes")
        self.two_sided = two_sided
        self._mode = "concordant" if self.two_sided else "directed"

    def _generate_description(self):
        description = (
            f"An image-based meta-analysis was performed with NiMARE {__version__} "
            "(RRID:SCR_017398; \\citealt{Salo2023}) on "
            f"{len(self.inputs_['id'])} z-statistic images using the Fisher "
            "combined probability method \\citep{fisher1946statistical}."
        )
        if self.use_sample_size:
            description += " Studies received weighted-Fisher coefficients equal to sample size."
        return description

    def _fit_model(self, stat_maps, study_mask=None, corr=None):
        """Fit the model to the data."""
        n_studies, n_voxels = stat_maps.shape

        if study_mask is None:
            # If no mask is provided, assume all studies are included. This is always the case
            # when using the aggressive mask.
            study_mask = np.arange(n_studies)

        est = pymare.estimators.FisherCombinationTest(mode=self._mode)

        # When studies contribute several images, Brown's method replaces
        # Fisher's chi-squared reference with a scaled one.
        groups = self._dependence(study_mask).labels
        sub_corr = None
        if groups is not None and corr is not None:
            sub_corr = corr[np.ix_(study_mask, study_mask)]

        weights = None
        if self.use_sample_size:
            weights = self._sample_sizes_for_mask(study_mask)[:, None]

        # Group labels and optional weights are per-image, so they are passed
        # as single columns rather than tiled across every voxel.
        pymare_dset = pymare.Dataset(y=stat_maps, n=weights, g=groups)
        est.fit_dataset(pymare_dset, corr=sub_corr)
        est_summary = est.summary()

        z_map = est_summary.z.squeeze()
        p_map = est_summary.p.squeeze()
        dof_map = np.tile(self._dependence(study_mask).dof, n_voxels).astype(np.int32)

        return z_map, p_map, dof_map

    def _fit(self, dataset):
        self.dataset = dataset
        self.masker = self.masker or dataset.masker
        if not isinstance(self.masker, NiftiMasker):
            raise ValueError(
                f"A {type(self.masker)} mask has been detected. "
                "Only NiftiMaskers are allowed for this Estimator. "
                "This is because aggregation, such as averaging values across ROIs, "
                "will produce invalid results."
            )

        if self.aggressive_mask:
            voxel_mask = self.inputs_["aggressive_mask"]
            result_maps = self._fit_model(
                self.inputs_["z_maps"][:, voxel_mask],
                corr=self.inputs_["corr_matrix"],
            )

            z_map, p_map, dof_map = tuple(
                map(lambda x: _boolean_unmask(x, voxel_mask), result_maps)
            )
        else:
            n_voxels = self.inputs_["z_maps"].shape[1]
            z_map = np.zeros(n_voxels, dtype=float)
            p_map = np.zeros(n_voxels, dtype=float)
            dof_map = np.zeros(n_voxels, dtype=np.int32)
            for bag in self.inputs_["data_bags"]["z_maps"]:
                (
                    z_map[bag["voxel_mask"]],
                    p_map[bag["voxel_mask"]],
                    dof_map[bag["voxel_mask"]],
                ) = self._fit_model(
                    bag["values"], bag["study_mask"], corr=self.inputs_["corr_matrix"]
                )

        maps = {"z": z_map, "p": p_map, "dof": dof_map}
        description = self._description_text()

        return maps, {}, description


class Stouffers(IBMAEstimator):
    """A t-test on z-statistic images.

    .. versionchanged:: 0.20.1

        * New parameter: ``groupby``, identifying images contributed by the same participants.
        * The ``dof`` map now counts independent groups rather than images.

    Requires z-statistic images.

    This method is described in :footcite:t:`stouffer1949american`.

    .. versionchanged:: 0.3.0

        * New parameter: ``two_sided``, controls the type of test to be performed. In addition,
            the default is now set to True (two-sided), which differs from previous versions
            where only one-sided tests were performed.
        * Add correction for multiple contrasts within a study.
        * New parameter: ``normalize_contrast_weights`` to normalized the weights by the
            number of contrasts in each study. Removed again in 0.20.1; see ``groupby``.

    .. versionchanged:: 0.2.1

        * New parameter: ``aggressive_mask``, to control whether to use an aggressive mask.

    Parameters
    ----------
    aggressive_mask : :obj:`bool`, optional
        Voxels with a value of zero of NaN in any of the input maps will be removed
        from the analysis.
        If False, all voxels are included by running a separate analysis on bags
        of voxels that belong that have a valid value across the same studies.
        Default is True.
    groupby : None, :obj:`str`, array-like, or False, optional
        How to identify images that share participants and are therefore dependent. None (the
        default) groups by ``study_id``. A :obj:`str` names a metadata field to group by
        instead, for a paper contributing independent samples (e.g. patients and controls).
        An array supplies one label per image. False treats every image as independent, which
        inflates significance whenever that is untrue. Default is None.
    use_sample_size : :obj:`bool`, optional
        Whether to use sample sizes for weights (i.e., "weighted Stouffer's") or not,
        as described in :footcite:t:`zaykin2011optimally`.
        Default is False.
    two_sided : :obj:`bool`, optional
        If True, performs an unsigned t-test. Both positive and negative effects are considered;
        the null hypothesis is that the effect is zero. If False, only positive effects are
        considered as relevant. The null hypothesis is that the effect is zero or negative.
        Default is True.

    Notes
    -----
    Requires ``z`` images and optionally the sample size metadata field.

    :meth:`fit` produces a :class:`~nimare.results.MetaResult` object with the following maps:

    ============== ===============================================================================
    "z"            Z-statistic map from one-sample test.
    "p"            P-value map from one-sample test.
    "dof"          Degrees of freedom map from one-sample test.
    ============== ===============================================================================

    Warnings
    --------
    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will result in invalid results. It cannot be used with these types of maskers.

    By default, all image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis. Setting ``aggressive_mask=False`` will
    instead run tha analysis in bags of voxels that have a valid value across
    the same studies.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`pymare.estimators.StoufferCombinationTest`:
        The PyMARE estimator called by this class.
    """

    # The variance inflation term needs the null correlation between the input maps;
    # see IBMAEstimator._requires_corr_matrix.
    _requires_corr_matrix = True

    _required_inputs = {"z_maps": ("image", "z")}

    def __init__(
        self,
        use_sample_size=False,
        two_sided=True,
        **kwargs,
    ):
        if "normalize_contrast_weights" in kwargs:
            raise TypeError(
                "normalize_contrast_weights was removed in 0.20.1. Repeated images are now "
                "combined into one variance-standardized statistic per group whenever "
                "`groupby` finds a group with more than one image, so the parameter no "
                "longer has anything to switch. Pass groupby=False to treat every image as "
                "independent instead. Note this is not a rename: the removed parameter "
                "divided weights by image count and kept every row."
            )
        super().__init__(**kwargs)
        self._required_inputs = dict(self._required_inputs)
        self.use_sample_size = use_sample_size
        if self.use_sample_size:
            self._required_inputs["sample_sizes"] = ("metadata", "sample_sizes")

        self.two_sided = two_sided
        self._mode = "concordant" if self.two_sided else "directed"

    def _generate_description(self):
        description = (
            f"An image-based meta-analysis was performed with NiMARE {__version__} "
            "(RRID:SCR_017398; \\citealt{Salo2023}) on "
            f"{len(self.inputs_['id'])} z-statistic images using the Stouffer "
            "method \\citep{stouffer1949american}"
        )

        if self.use_sample_size:
            description += (
                ", with studies weighted by the square root of the study sample sizes, per "
                "\\cite{zaykin2011optimally}."
            )
        else:
            description += "."

        return description

    def _fit_model(self, stat_maps, study_mask=None, corr=None):
        """Fit the model to the data."""
        n_studies, n_voxels = stat_maps.shape

        if study_mask is None:
            # If no mask is provided, assume all studies are included. This is always the case
            # when using the aggressive mask.
            study_mask = np.arange(n_studies)

        groups = self._dependence(study_mask).labels
        sub_corr = None
        if groups is not None and corr is not None:
            sub_corr = corr[np.ix_(study_mask, study_mask)]

        weights = np.ones(n_studies)

        # Correcting the variance but leaving the weights alone would still let a study
        # with fifty images outvote a study with one, so aggregate whenever there is a
        # group to aggregate. PyMARE forms one variance-standardized mean per group and
        # applies one weight to it.
        est = pymare.estimators.StoufferCombinationTest(
            mode=self._mode,
            group_level=groups is not None,
        )

        if self.use_sample_size:
            sample_sizes = self._sample_sizes_for_mask(study_mask)
            weights *= np.sqrt(sample_sizes)

        # Weights and group labels are per-image, not per-voxel, so they are
        # passed as columns rather than tiled across the whole map.
        pymare_dset = pymare.Dataset(y=stat_maps, n=weights[:, None], g=groups)
        est.fit_dataset(pymare_dset, corr=sub_corr)
        est_summary = est.summary()

        z_map = est_summary.z.squeeze()
        p_map = est_summary.p.squeeze()
        dof_map = np.tile(self._dependence(study_mask).dof, n_voxels).astype(np.int32)

        return z_map, p_map, dof_map

    def _fit(self, dataset):
        self.dataset = dataset
        self.masker = self.masker or dataset.masker
        if not isinstance(self.masker, NiftiMasker):
            raise ValueError(
                f"A {type(self.masker)} mask has been detected. "
                "Only NiftiMaskers are allowed for this Estimator. "
                "This is because aggregation, such as averaging values across ROIs, "
                "will produce invalid results."
            )

        if self.aggressive_mask:
            voxel_mask = self.inputs_["aggressive_mask"]

            result_maps = self._fit_model(
                self.inputs_["z_maps"][:, voxel_mask],
                corr=self.inputs_["corr_matrix"],
            )

            z_map, p_map, dof_map = tuple(
                map(lambda x: _boolean_unmask(x, voxel_mask), result_maps)
            )
        else:
            n_voxels = self.inputs_["z_maps"].shape[1]
            z_map = np.zeros(n_voxels, dtype=float)
            p_map = np.zeros(n_voxels, dtype=float)
            dof_map = np.zeros(n_voxels, dtype=np.int32)
            for bag in self.inputs_["data_bags"]["z_maps"]:
                (
                    z_map[bag["voxel_mask"]],
                    p_map[bag["voxel_mask"]],
                    dof_map[bag["voxel_mask"]],
                ) = self._fit_model(
                    bag["values"], bag["study_mask"], corr=self.inputs_["corr_matrix"]
                )

        maps = {"z": z_map, "p": p_map, "dof": dof_map}
        description = self._description_text()

        return maps, {}, description


class WeightedLeastSquares(_PyMARERegressionEstimator):
    """Weighted least-squares meta-regression.

    .. versionchanged:: 0.20.1

        * New parameters: ``groupby``, ``weight_scheme`` and ``rho``, controlling how images
          contributed by the same participants are grouped and weighted.
        * The ``dof`` map now reports the Satterthwaite degrees of freedom of the CR2
          cluster-robust standard errors, which is what the p-values are drawn from. It is
          floating point rather than ``int32``, and varies by voxel under the liberal mask.

    .. versionchanged:: 0.2.1

        * New parameter: ``aggressive_mask``, to control whether to use an aggressive mask.

    .. versionchanged:: 0.0.12

        * Add "se" to outputs.

    .. versionchanged:: 0.0.8

        * [FIX] Remove single-dimensional entries of each array of returns (:obj:`dict`).

    .. versionadded:: 0.0.4

    Provides the weighted least-squares estimate of the fixed effects given
    known/assumed between-study variance tau^2.
    When tau^2 = 0 (default), the model is the standard inverse-weighted
    fixed-effects meta-regression.

    This method was described in :footcite:t:`brockwell2001comparison`.

    Parameters
    ----------
    aggressive_mask : :obj:`bool`, optional
        Voxels with a value of zero of NaN in any of the input maps will be removed
        from the analysis.
        If False, all voxels are included by running a separate analysis on bags
        of voxels that belong that have a valid value across the same studies.
        Default is True.
    groupby : None, :obj:`str`, array-like, or False, optional
        How to identify images that share participants and are therefore dependent. None (the
        default) groups by ``study_id``. A :obj:`str` names a metadata field to group by
        instead, for a paper contributing independent samples (e.g. patients and controls).
        An array supplies one label per image. False treats every image as independent, which
        inflates significance whenever that is untrue. Default is None.
    weight_scheme : {'rescale', 'individual', 'collapse'}, optional
        How images within a group are weighted, passed to PyMARE. ``'rescale'`` (the default)
        divides each image's weight by its group size, so a group's total weight does not grow
        with the number of maps it contributed -- the correlated-effects model of
        :footcite:t:`hedges2010robust`, weighted as in :footcite:t:`fisher2015robumeta`.
        ``'individual'`` leaves the weights alone; ``'collapse'`` fits one row per group.
        Group labels also switch the standard errors to CR2 and the reference to a t
        distribution with Satterthwaite degrees of freedom :footcite:p:`tipton2015small`.
    rho : :obj:`float`, optional
        Assumed within-group correlation, in [0, 1]. Enters only through tau^2, to which
        results are weakly sensitive. Default is 0.8, matching ``robumeta``. Ignored when
        ``weight_scheme='individual'``.
    tau2 : :obj:`float` or 1D :class:`numpy.ndarray`, optional
        Assumed/known value of tau^2. Must be >= 0. Default is 0.

    Notes
    -----
    Requires :term:`beta` and :term:`varcope` images.

    :meth:`fit` produces a :class:`~nimare.results.MetaResult` object with the following maps:

    ============== ===============================================================================
    "z"            Z-statistic map from one-sample test.
    "p"            P-value map from one-sample test.
    "est"          Fixed effects estimate for intercept test.
    "se"           Standard error of fixed effects estimate.
    "dof"          Degrees of freedom map from one-sample test.
    ============== ===============================================================================

    Warnings
    --------
    Cluster-robust inference is asymptotic in the number of *groups*, not images. PyMARE warns
    at 10 or fewer groups, where robust variance estimation is anti-conservative
    :footcite:p:`hedges2010robust`, and when the Satterthwaite degrees of freedom fall below
    about 4 :footcite:p:`tipton2015small`. Both are common for small meta-analyses.

    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will likely result in biased results. The extent of this bias is currently
    unknown.

    By default, all image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis. Setting ``aggressive_mask=False`` will
    instead run tha analysis in bags of voxels that have a valid value across
    the same studies.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`pymare.estimators.WeightedLeastSquares`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"beta_maps": ("image", "beta"), "varcope_maps": ("image", "varcope")}

    def __init__(self, tau2=0, **kwargs):
        super().__init__(**kwargs)
        self.tau2 = tau2

    def _generate_description(self):
        description = (
            f"An image-based meta-analysis was performed with NiMARE {__version__} "
            "(RRID:SCR_017398; \\citealt{Salo2023}), on "
            f"{len(self.inputs_['id'])} beta images using the Weighted Least Squares approach "
            "\\citep{brockwell2001comparison}, "
            f"with an a priori tau-squared value of {self.tau2} defined across all voxels."
        )
        return description

    def _fit_model(self, beta_maps, varcope_maps, study_mask=None):
        """Fit the model to the data."""
        n_studies, n_voxels = beta_maps.shape

        if study_mask is None:
            # If no mask is provided, assume all studies are included. This is always the case
            # when using the aggressive mask.
            study_mask = np.arange(n_studies)

        # None when there is nothing to correct for, which is how PyMARE is told to use
        # model-based inference rather than cluster-robust ("sandwich") standard errors.
        groups = self._dependence(study_mask).labels

        pymare_dset = pymare.Dataset(y=beta_maps, v=varcope_maps, g=groups)
        est = pymare.estimators.WeightedLeastSquares(
            tau2=self.tau2,
            **self._pymare_weighting_kwargs(study_mask),
        )
        est.fit_dataset(pymare_dset)
        est_summary = est.summary()

        fe_stats = est_summary.get_fe_stats()
        z_map = fe_stats["z"].squeeze()
        p_map = fe_stats["p"].squeeze()
        est_map = fe_stats["est"].squeeze()
        se_map = fe_stats["se"].squeeze()
        dof_map = self._fe_dof_map(est_summary, study_mask, n_voxels)
        return z_map, p_map, est_map, se_map, dof_map

    def _fit(self, dataset):
        self.dataset = dataset
        self.masker = self.masker or dataset.masker
        if not isinstance(self.masker, NiftiMasker):
            LGR.warning(
                f"A {type(self.masker)} mask has been detected. "
                "Masks which average across voxels will likely produce biased results when used "
                "with this Estimator."
            )

        if self.aggressive_mask:
            voxel_mask = self.inputs_["aggressive_mask"]
            result_maps = self._fit_model(
                self.inputs_["beta_maps"][:, voxel_mask],
                self.inputs_["varcope_maps"][:, voxel_mask],
            )

            z_map, p_map, est_map, se_map, dof_map = tuple(
                map(lambda x: _boolean_unmask(x, voxel_mask), result_maps)
            )
        else:
            n_voxels = self.inputs_["beta_maps"].shape[1]

            z_map, p_map, est_map, se_map = [np.zeros(n_voxels, dtype=float) for _ in range(4)]
            dof_map = np.zeros(n_voxels, dtype=np.int32)

            beta_bags = self.inputs_["data_bags"]["beta_maps"]
            varcope_bags = self.inputs_["data_bags"]["varcope_maps"]
            for beta_bag, varcope_bag in zip(beta_bags, varcope_bags):
                (
                    z_map[beta_bag["voxel_mask"]],
                    p_map[beta_bag["voxel_mask"]],
                    est_map[beta_bag["voxel_mask"]],
                    se_map[beta_bag["voxel_mask"]],
                    dof_map[beta_bag["voxel_mask"]],
                ) = self._fit_model(
                    beta_bag["values"],
                    varcope_bag["values"],
                    beta_bag["study_mask"],
                )

        # tau2 is a float, not a map, so it can't go into the results dictionary
        tables = {"level-estimator": pd.DataFrame(columns=["tau2"], data=[self.tau2])}
        maps = {"z": z_map, "p": p_map, "est": est_map, "se": se_map, "dof": dof_map}
        description = self._description_text()

        return maps, tables, description


class DerSimonianLaird(_PyMARERegressionEstimator):
    """DerSimonian-Laird meta-regression estimator.

    .. versionchanged:: 0.20.1

        * New parameters: ``groupby``, ``weight_scheme`` and ``rho``, controlling how images
          contributed by the same participants are grouped and weighted.
        * The ``dof`` map now reports the Satterthwaite degrees of freedom of the CR2
          cluster-robust standard errors, which is what the p-values are drawn from. It is
          floating point rather than ``int32``, and varies by voxel under the liberal mask.

    .. versionchanged:: 0.2.1

        * New parameter: ``aggressive_mask``, to control whether to use an aggressive mask.

    .. versionchanged:: 0.0.12

        * Add "se" to outputs.

    .. versionchanged:: 0.0.8

        * [FIX] Remove single-dimensional entries of each array of returns (:obj:`dict`).

    .. versionadded:: 0.0.4

    Estimates the between-subject variance tau^2 using the :footcite:t:`dersimonian1986meta`
    method-of-moments approach :footcite:p:`dersimonian1986meta,kosmidis2017improving`.

    Parameters
    ----------
    aggressive_mask : :obj:`bool`, optional
        Voxels with a value of zero of NaN in any of the input maps will be removed
        from the analysis.
        If False, all voxels are included by running a separate analysis on bags
        of voxels that belong that have a valid value across the same studies.
        Default is True.
    groupby : None, :obj:`str`, array-like, or False, optional
        How to identify images that share participants and are therefore dependent. None (the
        default) groups by ``study_id``. A :obj:`str` names a metadata field to group by
        instead, for a paper contributing independent samples (e.g. patients and controls).
        An array supplies one label per image. False treats every image as independent, which
        inflates significance whenever that is untrue. Default is None.
    weight_scheme : {'rescale', 'individual', 'collapse'}, optional
        How images within a group are weighted, passed to PyMARE. ``'rescale'`` (the default)
        divides each image's weight by its group size, so a group's total weight does not grow
        with the number of maps it contributed -- the correlated-effects model of
        :footcite:t:`hedges2010robust`, weighted as in :footcite:t:`fisher2015robumeta`.
        ``'individual'`` leaves the weights alone; ``'collapse'`` fits one row per group.
        Group labels also switch the standard errors to CR2 and the reference to a t
        distribution with Satterthwaite degrees of freedom :footcite:p:`tipton2015small`.
    rho : :obj:`float`, optional
        Assumed within-group correlation, in [0, 1]. Enters only through tau^2, to which
        results are weakly sensitive. Default is 0.8, matching ``robumeta``. Ignored when
        ``weight_scheme='individual'``.

    Notes
    -----
    Requires :term:`beta` and :term:`varcope` images.

    :meth:`fit` produces a :class:`~nimare.results.MetaResult` object with the following maps:

    ============== ===============================================================================
    "z"            Z-statistic map from one-sample test.
    "p"            P-value map from one-sample test.
    "est"          Fixed effects estimate for intercept test.
    "se"           Standard error of fixed effects estimate.
    "tau2"         Estimated between-study variance.
    "dof"          Degrees of freedom map from one-sample test.
    ============== ===============================================================================

    Warnings
    --------
    Cluster-robust inference is asymptotic in the number of *groups*, not images. PyMARE warns
    at 10 or fewer groups, where robust variance estimation is anti-conservative
    :footcite:p:`hedges2010robust`, and when the Satterthwaite degrees of freedom fall below
    about 4 :footcite:p:`tipton2015small`. Both are common for small meta-analyses.

    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will likely result in biased results. The extent of this bias is currently
    unknown.

    By default, all image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis. Setting ``aggressive_mask=False`` will
    instead run tha analysis in bags of voxels that have a valid value across
    the same studies.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`pymare.estimators.DerSimonianLaird`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"beta_maps": ("image", "beta"), "varcope_maps": ("image", "varcope")}

    def _generate_description(self):
        description = (
            f"An image-based meta-analysis was performed with NiMARE {__version__} "
            "(RRID:SCR_017398; \\citealt{Salo2023}), on "
            f"{len(self.inputs_['id'])} beta and variance images using the "
            "DerSimonian-Laird method \\citep{dersimonian1986meta}, in which tau-squared is "
            "estimated on a voxel-wise basis using the method-of-moments approach "
            "\\citep{dersimonian1986meta,kosmidis2017improving}."
        )
        return description

    def _fit_model(self, beta_maps, varcope_maps, study_mask=None):
        """Fit the model to the data."""
        n_studies, n_voxels = beta_maps.shape

        if study_mask is None:
            # If no mask is provided, assume all studies are included. This is always the case
            # when using the aggressive mask.
            study_mask = np.arange(n_studies)

        # None when there is nothing to correct for, which is how PyMARE is told to use
        # model-based inference rather than cluster-robust ("sandwich") standard errors.
        groups = self._dependence(study_mask).labels

        pymare_dset = pymare.Dataset(y=beta_maps, v=varcope_maps, g=groups)
        est = pymare.estimators.DerSimonianLaird(**self._pymare_weighting_kwargs(study_mask))
        est.fit_dataset(pymare_dset)
        est_summary = est.summary()

        fe_stats = est_summary.get_fe_stats()
        z_map = fe_stats["z"].squeeze()
        p_map = fe_stats["p"].squeeze()
        est_map = fe_stats["est"].squeeze()
        se_map = fe_stats["se"].squeeze()
        tau2_map = est_summary.tau2.squeeze()
        dof_map = self._fe_dof_map(est_summary, study_mask, n_voxels)
        return z_map, p_map, est_map, se_map, tau2_map, dof_map

    def _fit(self, dataset):
        self.dataset = dataset
        self.masker = self.masker or dataset.masker
        if not isinstance(self.masker, NiftiMasker):
            LGR.warning(
                f"A {type(self.masker)} mask has been detected. "
                "Masks which average across voxels will likely produce biased results when used "
                "with this Estimator."
            )

        if self.aggressive_mask:
            voxel_mask = self.inputs_["aggressive_mask"]
            result_maps = self._fit_model(
                self.inputs_["beta_maps"][:, voxel_mask],
                self.inputs_["varcope_maps"][:, voxel_mask],
            )

            z_map, p_map, est_map, se_map, tau2_map, dof_map = tuple(
                map(lambda x: _boolean_unmask(x, voxel_mask), result_maps)
            )
        else:
            n_voxels = self.inputs_["beta_maps"].shape[1]

            z_map, p_map, est_map, se_map, tau2_map = [
                np.zeros(n_voxels, dtype=float) for _ in range(5)
            ]
            dof_map = np.zeros(n_voxels, dtype=np.int32)

            beta_bags = self.inputs_["data_bags"]["beta_maps"]
            varcope_bags = self.inputs_["data_bags"]["varcope_maps"]
            for beta_bag, varcope_bag in zip(beta_bags, varcope_bags):
                (
                    z_map[beta_bag["voxel_mask"]],
                    p_map[beta_bag["voxel_mask"]],
                    est_map[beta_bag["voxel_mask"]],
                    se_map[beta_bag["voxel_mask"]],
                    tau2_map[beta_bag["voxel_mask"]],
                    dof_map[beta_bag["voxel_mask"]],
                ) = self._fit_model(
                    beta_bag["values"],
                    varcope_bag["values"],
                    beta_bag["study_mask"],
                )

        maps = {
            "z": z_map,
            "p": p_map,
            "est": est_map,
            "se": se_map,
            "tau2": tau2_map,
            "dof": dof_map,
        }
        description = self._description_text()

        return maps, {}, description


class Hedges(_PyMARERegressionEstimator):
    """Hedges meta-regression estimator.

    .. versionchanged:: 0.20.1

        * New parameters: ``groupby``, ``weight_scheme`` and ``rho``, controlling how images
          contributed by the same participants are grouped and weighted.
        * The ``dof`` map now reports the Satterthwaite degrees of freedom of the CR2
          cluster-robust standard errors, which is what the p-values are drawn from. It is
          floating point rather than ``int32``, and varies by voxel under the liberal mask.

    .. versionchanged:: 0.2.1

        * New parameter: ``aggressive_mask``, to control whether to use an aggressive mask.

    .. versionchanged:: 0.0.12

        * Add "se" to outputs.

    .. versionchanged:: 0.0.8

        * [FIX] Remove single-dimensional entries of each array of returns (:obj:`dict`).

    .. versionadded:: 0.0.4

    Estimates the between-subject variance tau^2 using the :footcite:t:`hedges2014statistical`
    approach.

    Parameters
    ----------
    aggressive_mask : :obj:`bool`, optional
        Voxels with a value of zero of NaN in any of the input maps will be removed
        from the analysis.
        If False, all voxels are included by running a separate analysis on bags
        of voxels that belong that have a valid value across the same studies.
        Default is True.
    groupby : None, :obj:`str`, array-like, or False, optional
        How to identify images that share participants and are therefore dependent. None (the
        default) groups by ``study_id``. A :obj:`str` names a metadata field to group by
        instead, for a paper contributing independent samples (e.g. patients and controls).
        An array supplies one label per image. False treats every image as independent, which
        inflates significance whenever that is untrue. Default is None.
    weight_scheme : {'rescale', 'individual', 'collapse'}, optional
        How images within a group are weighted, passed to PyMARE. ``'rescale'`` (the default)
        divides each image's weight by its group size, so a group's total weight does not grow
        with the number of maps it contributed -- the correlated-effects model of
        :footcite:t:`hedges2010robust`, weighted as in :footcite:t:`fisher2015robumeta`.
        ``'individual'`` leaves the weights alone; ``'collapse'`` fits one row per group.
        Group labels also switch the standard errors to CR2 and the reference to a t
        distribution with Satterthwaite degrees of freedom :footcite:p:`tipton2015small`.
    rho : :obj:`float`, optional
        Assumed within-group correlation, in [0, 1]. Enters only through tau^2, to which
        results are weakly sensitive. Default is 0.8, matching ``robumeta``. Ignored when
        ``weight_scheme='individual'``.

    Notes
    -----
    Requires :term:`beta` and :term:`varcope` images.

    :meth:`fit` produces a :class:`~nimare.results.MetaResult` object with the following maps:

    ============== ===============================================================================
    "z"            Z-statistic map from one-sample test.
    "p"            P-value map from one-sample test.
    "est"          Fixed effects estimate for intercept test.
    "se"           Standard error of fixed effects estimate.
    "tau2"         Estimated between-study variance.
    "dof"          Degrees of freedom map from one-sample test.
    ============== ===============================================================================

    Warnings
    --------
    Cluster-robust inference is asymptotic in the number of *groups*, not images. PyMARE warns
    at 10 or fewer groups, where robust variance estimation is anti-conservative
    :footcite:p:`hedges2010robust`, and when the Satterthwaite degrees of freedom fall below
    about 4 :footcite:p:`tipton2015small`. Both are common for small meta-analyses.

    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will likely result in biased results. The extent of this bias is currently
    unknown.

    By default, all image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis. Setting ``aggressive_mask=False`` will
    instead run tha analysis in bags of voxels that have a valid value across
    the same studies.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`pymare.estimators.Hedges`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"beta_maps": ("image", "beta"), "varcope_maps": ("image", "varcope")}

    def _generate_description(self):
        description = (
            f"An image-based meta-analysis was performed with NiMARE {__version__} "
            "(RRID:SCR_017398; \\citealt{Salo2023}), on "
            f"{len(self.inputs_['id'])} beta and variance images using the Hedges "
            "method \\citep{hedges2014statistical}, in which tau-squared is estimated on a "
            "voxel-wise basis."
        )
        return description

    def _fit_model(self, beta_maps, varcope_maps, study_mask=None):
        """Fit the model to the data."""
        n_studies, n_voxels = beta_maps.shape

        if study_mask is None:
            # If no mask is provided, assume all studies are included. This is always the case
            # when using the aggressive mask.
            study_mask = np.arange(n_studies)

        # None when there is nothing to correct for, which is how PyMARE is told to use
        # model-based inference rather than cluster-robust ("sandwich") standard errors.
        groups = self._dependence(study_mask).labels

        pymare_dset = pymare.Dataset(y=beta_maps, v=varcope_maps, g=groups)
        est = pymare.estimators.Hedges(**self._pymare_weighting_kwargs(study_mask))
        est.fit_dataset(pymare_dset)
        est_summary = est.summary()

        fe_stats = est_summary.get_fe_stats()
        z_map = fe_stats["z"].squeeze()
        p_map = fe_stats["p"].squeeze()
        est_map = fe_stats["est"].squeeze()
        se_map = fe_stats["se"].squeeze()
        tau2_map = est_summary.tau2.squeeze()
        dof_map = self._fe_dof_map(est_summary, study_mask, n_voxels)
        return z_map, p_map, est_map, se_map, tau2_map, dof_map

    def _fit(self, dataset):
        self.dataset = dataset
        self.masker = self.masker or dataset.masker
        if not isinstance(self.masker, NiftiMasker):
            LGR.warning(
                f"A {type(self.masker)} mask has been detected. "
                "Masks which average across voxels will likely produce biased results when used "
                "with this Estimator."
            )

        if self.aggressive_mask:
            voxel_mask = self.inputs_["aggressive_mask"]
            result_maps = self._fit_model(
                self.inputs_["beta_maps"][:, voxel_mask],
                self.inputs_["varcope_maps"][:, voxel_mask],
            )

            z_map, p_map, est_map, se_map, tau2_map, dof_map = tuple(
                map(lambda x: _boolean_unmask(x, voxel_mask), result_maps)
            )
        else:
            n_voxels = self.inputs_["beta_maps"].shape[1]

            z_map, p_map, est_map, se_map, tau2_map = [
                np.zeros(n_voxels, dtype=float) for _ in range(5)
            ]
            dof_map = np.zeros(n_voxels, dtype=np.int32)

            beta_bags = self.inputs_["data_bags"]["beta_maps"]
            varcope_bags = self.inputs_["data_bags"]["varcope_maps"]
            for beta_bag, varcope_bag in zip(beta_bags, varcope_bags):
                (
                    z_map[beta_bag["voxel_mask"]],
                    p_map[beta_bag["voxel_mask"]],
                    est_map[beta_bag["voxel_mask"]],
                    se_map[beta_bag["voxel_mask"]],
                    tau2_map[beta_bag["voxel_mask"]],
                    dof_map[beta_bag["voxel_mask"]],
                ) = self._fit_model(
                    beta_bag["values"],
                    varcope_bag["values"],
                    beta_bag["study_mask"],
                )

        maps = {
            "z": z_map,
            "p": p_map,
            "est": est_map,
            "se": se_map,
            "tau2": tau2_map,
            "dof": dof_map,
        }
        description = self._description_text()

        return maps, {}, description


class SampleSizeBasedLikelihood(_PyMARERegressionEstimator):
    """Method estimates with known sample sizes but unknown sampling variances.

    .. versionchanged:: 0.20.1

        * New parameters: ``groupby``, ``weight_scheme`` and ``rho``, controlling how images
          contributed by the same participants are grouped and weighted.
        * The ``dof`` map now reports the Satterthwaite degrees of freedom of the CR2
          cluster-robust standard errors, which is what the p-values are drawn from. It is
          floating point rather than ``int32``, and varies by voxel under the liberal mask.

    .. versionchanged:: 0.2.1

        * New parameter: ``aggressive_mask``, to control whether to use an aggressive mask.

    .. versionchanged:: 0.0.12

        * Add "se" and "sigma2" to outputs.

    .. versionchanged:: 0.0.8

        * [FIX] Remove single-dimensional entries of each array of returns (:obj:`dict`).

    .. versionadded:: 0.0.4

    Iteratively estimates the between-subject variance tau^2 and fixed effect
    betas using the specified likelihood-based estimator (ML or REML).

    Parameters
    ----------
    aggressive_mask : :obj:`bool`, optional
        Voxels with a value of zero of NaN in any of the input maps will be removed
        from the analysis.
        If False, all voxels are included by running a separate analysis on bags
        of voxels that belong that have a valid value across the same studies.
        Default is True.
    groupby : None, :obj:`str`, array-like, or False, optional
        How to identify images that share participants and are therefore dependent. None (the
        default) groups by ``study_id``. A :obj:`str` names a metadata field to group by
        instead, for a paper contributing independent samples (e.g. patients and controls).
        An array supplies one label per image. False treats every image as independent, which
        inflates significance whenever that is untrue. Default is None.
    weight_scheme : {'rescale', 'individual', 'collapse'}, optional
        How images within a group are weighted, passed to PyMARE. ``'rescale'`` (the default)
        divides each image's weight by its group size, so a group's total weight does not grow
        with the number of maps it contributed -- the correlated-effects model of
        :footcite:t:`hedges2010robust`, weighted as in :footcite:t:`fisher2015robumeta`.
        ``'individual'`` leaves the weights alone; ``'collapse'`` fits one row per group.
        Group labels also switch the standard errors to CR2 and the reference to a t
        distribution with Satterthwaite degrees of freedom :footcite:p:`tipton2015small`.
    rho : :obj:`float`, optional
        Assumed within-group correlation, in [0, 1]. Enters only through tau^2, to which
        results are weakly sensitive. Default is 0.8, matching ``robumeta``. Ignored when
        ``weight_scheme='individual'``.
    method : {'ml', 'reml'}, optional
        The estimation method to use. The available options are

        ============== =============================
        "ml" (default) Maximum likelihood
        "reml"         Restricted maximum likelihood
        ============== =============================

    Notes
    -----
    Requires :term:`beta` images and sample size from metadata.

    :meth:`fit` produces a :class:`~nimare.results.MetaResult` object with the following maps:

    ============== ===============================================================================
    "z"            Z-statistic map from one-sample test.
    "p"            P-value map from one-sample test.
    "est"          Fixed effects estimate for intercept test.
    "se"           Standard error of fixed effects estimate.
    "tau2"         Estimated between-study variance.
    "sigma2"       Estimated within-study variance. Assumed to be the same for all studies.
    "dof"          Degrees of freedom map from one-sample test.
    ============== ===============================================================================

    Homogeneity of sigma^2 across studies is assumed.
    The ML and REML solutions are obtained via SciPy's scalar function
    minimizer (:func:`scipy.optimize.minimize`).
    Parameters to ``minimize()`` can be passed in as keyword arguments.

    Warnings
    --------
    Cluster-robust inference is asymptotic in the number of *groups*, not images. PyMARE warns
    at 10 or fewer groups, where robust variance estimation is anti-conservative
    :footcite:p:`hedges2010robust`, and when the Satterthwaite degrees of freedom fall below
    about 4 :footcite:p:`tipton2015small`. Both are common for small meta-analyses.

    Likelihood-based estimators are not parallelized across voxels, so this
    method should not be used on full brains, unless you can submit your code
    to a job scheduler.

    By default, all image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis. Setting ``aggressive_mask=False`` will
    instead run tha analysis in bags of voxels that have a valid value across
    the same studies.

    See Also
    --------
    :class:`pymare.estimators.SampleSizeBasedLikelihoodEstimator`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {
        "beta_maps": ("image", "beta"),
        "sample_sizes": ("metadata", "sample_sizes"),
    }

    def __init__(self, method="ml", **kwargs):
        super().__init__(**kwargs)
        self.method = method

    def _generate_description(self):
        description = (
            f"An image-based meta-analysis was performed with NiMARE {__version__} "
            "(RRID:SCR_017398; \\citealt{Salo2023}), on "
            f"{len(self.inputs_['id'])} beta images using sample size-based "
            "maximum likelihood estimation, in which tau-squared and sigma-squared are estimated "
            "on a voxel-wise basis."
        )
        return description

    def _fit_model(self, beta_maps, study_mask=None):
        """Fit the model to the data."""
        n_studies, n_voxels = beta_maps.shape

        if study_mask is None:
            # If no mask is provided, assume all studies are included. This is always the case
            # when using the aggressive mask.
            study_mask = np.arange(n_studies)

        sample_sizes = self._sample_sizes_for_mask(study_mask)
        n_maps = np.tile(sample_sizes, (n_voxels, 1)).T

        # None when there is nothing to correct for, which is how PyMARE is told to use
        # model-based inference rather than cluster-robust ("sandwich") standard errors.
        groups = self._dependence(study_mask).labels

        pymare_dset = pymare.Dataset(y=beta_maps, n=n_maps, g=groups)
        est = pymare.estimators.SampleSizeBasedLikelihoodEstimator(
            method=self.method,
            **self._pymare_weighting_kwargs(study_mask),
        )
        est.fit_dataset(pymare_dset)
        est_summary = est.summary()
        fe_stats = est_summary.get_fe_stats()

        z_map = fe_stats["z"].squeeze()
        p_map = fe_stats["p"].squeeze()
        est_map = fe_stats["est"].squeeze()
        se_map = fe_stats["se"].squeeze()
        tau2_map = est_summary.tau2.squeeze()
        sigma2_map = est.params_["sigma2"].squeeze()
        dof_map = self._fe_dof_map(est_summary, study_mask, n_voxels)
        return z_map, p_map, est_map, se_map, tau2_map, sigma2_map, dof_map

    def _fit(self, dataset):
        self.dataset = dataset
        self.masker = self.masker or dataset.masker

        if self.aggressive_mask:
            voxel_mask = self.inputs_["aggressive_mask"]
            result_maps = self._fit_model(
                self.inputs_["beta_maps"][:, voxel_mask],
            )

            z_map, p_map, est_map, se_map, tau2_map, sigma2_map, dof_map = tuple(
                map(lambda x: _boolean_unmask(x, voxel_mask), result_maps)
            )
        else:
            n_voxels = self.inputs_["beta_maps"].shape[1]

            z_map, p_map, est_map, se_map, tau2_map, sigma2_map = [
                np.zeros(n_voxels, dtype=float) for _ in range(6)
            ]
            dof_map = np.zeros(n_voxels, dtype=np.int32)

            for bag in self.inputs_["data_bags"]["beta_maps"]:
                (
                    z_map[bag["voxel_mask"]],
                    p_map[bag["voxel_mask"]],
                    est_map[bag["voxel_mask"]],
                    se_map[bag["voxel_mask"]],
                    tau2_map[bag["voxel_mask"]],
                    sigma2_map[bag["voxel_mask"]],
                    dof_map[bag["voxel_mask"]],
                ) = self._fit_model(bag["values"], bag["study_mask"])

        maps = {
            "z": z_map,
            "p": p_map,
            "est": est_map,
            "se": se_map,
            "tau2": tau2_map,
            "sigma2": sigma2_map,
            "dof": dof_map,
        }
        description = self._description_text()

        return maps, {}, description


class VarianceBasedLikelihood(_PyMARERegressionEstimator):
    """A likelihood-based meta-analysis method for estimates with known variances.

    .. versionchanged:: 0.20.1

        * New parameters: ``groupby``, ``weight_scheme`` and ``rho``, controlling how images
          contributed by the same participants are grouped and weighted.
        * The ``dof`` map now reports the Satterthwaite degrees of freedom of the CR2
          cluster-robust standard errors, which is what the p-values are drawn from. It is
          floating point rather than ``int32``, and varies by voxel under the liberal mask.

    .. versionchanged:: 0.2.1

        * New parameter: ``aggressive_mask``, to control whether to use an aggressive mask.

    .. versionchanged:: 0.0.12

        Add "se" output.

    .. versionchanged:: 0.0.8

        * [FIX] Remove single-dimensional entries of each array of returns (:obj:`dict`).

    .. versionadded:: 0.0.4

    Iteratively estimates the between-subject variance tau^2 and fixed effect
    coefficients using the specified likelihood-based estimator (ML or REML)
    :footcite:p:`dersimonian1986meta,kosmidis2017improving`.

    Parameters
    ----------
    aggressive_mask : :obj:`bool`, optional
        Voxels with a value of zero of NaN in any of the input maps will be removed
        from the analysis.
        If False, all voxels are included by running a separate analysis on bags
        of voxels that belong that have a valid value across the same studies.
        Default is True.
    groupby : None, :obj:`str`, array-like, or False, optional
        How to identify images that share participants and are therefore dependent. None (the
        default) groups by ``study_id``. A :obj:`str` names a metadata field to group by
        instead, for a paper contributing independent samples (e.g. patients and controls).
        An array supplies one label per image. False treats every image as independent, which
        inflates significance whenever that is untrue. Default is None.
    weight_scheme : {'rescale', 'individual', 'collapse'}, optional
        How images within a group are weighted, passed to PyMARE. ``'rescale'`` (the default)
        divides each image's weight by its group size, so a group's total weight does not grow
        with the number of maps it contributed -- the correlated-effects model of
        :footcite:t:`hedges2010robust`, weighted as in :footcite:t:`fisher2015robumeta`.
        ``'individual'`` leaves the weights alone; ``'collapse'`` fits one row per group.
        Group labels also switch the standard errors to CR2 and the reference to a t
        distribution with Satterthwaite degrees of freedom :footcite:p:`tipton2015small`.
    rho : :obj:`float`, optional
        Assumed within-group correlation, in [0, 1]. Enters only through tau^2, to which
        results are weakly sensitive. Default is 0.8, matching ``robumeta``. Ignored when
        ``weight_scheme='individual'``.
    method : {'ml', 'reml'}, optional
        The estimation method to use. The available options are

        ============== =============================
        "ml" (default) Maximum likelihood
        "reml"         Restricted maximum likelihood
        ============== =============================

    Notes
    -----
    Requires :term:`beta` and :term:`varcope` images.

    :meth:`fit` produces a :class:`~nimare.results.MetaResult` object with the following maps:

    ============== ===============================================================================
    "z"            Z-statistic map from one-sample test.
    "p"            P-value map from one-sample test.
    "est"          Fixed effects estimate for intercept test.
    "se"           Standard error of fixed effects estimate.
    "tau2"         Estimated between-study variance.
    "dof"          Degrees of freedom map from one-sample test.
    ============== ===============================================================================

    The ML and REML solutions are obtained via SciPy's scalar function
    minimizer (:func:`scipy.optimize.minimize`).
    Parameters to ``minimize()`` can be passed in as keyword arguments.

    Warnings
    --------
    Cluster-robust inference is asymptotic in the number of *groups*, not images. PyMARE warns
    at 10 or fewer groups, where robust variance estimation is anti-conservative
    :footcite:p:`hedges2010robust`, and when the Satterthwaite degrees of freedom fall below
    about 4 :footcite:p:`tipton2015small`. Both are common for small meta-analyses.

    Likelihood-based estimators are not parallelized across voxels, so this
    method should not be used on full brains, unless you can submit your code
    to a job scheduler.

    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will likely result in biased results. The extent of this bias is currently
    unknown.

    By default, all image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis. Setting ``aggressive_mask=False`` will
    instead run tha analysis in bags of voxels that have a valid value across
    the same studies.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`pymare.estimators.VarianceBasedLikelihoodEstimator`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"beta_maps": ("image", "beta"), "varcope_maps": ("image", "varcope")}

    def __init__(self, method="ml", **kwargs):
        super().__init__(**kwargs)
        self.method = method

    def _generate_description(self):
        description = (
            f"An image-based meta-analysis was performed with NiMARE {__version__} "
            "(RRID:SCR_017398; \\citealt{Salo2023}), on "
            f"{len(self.inputs_['id'])} beta and variance images using "
            "variance-based maximum likelihood estimation, in which tau-squared is estimated on a "
            "voxel-wise basis."
        )
        return description

    def _fit_model(self, beta_maps, varcope_maps, study_mask=None):
        """Fit the model to the data."""
        n_studies, n_voxels = beta_maps.shape

        if study_mask is None:
            # If no mask is provided, assume all studies are included. This is always the case
            # when using the aggressive mask.
            study_mask = np.arange(n_studies)

        # None when there is nothing to correct for, which is how PyMARE is told to use
        # model-based inference rather than cluster-robust ("sandwich") standard errors.
        groups = self._dependence(study_mask).labels

        pymare_dset = pymare.Dataset(y=beta_maps, v=varcope_maps, g=groups)
        est = pymare.estimators.VarianceBasedLikelihoodEstimator(
            method=self.method,
            **self._pymare_weighting_kwargs(study_mask),
        )
        est.fit_dataset(pymare_dset)
        est_summary = est.summary()
        fe_stats = est_summary.get_fe_stats()

        z_map = fe_stats["z"].squeeze()
        p_map = fe_stats["p"].squeeze()
        est_map = fe_stats["est"].squeeze()
        se_map = fe_stats["se"].squeeze()
        tau2_map = est_summary.tau2.squeeze()
        dof_map = self._fe_dof_map(est_summary, study_mask, n_voxels)
        return z_map, p_map, est_map, se_map, tau2_map, dof_map

    def _fit(self, dataset):
        self.dataset = dataset
        self.masker = self.masker or dataset.masker

        if not isinstance(self.masker, NiftiMasker):
            LGR.warning(
                f"A {type(self.masker)} mask has been detected. "
                "Masks which average across voxels will likely produce biased results when used "
                "with this Estimator."
            )

        if self.aggressive_mask:
            voxel_mask = self.inputs_["aggressive_mask"]
            result_maps = self._fit_model(
                self.inputs_["beta_maps"][:, voxel_mask],
                self.inputs_["varcope_maps"][:, voxel_mask],
            )

            z_map, p_map, est_map, se_map, tau2_map, dof_map = tuple(
                map(lambda x: _boolean_unmask(x, voxel_mask), result_maps)
            )
        else:
            n_voxels = self.inputs_["beta_maps"].shape[1]

            z_map, p_map, est_map, se_map, tau2_map = [
                np.zeros(n_voxels, dtype=float) for _ in range(5)
            ]
            dof_map = np.zeros(n_voxels, dtype=np.int32)

            beta_bags = self.inputs_["data_bags"]["beta_maps"]
            varcope_bags = self.inputs_["data_bags"]["varcope_maps"]
            for beta_bag, varcope_bag in zip(beta_bags, varcope_bags):
                (
                    z_map[beta_bag["voxel_mask"]],
                    p_map[beta_bag["voxel_mask"]],
                    est_map[beta_bag["voxel_mask"]],
                    se_map[beta_bag["voxel_mask"]],
                    tau2_map[beta_bag["voxel_mask"]],
                    dof_map[beta_bag["voxel_mask"]],
                ) = self._fit_model(
                    beta_bag["values"],
                    varcope_bag["values"],
                    beta_bag["study_mask"],
                )

        maps = {
            "z": z_map,
            "p": p_map,
            "est": est_map,
            "se": se_map,
            "tau2": tau2_map,
            "dof": dof_map,
        }
        description = self._description_text()

        return maps, {}, description


class PermutedOLS(IBMAEstimator):
    r"""An analysis with permuted ordinary least squares (OLS).

    .. versionchanged:: 0.20.1

        * New parameters: ``groupby``, identifying images that are statistically dependent,
          and ``use_sample_size``, weighting each group by its participant count.
        * The statistic is now computed over one contribution per group, and its reference is
          the Satterthwaite degrees of freedom of the CR2 cluster-robust standard error rather
          than a count of images.
        * Nilearn's :func:`~nilearn.mass_univariate.permuted_ols` is no longer called; the
          equivalent one-sample scheme is implemented in NiMARE so that exchangeability blocks
          work on every supported Nilearn version. The observed ``t`` map is unchanged.

    .. versionchanged:: 0.2.1

        * New parameter: ``aggressive_mask``, to control whether to use an aggressive mask.

    .. versionchanged:: 0.0.12

        * Use beta maps instead of z maps.

    .. versionchanged:: 0.0.8

        * [FIX] Remove single-dimensional entries of each array of returns (:obj:`dict`).

    .. versionadded:: 0.0.4

    This approach is described in :footcite:t:`freedman1983nonstochastic`.

    Parameters
    ----------
    aggressive_mask : :obj:`bool`, optional
        Voxels with a value of zero of NaN in any of the input maps will be removed
        from the analysis.
        If False, all voxels are included by running a separate analysis on bags
        of voxels that belong that have a valid value across the same studies.
        Default is True.
    groupby : None, :obj:`str`, array-like, or False, optional
        How to identify images that share participants and are therefore dependent. None (the
        default) groups by ``study_id``. A :obj:`str` names a metadata field to group by
        instead, for a paper contributing independent samples (e.g. patients and controls).
        An array supplies one label per image. False treats every image as independent, which
        inflates significance whenever that is untrue. Default is None.
    use_sample_size : :obj:`bool`, optional
        Whether to weight each group's contribution by its sample size. When False (the
        default), every group is weighted equally and the statistic is the ordinary one-sample
        t over group means. Default is False.
    two_sided : :obj:`bool`, optional
        If True, performs an unsigned t-test. Both positive and negative effects are considered;
        the null hypothesis is that the effect is zero. If False, only positive effects are
        considered as relevant. The null hypothesis is that the effect is zero or negative.
        Default is True.
    random_state : :obj:`int` or None, optional
        Seed for the sign-flip null. Default is 42.

    Notes
    -----
    Requires ``beta`` images, and ``sample_sizes`` metadata when ``use_sample_size=True``.

    Each group contributes the mean of its available images, so a study that uploaded fifty
    maps carries the same weight as one that uploaded a single map. Groups are then sign-flipped
    as whole exchangeability blocks :footcite:p:`winkler2014permutation`. With equal weights the
    statistic reduces algebraically to the ordinary one-sample t over group means; with sample
    size weights it is the intercept-only CR2 cluster-robust statistic of
    :footcite:t:`hedges2010robust`, referred to Satterthwaite degrees of freedom
    :footcite:p:`tipton2015small`.

    :meth:`fit` produces a :class:`~nimare.results.MetaResult` object with the following maps:

    ============== ===============================================================================
    "t"            T-statistic map from one-sample test.
    "z"            Z-statistic map from one-sample test.
    "dof"          Degrees of freedom map from one-sample test.
    ============== ===============================================================================

    Available correction methods: :func:`PermutedOLS.correct_fwe_montecarlo`

    Warnings
    --------
    With ``use_sample_size=True`` the degrees of freedom are Satterthwaite rather than a count
    of groups, and one dominant study can drag them well below the group count. PyMARE warns
    when they fall under about 4, where the approximation leaves the range
    :footcite:t:`tipton2015small` validated.

    By default, all image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis. Setting ``aggressive_mask=False`` will
    instead run tha analysis in bags of voxels that have a valid value across
    the same studies.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    nilearn.mass_univariate.permuted_ols : The Nilearn function this implementation is derived
        from, and which it reproduces for the ungrouped, unweighted case.
    """

    _required_inputs = {"beta_maps": ("image", "beta")}

    def __init__(
        self,
        two_sided=True,
        use_sample_size=False,
        n_jobs=1,
        random_state=42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_sample_size = use_sample_size
        if self.use_sample_size:
            self._required_inputs = dict(self._required_inputs)
            self._required_inputs["sample_sizes"] = ("metadata", "sample_sizes")
        self.two_sided = two_sided
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.parameters_ = {}

    def _generate_description(self):
        description = (
            f"An image-based meta-analysis was performed with NiMARE {__version__} "
            "(RRID:SCR_017398; \\citealt{Salo2023}), on "
            f"{len(self.inputs_['id'])} beta images using a permuted ordinary least squares "
            "method derived from Nilearn's \\citep{10.3389/fninf.2014.00014}."
        )
        if self.use_sample_size:
            description += " Each group was weighted by its sample size."
        return description

    def _blocks_and_weights(self, study_mask):
        """Return per-image block labels and one optional weight per block.

        ``groupby=False``, or a dataset in which no group holds more than one image, gives
        every image its own block, so collapsing is the identity and the ordinary one-sample
        test is recovered.
        """
        dependence = self._dependence(study_mask)
        if not self.use_sample_size:
            return dependence.blocks, dependence.group_order, None

        weights = dependence.per_group(self._sample_sizes_for_mask(study_mask))
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("Sample sizes must be finite positive values.")
        return dependence.blocks, dependence.group_order, weights

    def _fit_model(self, beta_maps, n_perm=0, study_mask=None, n_jobs=None, sign_flips=None):
        """Fit the model to the data."""
        n_maps, n_voxels = beta_maps.shape

        if study_mask is None:
            # If no mask is provided, assume all studies are included. This is always the case
            # when using the aggressive mask.
            study_mask = np.arange(n_maps)

        blocks, _, weights = self._blocks_and_weights(study_mask)

        result = _permuted_ols(
            beta_maps,
            exchangeability_blocks=blocks,
            group_weights=weights,
            n_perm=n_perm,
            two_sided_test=self.two_sided,
            random_state=self.random_state,
            n_jobs=self.n_jobs if n_jobs is None else n_jobs,
            sign_flips=sign_flips,
        )

        if n_perm:
            self.null_distributions_ = {
                "values_level-voxel_corr-fwe_method-montecarlo": result["h0_max_t"]
            }

        t_map = result["t"].squeeze()
        dof = result["dof"]
        z_map = t_to_z(t_map, dof)
        dof_map = np.full(n_voxels, dof, dtype=float)

        return result["logp_max_t"].squeeze(), t_map, z_map, dof_map

    def _fit(self, dataset):
        self.dataset = dataset

        if self.aggressive_mask:
            voxel_mask = self.inputs_["aggressive_mask"]
            result_maps = self._fit_model(self.inputs_["beta_maps"][:, voxel_mask])

            # Skip log_p_map
            t_map, z_map, dof_map = tuple(
                map(lambda x: _boolean_unmask(x, voxel_mask), result_maps[1:])
            )
        else:
            n_voxels = self.inputs_["beta_maps"].shape[1]
            t_map = np.zeros(n_voxels, dtype=float)
            z_map = np.zeros(n_voxels, dtype=float)
            dof_map = np.zeros(n_voxels, dtype=float)

            for bag in self.inputs_["data_bags"]["beta_maps"]:
                (
                    _,  # Skip log_p_map
                    t_map[bag["voxel_mask"]],
                    z_map[bag["voxel_mask"]],
                    dof_map[bag["voxel_mask"]],
                ) = self._fit_model(bag["values"], study_mask=bag["study_mask"])

        maps = {"t": t_map, "z": z_map, "dof": dof_map}
        description = self._description_text()

        return maps, {}, description

    def correct_fwe_montecarlo(self, result, n_iters=5000, n_cores=1):
        """Perform FWE correction using the max-value permutation method.

        .. versionchanged:: 0.20.1

            One sign-flip null is now shared across every liberal-mask bag, so the
            max-statistic distribution describes the whole brain rather than one bag of it.

        .. versionadded:: 0.0.4

        Only call this method from within a Corrector.

        Parameters
        ----------
        result : :obj:`~nimare.results.MetaResult`
            Result object from an image-based meta-analysis.
        n_iters : :obj:`int`, default=5000
            The number of iterations to run in estimating the null distribution.
            Default is 5000.
        n_cores : :obj:`int`, default=1
            Number of cores to use for parallelization.
            If <=0, defaults to using all available cores. Default is 1.

        Returns
        -------
        images : :obj:`dict`
            Dictionary of 1D arrays corresponding to masked images generated by
            the correction procedure. The following arrays are generated by
            this method: 'p_level-voxel', 'z_level-voxel', 'logp_level-voxel'.

        See Also
        --------
        nimare.correct.FWECorrector : The Corrector from which to call this method.

        Examples
        --------
        >>> meta = PermutedOLS()
        >>> result = meta.fit(dset)
        >>> corrector = FWECorrector(method='montecarlo',
                                     n_iters=5, n_cores=1)
        >>> cresult = corrector.transform(result)
        """
        if not isinstance(n_iters, (int, np.integer)) or n_iters <= 0:
            raise ValueError("n_iters must be a positive integer.")
        n_cores = _check_ncores(n_cores)

        n_images = len(self.inputs_["id"])
        # Sorted, so that searchsorted below can map each bag's labels onto the columns of
        # the shared sign-flip matrix.
        global_labels = np.unique(self._dependence().blocks)
        rng = np.random.RandomState(self.random_state)
        global_sign_flips = rng.choice((-1.0, 1.0), size=(n_iters, global_labels.size))

        if self.aggressive_mask:
            model_bags = [
                {
                    "values": self.inputs_["beta_maps"][:, self.inputs_["aggressive_mask"]],
                    "study_mask": np.arange(n_images),
                    "voxel_mask": self.inputs_["aggressive_mask"],
                }
            ]
        else:
            model_bags = self.inputs_["data_bags"]["beta_maps"]

        n_voxels = self.inputs_["beta_maps"].shape[1]
        observed_t = np.full(n_voxels, np.nan, dtype=float)
        h0_max_t = np.full(n_iters, -np.inf, dtype=float)
        for bag in model_bags:
            study_mask = bag["study_mask"]
            # group_order is by first occurrence, which is the column order _permuted_ols
            # expects; map those onto the shared matrix's sorted columns.
            local_indices = np.searchsorted(
                global_labels, self._dependence(study_mask).group_order
            )
            bag_result = self._fit_model(
                bag["values"],
                n_perm=n_iters,
                study_mask=study_mask,
                n_jobs=n_cores,
                sign_flips=global_sign_flips[:, local_indices],
            )
            observed_t[bag["voxel_mask"]] = bag_result[1]
            np.maximum(
                h0_max_t,
                self.null_distributions_["values_level-voxel_corr-fwe_method-montecarlo"],
                out=h0_max_t,
            )

        p_map = np.full(n_voxels, np.nan, dtype=float)
        valid = np.isfinite(observed_t)
        p_map[valid] = _empirical_max_p(observed_t[valid], h0_max_t, self.two_sided)

        sign = np.sign(observed_t)
        sign[sign == 0] = 1
        tail = "two" if self.two_sided else "one"
        z_map = p_to_z(p_map, tail=tail) * sign
        log_p_map = -np.log10(p_map)

        self.null_distributions_ = {"values_level-voxel_corr-fwe_method-montecarlo": h0_max_t}
        maps = {
            "p_level-voxel": p_map,
            "z_level-voxel": z_map,
            "logp_level-voxel": log_p_map,
        }
        description = (
            "Family-wise error rate correction was performed with a max-statistic null "
            "distribution generated by sign-flipping each group's contribution as one "
            "exchangeability block \\citep{winkler2014permutation}, following the permutation "
            "scheme of \\cite{freedman1983nonstochastic}. "
            f"{n_iters} iterations were performed to generate the null distribution."
        )

        return maps, {}, description


class FixedEffectsHedges(_PyMARERegressionEstimator):
    """Fixed Effects Hedges meta-regression estimator.

    .. versionchanged:: 0.20.1

        * New parameters: ``groupby``, ``weight_scheme`` and ``rho``, controlling how images
          contributed by the same participants are grouped and weighted.
        * The ``dof`` map now reports the Satterthwaite degrees of freedom of the CR2
          cluster-robust standard errors, which is what the p-values are drawn from. It is
          floating point rather than ``int32``, and varies by voxel under the liberal mask.

    .. versionadded:: 0.4.0

    Provides the weighted least-squares estimate of the fixed effects using Hedge's g
    as the point estimate and the variance of bias-corrected Cohen's d as the variance
    estimate, and given known/assumed between-study variance tau^2.
    When tau^2 = 0 (default), the model is the standard inverse-weighted
    fixed-effects meta-regression.

    This method was described in :footcite:t:`bossier2019`.

    Parameters
    ----------
    aggressive_mask : :obj:`bool`, optional
        Voxels with a value of zero of NaN in any of the input maps will be removed
        from the analysis.
        If False, all voxels are included by running a separate analysis on bags
        of voxels that belong that have a valid value across the same studies.
        Default is True.
    groupby : None, :obj:`str`, array-like, or False, optional
        How to identify images that share participants and are therefore dependent. None (the
        default) groups by ``study_id``. A :obj:`str` names a metadata field to group by
        instead, for a paper contributing independent samples (e.g. patients and controls).
        An array supplies one label per image. False treats every image as independent, which
        inflates significance whenever that is untrue. Default is None.
    weight_scheme : {'rescale', 'individual', 'collapse'}, optional
        How images within a group are weighted, passed to PyMARE. ``'rescale'`` (the default)
        divides each image's weight by its group size, so a group's total weight does not grow
        with the number of maps it contributed -- the correlated-effects model of
        :footcite:t:`hedges2010robust`, weighted as in :footcite:t:`fisher2015robumeta`.
        ``'individual'`` leaves the weights alone; ``'collapse'`` fits one row per group.
        Group labels also switch the standard errors to CR2 and the reference to a t
        distribution with Satterthwaite degrees of freedom :footcite:p:`tipton2015small`.
    rho : :obj:`float`, optional
        Assumed within-group correlation, in [0, 1]. Enters only through tau^2, to which
        results are weakly sensitive. Default is 0.8, matching ``robumeta``. Ignored when
        ``weight_scheme='individual'``.
    tau2 : :obj:`float` or 1D :class:`numpy.ndarray`, optional
        Assumed/known value of tau^2. Must be >= 0. Default is 0.

    Notes
    -----
    Requires `t` images and sample size from metadata.

    :meth:`fit` produces a :class:`~nimare.results.MetaResult` object with the following maps:

    ============== ===============================================================================
    "z"            Z-statistic map from one-sample test.
    "p"            P-value map from one-sample test.
    "est"          Fixed effects estimate for intercept test.
    "se"           Standard error of fixed effects estimate.
    "dof"          Degrees of freedom map from one-sample test.
    ============== ===============================================================================

    Warnings
    --------
    Cluster-robust inference is asymptotic in the number of *groups*, not images. PyMARE warns
    at 10 or fewer groups, where robust variance estimation is anti-conservative
    :footcite:p:`hedges2010robust`, and when the Satterthwaite degrees of freedom fall below
    about 4 :footcite:p:`tipton2015small`. Both are common for small meta-analyses.

    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will likely result in biased results. The extent of this bias is currently
    unknown.

    By default, all image-based meta-analysis estimators adopt an aggressive masking
    strategy, in which any voxels with a value of zero in any of the input maps
    will be removed from the analysis. Setting ``aggressive_mask=False`` will
    instead run tha analysis in bags of voxels that have a valid value across
    the same studies.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`pymare.estimators.WeightedLeastSquares`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"t_maps": ("image", "t"), "sample_sizes": ("metadata", "sample_sizes")}

    def __init__(self, tau2=0, **kwargs):
        super().__init__(**kwargs)
        self.tau2 = tau2

    def _generate_description(self):
        description = (
            f"An image-based meta-analysis was performed with NiMARE {__version__} "
            "(RRID:SCR_017398; \\citealt{Salo2023}), on "
            f"{len(self.inputs_['id'])} t-statistic images using Heges' g as point estimates "
            "and the variance of bias-corrected Cohen's in a Weighted Least Squares approach "
            "\\citep{brockwell2001comparison,bossier2019}, "
            f"with an a priori tau-squared value of {self.tau2} defined across all voxels."
        )
        return description

    def _fit_model(self, t_maps, study_mask=None):
        """Fit the model to the data."""
        n_studies, n_voxels = t_maps.shape

        if study_mask is None:
            # If no mask is provided, assume all studies are included. This is always the case
            # when using the aggressive mask.
            study_mask = np.arange(n_studies)

        sample_sizes = self._sample_sizes_for_mask(study_mask)
        n_maps = np.tile(sample_sizes, (n_voxels, 1)).T

        # Calculate Hedge's g maps: Standardized mean
        cohens_maps = t_to_d(t_maps, n_maps)
        hedges_maps, var_hedges_maps = d_to_g(cohens_maps, n_maps, return_variance=True)

        del n_maps, sample_sizes, cohens_maps

        # None when there is nothing to correct for, which is how PyMARE is told to use
        # model-based inference rather than cluster-robust ("sandwich") standard errors.
        groups = self._dependence(study_mask).labels

        pymare_dset = pymare.Dataset(y=hedges_maps, v=var_hedges_maps, g=groups)
        est = pymare.estimators.WeightedLeastSquares(
            tau2=self.tau2,
            **self._pymare_weighting_kwargs(study_mask),
        )
        est.fit_dataset(pymare_dset)
        est_summary = est.summary()

        fe_stats = est_summary.get_fe_stats()
        z_map = fe_stats["z"].squeeze()
        p_map = fe_stats["p"].squeeze()
        est_map = fe_stats["est"].squeeze()
        se_map = fe_stats["se"].squeeze()
        dof_map = self._fe_dof_map(est_summary, study_mask, n_voxels)
        return z_map, p_map, est_map, se_map, dof_map

    def _fit(self, dataset):
        self.dataset = dataset
        self.masker = self.masker or dataset.masker
        if not isinstance(self.masker, NiftiMasker):
            LGR.warning(
                f"A {type(self.masker)} mask has been detected. "
                "Masks which average across voxels will likely produce biased results when used "
                "with this Estimator."
            )

        if self.aggressive_mask:
            voxel_mask = self.inputs_["aggressive_mask"]
            result_maps = self._fit_model(self.inputs_["t_maps"][:, voxel_mask])

            z_map, p_map, est_map, se_map, dof_map = tuple(
                map(lambda x: _boolean_unmask(x, voxel_mask), result_maps)
            )
        else:
            n_voxels = self.inputs_["t_maps"].shape[1]

            z_map, p_map, est_map, se_map = [np.zeros(n_voxels, dtype=float) for _ in range(4)]
            dof_map = np.zeros(n_voxels, dtype=np.int32)

            for bag in self.inputs_["data_bags"]["t_maps"]:
                (
                    z_map[bag["voxel_mask"]],
                    p_map[bag["voxel_mask"]],
                    est_map[bag["voxel_mask"]],
                    se_map[bag["voxel_mask"]],
                    dof_map[bag["voxel_mask"]],
                ) = self._fit_model(bag["values"], bag["study_mask"])

        # tau2 is a float, not a map, so it can't go into the results dictionary
        tables = {"level-estimator": pd.DataFrame(columns=["tau2"], data=[self.tau2])}
        maps = {"z": z_map, "p": p_map, "est": est_map, "se": se_map, "dof": dof_map}
        description = self._description_text()

        return maps, tables, description
