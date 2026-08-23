"""Image-based meta-analysis estimators."""

from __future__ import division

import logging
import re

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

from pymare.estimators.estimators import SMALL_SAMPLE_CORRECTIONS, WEIGHT_SCHEMES
from pymare.stats import estimate_null_correlation, undo_centering_shrinkage

from nimare import _version
from nimare.estimator import Estimator
from nimare.meta._dependence import (
    MIN_INDEPENDENT_UNITS,
    DependenceModel,
    hashable_label,
)
from nimare.meta._permutation import _empirical_max_p, _permuted_ols
from nimare.meta.utils import _liberal_mask_bags, _liberal_mask_values, _usable
from nimare.transforms import d_to_g, p_to_z, t_to_d, t_to_nlogp, t_to_z
from nimare.utils import (
    _check_ncores,
    _nlogp_to_logp_values,
    get_masker,
    get_masker_mask_image,
)

LGR = logging.getLogger(__name__)
__version__ = _version.get_versions()["version"]

#: Parameter descriptions and prose shared by the estimators below, substituted into their
#: docstrings by :func:`_fill_doc` so that a wording fix lands in one place. Written relative
#: to the placeholder: the first line of a block sits where the placeholder sat, and the rest
#: is indented from there.
_DOC_DICT = {
    "aggressive_mask": """aggressive_mask : :obj:`bool`, optional
    Voxels with a value of zero of NaN in any of the input maps will be removed
    from the analysis.
    If False, all voxels are included by running a separate analysis on bags
    of voxels that belong that have a valid value across the same studies.
    Default is False.""",
    "groupby": """groupby : None, :obj:`str`, array-like, or False, optional
    How to identify images that share participants and are therefore dependent. None (the
    default) groups by ``study_id``. A :obj:`str` names a metadata field to group by
    instead, for a paper contributing independent samples (e.g. patients and controls).
    An array supplies one label per image. False treats every image as independent, which
    inflates significance whenever that is untrue. Default is None.""",
    "weighting": """weight_scheme : {'rescale', 'individual', 'collapse'}, optional
    How images within a group are weighted, passed to PyMARE. ``'rescale'`` (the default)
    divides each image's weight by its group size, so a group's total weight does not grow
    with the number of maps it contributed -- the correlated-effects model of
    :footcite:t:`hedges2010robust`, weighted as in :footcite:t:`fisher2015robumeta`.
    ``'individual'`` leaves the weights alone; ``'collapse'`` fits one row per group.
    Group labels also switch the standard errors to CR2 and the reference to a t
    distribution with Satterthwaite degrees of freedom :footcite:p:`tipton2015small`.
small_sample_correction : None or {'knapp-hartung', 'knapp-hartung-conservative', 'wald'}, optional
    How PyMARE corrects its model-based standard errors for a small number of studies,
    passed through unchanged. The Wald reference treats an estimated tau^2 as if it were
    known, which is anti-conservative with few studies; the Knapp-Hartung adjustment
    rescales the variance by the observed weighted residual spread and refers it to a t
    distribution :footcite:p:`knapp2003improved`. Default is None, which leaves each
    PyMARE estimator on its own default -- the adjustment where tau^2 is estimated, and
    ``'wald'`` for :class:`WeightedLeastSquares`, whose tau^2 is supplied rather than
    estimated. Ignored once group labels are present, where CR2 and Satterthwaite degrees
    of freedom are the correction instead.
rho : :obj:`float`, optional
    Assumed within-group correlation, in [0, 1]. Enters only through tau^2, to which
    results are weakly sensitive. Default is 0.8, matching ``robumeta``. Ignored when
    ``weight_scheme='individual'``. Together the two choose a working model in the sense
    of :footcite:t:`pustejovsky2022expanding`: cluster-robust standard errors stay valid
    if it is wrong, so the choice costs precision rather than validity.""",
    "cluster_robust_warning": """Cluster-robust inference is asymptotic in the number of *groups*,
not images. PyMARE warns at 10 or fewer groups, where robust variance estimation is
anti-conservative :footcite:p:`hedges2010robust`, and when the Satterthwaite degrees of freedom
fall below about 4 :footcite:p:`tipton2015small`. Both are common for small meta-analyses.""",
    "liberal_mask_notes": """By default, image-based meta-analysis estimators run the analysis in
bags of voxels, where each bag holds the voxels that have a valid value across the same studies.
A voxel is therefore only dropped from the studies that are missing it. Setting
``aggressive_mask=True`` instead removes any voxel with a value of zero or NaN in any
input map from the analysis entirely. Either way, a bag whose valid images all belong to
one group is skipped -- one group cannot support the inference -- and its voxels come back
as NaN.""",
}

_DOC_PLACEHOLDER = re.compile(r"^(?P<indent> *)%\((?P<key>\w+)\)s *$")

#: Constructor arguments that used to exist, and what to tell a caller still passing one.
#: Without these, the generic "unexpected keyword argument" error points at the
#: ``resample__`` prefix, which for these names is the wrong advice: they are gone, not
#: renamed, and prefixing them would silently forward nonsense to
#: :func:`~nilearn.image.resample_to_img`.
_REMOVED_PARAMETERS = {
    "resample": (
        "resample was removed in 0.2.0. Images are now resampled only when their shape or "
        "affine differs from the mask's, which needs no switch. Pass 'resample__<name>' to "
        "control how that resampling is done."
    ),
    "memory_limit": (
        "memory_limit was removed in 0.2.0. Input images are loaded, resampled and masked "
        "one at a time, so peak memory no longer depends on a limit set here."
    ),
}


#: Variance at or below which a dependence group's aggregated z statistic carries no
#: information, because its images cancel. Matches the floor PyMARE's combination tests
#: refuse to aggregate below, so a group is excluded here before PyMARE raises on it. Not a
#: statistical threshold: it screens out exact cancellation and nothing else.
_CANCELLATION_FLOOR = 1e-12

#: How close an empirical correlation must be to -1 to count as a sign-flipped duplicate. A
#: float-error tolerance, not a warning band.
_MIRROR_TOLERANCE = 1e-6


def _mirrored_pairs(values):
    """Return the row pairs of ``values`` that are exact mirror images of each other.

    Parameters
    ----------
    values : :obj:`numpy.ndarray` of shape (K, V)
        Image data, one row per image.

    Returns
    -------
    :obj:`list` of :obj:`tuple`
        ``(i, j)`` row pairs, ``i < j``, whose *empirical* correlation is -1 to within
        :data:`_MIRROR_TOLERANCE`. Empirical rather than estimated-null: a mirror pair
        correlates -1 however much signal it shares, while its estimated null correlation is
        pulled well away from -1 by that shared signal.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        empirical = np.corrcoef(values)

    # np.corrcoef reports NaN against a constant row, which is not a mirror image.
    mirrored = np.isfinite(empirical) & (empirical <= -1.0 + _MIRROR_TOLERANCE)
    return list(zip(*np.nonzero(np.triu(mirrored, k=1))))


def _fill_doc(cls):
    """Substitute the shared blocks of :data:`_DOC_DICT` into a class docstring.

    Re-indents each block to its placeholder's own indentation rather than using ``%``
    formatting directly: Python 3.13 strips a docstring's common leading whitespace at
    compile time and earlier versions do not, so a block written at one fixed indentation
    would come out wrong on one of them.
    """
    if not cls.__doc__:
        return cls

    lines = []
    for line in cls.__doc__.split("\n"):
        match = _DOC_PLACEHOLDER.match(line)
        if match is None:
            lines.append(line)
            continue

        indent = match.group("indent")
        block = _DOC_DICT[match.group("key")]
        lines.extend(indent + text if text else "" for text in block.split("\n"))

    cls.__doc__ = "\n".join(lines)
    return cls


def _null_correlation(maps, groups):
    """Estimate the null correlation between images that share no set of valid voxels.

    Parameters
    ----------
    maps : :obj:`numpy.ndarray` of shape (K, V)
        One masked image per row.
    groups : :obj:`numpy.ndarray` of shape (K,)
        Group codes, one per image.

    Returns
    -------
    None or :obj:`numpy.ndarray` of shape (K, K)
        The estimated correlation matrix, or None when no pair of images has two voxels
        both are valid at.

    Notes
    -----
    Real NeuroVault maps carry an exact 0 where they have no coverage, and no two of them
    need cover the same voxels: on a 22-image staging studyset, 20.8% of the values are
    structural zeros and *no* voxel is valid in every image. Correlating all of them anyway
    correlates those placeholders, and centering is what makes that bite -- where two images
    are both uncovered their residuals are both minus the across-image mean, so they track
    each other perfectly. The bias runs in whichever direction the coverage pattern points:
    with a shared signal of twice the noise sd, a true correlation of 0 between images
    sharing a field of view is estimated at 0.67, while a true 0.5 between images with
    independent fields of view is estimated at 0.10.

    So each entry is estimated on the voxels where its own two images are both valid,
    rather than one voxel set being imposed on every entry.

    * The result is not guaranteed positive semi-definite, but neither is
      :func:`~pymare.stats.estimate_null_correlation`'s, which is indefinite on 9 of 14 real
      staging studysets. The combination tests read within-group blocks only, and
      :func:`~pymare.stats.undo_centering_shrinkage` floors each block's mean correlation at
      ``-1/(size - 1)``, which keeps every block sum non-negative.
    * A pair with fewer than two shared voxels is left at zero. Those entries are never
      read: a liberal-mask bag holds only images valid over the same voxels, so two images
      that share none never appear in one fit.
    * Centering over the images valid at each voxel, rather than over all K, shrinks the
      residual correlations by more than the bias correction inverts, so this is biased
      toward zero where coverage is sparse: about -0.05 at 70% coverage and -0.21 at 30%.
      Inverting the shrinkage at the depth actually centered over would recover most of it,
      but that derivation belongs in :func:`~pymare.stats.estimate_null_correlation`.
    """
    valid = _usable(maps)
    if valid.all():
        # Nothing to delete pairwise, which is where the aggressive-mask path always lands.
        return estimate_null_correlation(maps, groups=groups)

    # Strip the shared signal, as estimate_null_correlation does, but per voxel over the
    # images valid there. Uncovered entries are zeroed so the sums below can be matrix
    # products; the validity weights keep them out of every count.
    counts = valid.sum(axis=0)
    totals = np.where(valid, maps, 0.0).sum(axis=0, dtype=float)
    means = np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)
    residuals = np.where(valid, np.asarray(maps, dtype=float) - means, 0.0)

    # Pairwise-complete correlation for every pair at once. pandas.DataFrame.corr does the
    # same thing in one call and agrees to 1.5e-14, but its Cython pair loop costs 6.7x
    # these matrix products at 22 images and 20.7x at 100 (8.9 s against 0.43 s), since only
    # these reach BLAS.
    weights = valid.astype(float)
    pair_voxels = weights @ weights.T
    sums = residuals @ weights.T
    squares = (residuals * residuals) @ weights.T
    cross = residuals @ residuals.T
    with np.errstate(invalid="ignore", divide="ignore"):
        pair_means = sums / pair_voxels
        covariance = cross / pair_voxels - pair_means * pair_means.T
        variance = squares / pair_voxels - pair_means * pair_means
        corr = covariance / np.sqrt(variance * variance.T)

    estimable = np.isfinite(corr) & (pair_voxels >= 2)
    if not estimable[~np.eye(corr.shape[0], dtype=bool)].any():
        return None

    corr = np.where(estimable, corr, 0.0)
    # The two triangles are one quantity over one voxel set, so they differ only by rounding.
    corr = np.clip((corr + corr.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return undo_centering_shrinkage(corr, groups)


class IBMAEstimator(Estimator):
    """Base class for meta-analysis methods in :mod:`~nimare.meta`.

    .. warning::
        Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed in
        NiMARE 1.0.0. Prefer :class:`~nimare.nimads.Studyset`.

    .. versionchanged:: 0.21.0

        - New parameter: ``groupby``, identifying which images are statistically dependent on
          each other because they come from the same participants.
        - ``aggressive_mask`` now defaults to False, so voxels are no longer dropped just
          because they are missing from one input map.
        - ``generate_description`` is now accepted and forwarded, as it is for CBMA
          estimators. It was previously swallowed by ``**kwargs`` and had no effect.
        - Unrecognized keyword arguments now raise :obj:`TypeError` instead of being logged
          and ignored.
        - Fitting fewer than two analyses now raises :obj:`ValueError` rather than
          returning maps that are NaN at every voxel.
        - An image with no usable voxel is dropped and named; ``drop_invalid=False`` raises.

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

    #: The label each integer group code in ``inputs_["contrast_names"]`` stands for, so a
    #: warning can name the study rather than the code. Filled in by
    #: :meth:`_preprocess_dependence`.
    _group_labels = {}

    #: How this estimator reacts to a masker that averages across voxels (e.g. a
    #: :class:`~nilearn.maskers.NiftiLabelsMasker`) rather than only selecting them.
    #: ``"reject"`` for the combination tests, whose statistics are invalid once voxels are
    #: averaged; ``"warn"`` where the bias is real but unquantified; None where it does not
    #: apply.
    _voxel_averaging = "warn"

    #: Fewest analyses these estimators will fit; below it every output map is NaN. An
    #: analysis is at most one independent unit, so the floor on units is also the floor here.
    _min_analyses = MIN_INDEPENDENT_UNITS
    
    #: Which collected images ``_drop_empty_images`` kept, or None. ``groupby`` labels given
    #: as a sequence are positional, so nothing else narrows them alongside the images.
    _images_kept = None

    def __init__(
        self,
        aggressive_mask=False,
        memory=Memory(location=None, verbose=0),
        memory_level=0,
        generate_description=True,
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

        # Filled in only when a caller asks for masked images to be reused between fits; see
        # ``share_masked_image_cache``.
        self._masked_image_cache = None

        super().__init__(
            memory=memory,
            memory_level=memory_level,
            generate_description=generate_description,
        )

        # defaults for resampling images (nilearn's defaults do not work well)
        self._resample_kwargs = {
            "clip": True,
            "interpolation": "linear",
            "copy_header": True,
        }

        # Identify any kwargs
        resample_kwargs = {k: v for k, v in kwargs.items() if k.startswith("resample__")}

        # Reject any extraneous kwargs, rather than silently computing a plausible result
        # with settings the caller never asked for.
        other_kwargs = sorted(set(kwargs) - set(resample_kwargs))
        removed = [name for name in other_kwargs if name in _REMOVED_PARAMETERS]
        if removed:
            raise TypeError(" ".join(_REMOVED_PARAMETERS[name] for name in removed))

        if other_kwargs:
            raise TypeError(
                f"{type(self).__name__} got unexpected keyword argument(s): "
                f"{', '.join(repr(k) for k in other_kwargs)}. Resampling arguments must be "
                "prefixed with 'resample__'."
            )

        # Update the default resampling parameters
        resample_kwargs = {k.split("resample__")[1]: v for k, v in resample_kwargs.items()}
        self._resample_kwargs.update(resample_kwargs)

    def _check_enough_analyses(self, dataset):
        """Refuse to fit fewer than :attr:`_min_analyses` analyses.

        Parameters
        ----------
        dataset : :obj:`~nimare.nimads.Studyset` or :obj:`~nimare.dataset.Dataset`
            The collection ``fit`` was called with. Only its analysis count is read, taken
            before analyses lacking the required data were dropped.

        Raises
        ------
        ValueError
            If fewer than :attr:`_min_analyses` analyses carry the data this estimator needs.

        Notes
        -----
        One analysis is NaN at every voxel either way: under ``aggressive_mask`` its single
        model fails :attr:`~nimare.meta._dependence.DependenceModel.supports_inference`, and
        otherwise :func:`~nimare.meta.utils._liberal_mask_bags` drops every bag for holding
        fewer than two studies, leaving no skipped model to warn about.
        """
        n_used = len(self.inputs_["id"])
        if n_used >= self._min_analyses:
            return

        n_submitted = len(dataset.ids)
        dropped = ""
        if n_submitted > n_used:
            required = ", ".join(repr(name) for name in self._required_inputs)
            dropped = f"; the other {n_submitted - n_used} lacked required data ({required})"

        raise ValueError(
            f"{type(self).__name__} needs at least {self._min_analyses} analyses, but "
            f"{n_used} of the {n_submitted} submitted reached the estimator{dropped}. One "
            "analysis has no independent replication to estimate a variance from, so every "
            "output map would be NaN."
        )

    def _preprocess_input(self, dataset):
        """Preprocess inputs to the Estimator from the Dataset as needed.

        .. warning::
            Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed
            in NiMARE 1.0.0. Prefer :class:`~nimare.nimads.Studyset`.
        """
        # Before any image is loaded: the answer does not depend on their contents.
        self._check_enough_analyses(dataset)

        masker, mask_img = get_masker_mask_image(self.masker, dataset=dataset)

        # Whatever a previous fit dropped says nothing about this one.
        self._images_kept = None

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

        image_names = list(self._image_input_fields())
        for name in image_names:
            # Mask required input images using either the dataset's mask or the estimator's.
            temp_arr = self._mask_images(masker, mask_img, self.inputs_[name])
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

        # A model reads every image input at once, so an entry is usable only where all of
        # them are. Both masking strategies intersect over inputs for that reason -- and for
        # the liberal path it also keeps the per-input bag lists aligned, since a varcope
        # blanked above would otherwise cut its maps into different bags from its betas.
        validity = np.logical_and.reduce([_usable(self.inputs_[name]) for name in image_names])

        # Neither masking strategy notices an image with no usable voxel: the aggressive mask
        # intersects its empty row into every output map, and the liberal mask leaves it out
        # of every bag while it keeps its row in inputs_.
        usable = validity.any(axis=1)
        if not usable.all():
            self._drop_empty_images(usable, image_names)
            validity = validity[usable]

        if self.aggressive_mask:
            # Further reduce image-based inputs to remove "bad" voxels
            # (voxels with zeros or NaNs in any studies)
            self.inputs_["aggressive_mask"] = np.all(validity, axis=0)
            if n_bad_voxels := (
                self.inputs_["aggressive_mask"].size - self.inputs_["aggressive_mask"].sum()
            ):
                LGR.warning(f"Masking out {n_bad_voxels} additional voxels.")
        else:
            # One grouping for every input, worked out once. Besides being the cheaper half
            # of the work, it is what keeps the per-input bag lists aligned.
            bags = _liberal_mask_bags(validity)
            for name in image_names:
                self.inputs_["data_bags"][name] = [
                    {"values": values, "voxel_mask": voxel_mask, "study_mask": study_mask}
                    for values, (voxel_mask, study_mask) in zip(
                        _liberal_mask_values(self.inputs_[name], bags), bags
                    )
                ]

        self._preprocess_dependence(dataset)

    def _drop_empty_images(self, keep, image_names):
        """Drop the input images that hold no usable voxel, and realign every input to them.

        Parameters
        ----------
        keep : :obj:`numpy.ndarray`
            Boolean, one entry per collected image, False where every voxel is unusable.
        image_names : :obj:`list` of :obj:`str`
            The ``inputs_`` keys holding masked image arrays.

        Raises
        ------
        ValueError
            If ``drop_invalid`` is False, or if no image would be left.

        Notes
        -----
        Narrowing ``studyset_`` and collecting from it again is what keeps ``inputs_``,
        ``blocks_`` and ``studyset_`` describing the same analyses. The masked arrays are
        carried across so that no image is read off disk twice.
        """
        ids = np.asarray(self.inputs_["id"], dtype=str)
        n_total = ids.size
        dropped = ", ".join(ids[~keep])
        n_dropped = n_total - int(keep.sum())

        if not self._drop_invalid:
            raise ValueError(
                f"{n_dropped} of {n_total} images have no voxel with a usable value in every "
                f"required input, so they can contribute nothing: {dropped}. Pass "
                "drop_invalid=True to drop them and analyse the rest."
            )

        if not keep.any():
            raise ValueError(
                f"None of the {n_total} images have a voxel with a usable value in every "
                f"required input, so there is nothing to meta-analyse: {dropped}."
            )

        LGR.warning(
            "Dropping %d of %d image(s) with no voxel holding a usable value in every "
            "required input: %s.",
            n_dropped,
            n_total,
            dropped,
        )

        arrays = {name: self.inputs_[name][keep] for name in image_names}
        self._images_kept = keep
        self.studyset_ = self.studyset_.select_analyses(keep)
        self._collect_inputs(self.studyset_)
        self.inputs_.update(arrays)

    def _load_image(self, filename, mask_img):
        """Load one input image, resampling it only if its FOV differs from the mask's."""
        img = nib.load(filename)
        if check_same_fov(img, reference_masker=mask_img):
            return img

        return resample_to_img(img, mask_img, **self._resample_kwargs)

    def share_masked_image_cache(self, cache):
        """Reuse already-masked input images across repeated fits of the same studyset.

        A leave-one-out diagnostic fits this estimator once per study, over subsets of one
        fixed set of images. Masking is per-image and does not depend on which other images
        are in the fit, so the default -- reload, resample and mask every file on every fit --
        repeats the same work a number of times that grows with the square of the studyset.

        Parameters
        ----------
        cache : :obj:`dict` or None
            Mapping used to hold one masked row per image path. Callers that want the reuse
            to span several estimators (the copies a diagnostic refits) must hand the same
            dict to each of them; passing None turns the reuse off again. The caller owns the
            dict, and therefore its lifetime: entries live until it is dropped. It is only
            valid while the files it was filled from are unchanged.

        Notes
        -----
        The cached row is never handed out, only copied into each fit's own array, so a
        caller that edits ``inputs_`` cannot corrupt a later fit.
        """
        self._masked_image_cache = cache

    def _mask_images(self, masker, mask_img, filenames):
        """Return an (S x V) array holding the in-mask values of each input image.

        Where possible, images are loaded, resampled and masked one at a time. Concatenating
        them into a single 4D image first would hold a full-resolution copy of the entire
        studyset in memory -- at 2 mm that is roughly 7 MB per image, an order of magnitude
        more than the masked array it is immediately reduced to -- only to discard it.
        """
        filenames = list(filenames)
        if not filenames:
            raise ValueError(
                f"No images were found for a required input of {type(self).__name__}."
            )

        # Masking one image at a time only gives bit-for-bit the same answer when the masker
        # does nothing but select voxels, which is how NiMARE configures a NiftiMasker.
        # Standardizing, detrending, smoothing and label averaging all depend on how many
        # images the masker is handed at once, so they keep the 4D path.
        selects_voxels_only = (
            isinstance(masker, NiftiMasker)
            and not getattr(masker, "standardize", False)
            and not getattr(masker, "detrend", False)
            and getattr(masker, "smoothing_fwhm", None) is None
        )
        if not selects_voxels_only:
            imgs = [self._load_image(filename, mask_img) for filename in filenames]
            return masker.transform(concat_imgs(imgs, ensure_ndim=4))

        # Only the one-image-at-a-time path can be reused, since it is the only one whose
        # result for a file does not depend on the rest of the fit.
        cache = getattr(self, "_masked_image_cache", None)

        data = None
        for i, filename in enumerate(filenames):
            row = None if cache is None else cache.get(filename)
            if row is None:
                # A one-image list keeps the masker on its 4D code path; a bare 3D image
                # comes back as a 1D array instead.
                row = masker.transform([self._load_image(filename, mask_img)])[0]
                if cache is not None:
                    cache[filename] = row

            if data is None:
                data = np.empty((len(filenames), row.shape[0]), dtype=row.dtype)
            data[i] = row

        return data

    def _dependence(self, study_mask=None):
        """Return the :class:`~nimare.meta._dependence.DependenceModel` for the images fitted.

        ``study_mask`` holds the image indices the model is fitted over; None means every
        image, which is what a caller reasoning about the whole dataset wants. Inside
        ``_fit_model`` it is always explicit, supplied by :meth:`_fit_over_bags`.
        """
        model = DependenceModel(self.inputs_["contrast_names"])
        return model if study_mask is None else model.for_images(study_mask)

    def _group_name(self, code):
        """Return a human-readable name for one integer group code."""
        return str(self._group_labels.get(code, code))

    def _image_names(self, indices):
        """Return the ids of the images at ``indices``, as a comma-separated string."""
        ids = self.inputs_["id"]
        return ", ".join(str(ids[index]) for index in indices)

    def _screen_dependence_groups(self, study_mask, corr, stat_maps):
        """Judge each dependence group of one model before it is handed to PyMARE.

        Parameters
        ----------
        study_mask : :obj:`numpy.ndarray` of shape (K,)
            Indices into ``inputs_["id"]`` of the images this model is fitted over.
        corr : :obj:`numpy.ndarray` of shape (N, N)
            The estimated null correlation between every image, as ``inputs_["corr_matrix"]``
            holds it.
        stat_maps : :obj:`numpy.ndarray` of shape (K, V)
            The image data this model is fitted over, in ``study_mask`` order.

        Returns
        -------
        cancelling : :obj:`list` of :obj:`tuple`
            ``(code, images, variance)`` for every group whose images cancel. ``images``
            indexes ``inputs_["id"]``.
        mirrored : :obj:`list` of :obj:`tuple`
            ``(code, image_i, image_j)`` for every pair of images in a *surviving* group
            that are exact mirror images of each other.

        Notes
        -----
        Per model, because a group's membership is. Under a liberal mask a two-image group
        can only cancel where both images are valid, and is a lone contrast elsewhere.

        The two verdicts read different correlations. Cancellation is decided on the
        estimated null correlation, since that is the matrix PyMARE aggregates the group
        with; a sign-flipped duplicate is decided on the empirical one, which is what says
        two uploads are the same contrast both ways round. See :func:`_mirrored_pairs`.
        """
        dependence = self._dependence(study_mask)
        if not dependence.has_dependence:
            return [], []

        sub_corr = corr[np.ix_(study_mask, study_mask)]
        cancelling, mirrored = [], []
        for code, rows in dependence.group_rows():
            if rows.size < 2:
                # A group of one is aggregated as itself, with variance 1 by definition.
                continue

            # Mirrors StoufferCombinationTest._group_statistics, which computes this and
            # then raises rather than returning it. Kept in step by
            # test_cancellation_floor_still_matches_pymare.
            block = sub_corr[np.ix_(rows, rows)]
            variance = float(block.sum() / rows.size**2)
            if not np.isfinite(variance) or variance <= _CANCELLATION_FLOOR:
                cancelling.append((code, study_mask[rows], variance))
                continue

            mirrored.extend(
                (code, study_mask[rows[i]], study_mask[rows[j]])
                for i, j in _mirrored_pairs(stat_maps[rows])
            )

        return cancelling, mirrored

    def _resolve_masker(self, dataset):
        """Adopt the dataset's masker when none was supplied, and check it is usable here."""
        self.masker = self.masker or dataset.masker
        if self._voxel_averaging is None or isinstance(self.masker, NiftiMasker):
            return

        if self._voxel_averaging == "reject":
            raise ValueError(
                f"A {type(self.masker)} mask has been detected. "
                "Only NiftiMaskers are allowed for this Estimator. "
                "This is because aggregation, such as averaging values across ROIs, "
                "will produce invalid results."
            )

        LGR.warning(
            f"A {type(self.masker)} mask has been detected. "
            "Masks which average across voxels will likely produce biased results when used "
            "with this Estimator."
        )

    def _model_bags(self, input_names):
        """Yield ``(study_mask, voxel_mask, arrays)`` for each model to fit.

        One entry under ``aggressive_mask``, covering every image; otherwise one per
        liberal-mask bag. ``arrays`` holds one (K x V) array per name in ``input_names``,
        in that order.
        """
        if self.aggressive_mask:
            voxel_mask = self.inputs_["aggressive_mask"]
            n_images = len(self.inputs_["id"])
            arrays = [self.inputs_[name][:, voxel_mask] for name in input_names]
            yield np.arange(n_images), voxel_mask, arrays
            return

        # Every input was cut into the same bags by _preprocess_input, so the lists zip and
        # one bag's masks describe all of them.
        bag_lists = [self.inputs_["data_bags"][name] for name in input_names]
        for bags in zip(*bag_lists):
            yield bags[0]["study_mask"], bags[0]["voxel_mask"], [bag["values"] for bag in bags]

    def _fit_over_bags(self, input_names, map_names, **kwargs):
        """Run ``_fit_model`` over every model this estimator fits, and assemble its maps.

        Collects the outputs of ``_fit_model`` -- which come back in the order of
        ``map_names`` -- into full-length voxel maps. Voxels no model covers come back NaN,
        the same way out-of-mask voxels already did.

        Where a correlation matrix is passed through -- the combination tests, and only
        them -- :meth:`_screen_dependence_groups` screens each model first, and a dependence
        group whose images cancel is excluded from that model.

        Parameters
        ----------
        input_names : :obj:`list` of :obj:`str`
            The ``inputs_`` keys holding the image data, in the order ``_fit_model`` takes
            them as positional arguments.
        map_names : :obj:`list` of :obj:`str`
            Output map names, in the order ``_fit_model`` returns them.
        kwargs
            Passed through to ``_fit_model`` unchanged.

        Returns
        -------
        :obj:`dict` of :obj:`numpy.ndarray`
            One 1D map per entry in ``map_names``.
        """
        n_voxels = self.inputs_[input_names[0]].shape[1]
        maps = {name: np.full(n_voxels, np.nan, dtype=float) for name in map_names}

        corr = kwargs.get("corr")
        n_skipped = 0
        n_models = 0
        dropped, mirrored = {}, {}
        for study_mask, voxel_mask, arrays in self._model_bags(input_names):
            n_models += 1

            if corr is not None:
                # study_mask is filtered alongside the data because everything derived from
                # it -- the weights passed to PyMARE, the dof written back -- must keep
                # describing the fit that actually happened.
                cancelling, pairs = self._screen_dependence_groups(study_mask, corr, arrays[0])
                for code, images, variance in cancelling:
                    record = dropped.setdefault(
                        code, {"images": images, "models": 0, "voxels": 0, "variance": variance}
                    )
                    record["images"] = np.union1d(record["images"], images)
                    record["models"] += 1
                    record["voxels"] += arrays[0].shape[1]

                if cancelling:
                    keep = ~np.isin(
                        study_mask, np.concatenate([images for _, images, _ in cancelling])
                    )
                    study_mask = study_mask[keep]
                    arrays = [values[keep] for values in arrays]

                for code, first, second in pairs:
                    mirrored.setdefault((int(first), int(second)), code)

            if not self._dependence(study_mask).supports_inference:
                # Every image here comes from one group, so there is no independent
                # replication to estimate a variance from. Leave the voxels NaN rather than
                # report a statistic the data cannot support. Also catches a model the
                # drop above left with one group.
                n_skipped += 1
                continue

            results = self._fit_model(*arrays, study_mask=study_mask, **kwargs)
            for name, values in zip(map_names, results):
                maps[name][voxel_mask] = values

        self._report_screened_groups(dropped, mirrored, n_models)

        if n_skipped:
            LGR.warning(
                "Skipped %d set(s) of voxels whose valid images all come from a single group. "
                "One group carries no independent replication to estimate a variance from, so "
                "those voxels are NaN.",
                n_skipped,
            )

        return maps

    def _report_screened_groups(self, dropped, mirrored, n_models):
        """Warn once per dependence group excluded, and once per mirror pair kept.

        Aggregated over the models rather than emitted in the fitting loop, so a group
        dropped from every bag of a liberal-mask run is reported once with a count.
        """
        for code, record in dropped.items():
            LGR.warning(
                "Dependence group %s (image(s) %s) was excluded from %d of %d model(s), "
                "covering %d voxel(s): its aggregated z statistic has variance %.3g there, "
                "so its images cancel and carry no information. The rest were fitted. "
                "Usually one contrast was uploaded in both directions, or the group's "
                "contrasts partition one cohort.",
                self._group_name(code),
                self._image_names(record["images"]),
                record["models"],
                n_models,
                record["voxels"],
                record["variance"],
            )

        for (first, second), code in mirrored.items():
            LGR.warning(
                "Image %s and image %s, both in dependence group %s, correlate exactly -1: "
                "one is the other with its sign flipped. They do not cancel, so both were "
                "kept, but almost certainly one contrast was uploaded twice in opposite "
                "directions.",
                self._image_names([first]),
                self._image_names([second]),
                self._group_name(code),
            )

    def _dof_map(self, study_mask, n_voxels, est_summary=None):
        """Return the degrees of freedom map, one value per voxel.

        PyMARE reports Satterthwaite degrees of freedom whenever group labels reached a
        meta-regression estimator :footcite:p:`tipton2015small`; these are non-integer and
        vary by voxel. The combination tests have no covariance to derive them from, and
        neither does a fit without group labels, so both fall back to the group count.

        Always float, so that out-of-mask voxels come back as NaN like every other map. An
        integer map would carry ``INT_MIN`` there instead.
        """
        dof = None if est_summary is None else est_summary.fe_dof
        if dof is None:
            return np.full(n_voxels, self._dependence(study_mask).dof, dtype=float)

        # PyMARE returns (p, n_voxels) with p == 1, since every IBMA model is
        # intercept-only. Reshape rather than slice, so any other shape raises here instead
        # of being silently broadcast or clipped into the output map.
        return np.asarray(dof, dtype=float).reshape(n_voxels)

    def _sample_sizes_for_mask(self, study_mask):
        """Return per-image sample sizes aligned to a fitted model.

        Validation is left to PyMARE, which rejects non-finite or non-positive weights.
        """
        return np.asarray(
            [np.mean(self.inputs_["sample_sizes"][idx]) for idx in study_mask],
            dtype=float,
        )

    def _group_sample_sizes_for_mask(self, study_mask):
        """Return per-image sample sizes averaged within each dependence group.

        PyMARE's grouped combination tests take one weight per image but require every image
        in a group to agree on it, and a study routinely reports a different sample size per
        contrast. Averaging within the group is what :class:`PermutedOLS` already does to
        build its per-group weights; this is the same reduction, held at one value per image.
        """
        dependence = self._dependence(study_mask)
        return dependence.per_image(self._sample_sizes_for_mask(study_mask))

    @staticmethod
    def _combination_maps(est_summary):
        """Return the z, p and logp maps of a fitted PyMARE combination test.

        PyMARE evaluates all three in log space, so the z map and the logp map both carry
        tails past the smallest p-value a float can hold.
        """
        return (
            np.asarray(est_summary.z, dtype=float).squeeze(),
            np.asarray(est_summary.p, dtype=float).squeeze(),
            _nlogp_to_logp_values(np.asarray(est_summary.logp, dtype=float).squeeze()),
        )

    def _resolve_group_labels(self, dataset):
        """Return one group label per image, in ``inputs_["id"]`` order."""
        if isinstance(self.groupby, str):
            # Registered as a metadata requirement in __init__, so it is already
            # collected and aligned.
            return [hashable_label(v) for v in self.inputs_["dependence_groups"]]

        if self.groupby is not None and self.groupby is not False:  # explicit labels
            labels = [hashable_label(v) for v in np.asarray(self.groupby).ravel()]
            kept = self._images_kept
            if kept is not None and len(labels) == kept.size:
                # One label per collected image, so a dropped image takes its label with it.
                labels = [label for label, keep in zip(labels, kept) if keep]
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

        Populates ``contrast_names`` (an integer group index per image) and, for the
        combination tests only, ``corr_matrix`` (the empirical null correlation between the
        input maps). Estimators pass these on so that their inference accounts for the
        dependence.
        """
        labels = self._resolve_group_labels(dataset)

        if self.groupby is False:
            # Give every image its own label, so that "no dependence" is expressed in the
            # codes themselves rather than as a flag every consumer has to remember to read.
            # The resolved labels are still worth computing: they are what says whether the
            # caller has opted out of a correction they actually needed.
            repeated = len(labels) - len(set(labels))
            if repeated:
                LGR.warning(
                    f"{repeated} image(s) share a group with another image, but "
                    "groupby=False was requested, so they will be treated as independent. "
                    "This inflates significance."
                )
            labels = list(self.inputs_["id"])

        # Sorting keeps the code assignment stable across runs; plain set() iteration
        # depends on string hash randomization, which would make seeded permutations
        # irreproducible. str() first so a mix of label types still orders.
        label_to_int = {label: i for i, label in enumerate(sorted(set(labels), key=str))}
        self.inputs_["contrast_names"] = np.array([label_to_int[label] for label in labels])
        self._group_labels = {code: label for label, code in label_to_int.items()}

        dependence = self._dependence()
        if not dependence.has_dependence:
            # Every group contains exactly one image, so there is no within-group dependence
            # to correct for. DependenceModel owns that decision; see its `has_dependence`.
            return

        n_studies = len(self.inputs_["id"])
        n_groups = dependence.n_groups
        if not self._requires_corr_matrix:
            LGR.info(
                "Accounting for dependence among %d image(s) from %d group(s).",
                n_studies,
                n_groups,
            )
            return

        # Calculate the correlation matrix on the first image-based input,
        # which is the map the dependence acts on.
        image_names = list(self._image_input_fields())
        if not image_names:
            return
        maps = self.inputs_[image_names[0]]

        # Correlating the raw maps measures how much studies agree, not how dependent they
        # are: every map carries the same activation, so studies independent by construction
        # still come out correlated. The shared signal is stripped first, which is what
        # Brown's method and Stouffer's inflation term require, and the groups let the
        # shrinkage that centering induces be inverted.
        corr_matrix = _null_correlation(maps, self.inputs_["contrast_names"])
        if corr_matrix is None:
            LGR.warning(
                "No two of the %d image(s) have two voxels both are valid at, so the null "
                "correlation between them cannot be estimated. They are still grouped, but "
                "the reference distribution will not be inflated for the correlation within "
                "a group, so p-values may be anti-conservative.",
                n_studies,
            )
            return
        self.inputs_["corr_matrix"] = corr_matrix

        off_diagonal = corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)]
        LGR.info(
            "Correcting for dependence among %d image(s) from %d group(s) "
            "(max off-diagonal correlation %.3f).",
            n_studies,
            n_groups,
            np.nanmax(np.abs(off_diagonal)) if off_diagonal.size else np.nan,
        )


#: How to read each entry of ``_PyMARERegressionEstimator._extra_maps`` off a fitted PyMARE
#: estimator, so that a subclass names the extra maps it returns instead of restating the
#: whole unpacking.
_EXTRA_MAP_SOURCES = {
    "tau2": lambda est, summary: summary.tau2.squeeze(),
    "sigma2": lambda est, summary: est.params_["sigma2"].squeeze(),
}


class _PyMARERegressionEstimator(IBMAEstimator):
    """Base class for the IBMA estimators backed by a PyMARE meta-regression estimator.

    These all share the same two dependence parameters, because the PyMARE estimators
    underneath them do, and the same fit: build a PyMARE dataset, fit, and read the fixed
    effects off the summary. A subclass supplies :attr:`_pymare_estimator_class` and, where
    they differ from the defaults, :attr:`_image_inputs`, :attr:`_extra_maps`,
    :meth:`_pymare_estimator_kwargs`, :meth:`_pymare_dataset` and :meth:`_tables`.

    The combination tests (:class:`Fishers`, :class:`Stouffers`) have a different
    parameterization in PyMARE and so do not inherit from this.
    """

    #: The PyMARE estimator this wraps.
    _pymare_estimator_class = None

    #: ``inputs_`` keys holding image data, in the order ``_pymare_dataset`` reads them.
    _image_inputs = ("beta_maps", "varcope_maps")

    #: Maps returned between the shared "z"/"p"/"est"/"se" and the trailing "dof". Each name
    #: must have an entry in :data:`_EXTRA_MAP_SOURCES`.
    _extra_maps = ()

    def __init__(self, weight_scheme="rescale", rho=0.8, small_sample_correction=None, **kwargs):
        if weight_scheme not in WEIGHT_SCHEMES:
            raise ValueError(
                f"Invalid weight_scheme '{weight_scheme}'; must be one of "
                f"{sorted(WEIGHT_SCHEMES)}."
            )
        if small_sample_correction is not None and (
            small_sample_correction not in SMALL_SAMPLE_CORRECTIONS
        ):
            raise ValueError(
                f"Invalid small_sample_correction '{small_sample_correction}'; must be None "
                f"or one of {sorted(SMALL_SAMPLE_CORRECTIONS)}."
            )
        if not isinstance(rho, (int, float, np.integer, np.floating)) or isinstance(rho, bool):
            raise ValueError(f"Invalid rho {rho!r}; must be a number.")
        if not 0.0 <= float(rho) <= 1.0:
            raise ValueError(f"Invalid rho {rho!r}; must lie in [0, 1].")

        self.weight_scheme = weight_scheme
        self.rho = float(rho)
        self.small_sample_correction = small_sample_correction
        super().__init__(**kwargs)

    def _pymare_weighting_kwargs(self, study_mask):
        """Return the PyMARE weighting arguments for the current model.

        ``rho`` is omitted under ``weight_scheme="individual"``, which models no
        within-group correlation and warns if it is supplied anyway.
        """
        if self.weight_scheme == "individual" or self._dependence(study_mask).labels is None:
            return {"weight_scheme": self.weight_scheme}
        return {"weight_scheme": self.weight_scheme, "rho": self.rho}

    def _pymare_estimator_kwargs(self):
        """Return the arguments specific to :attr:`_pymare_estimator_class`."""
        return {}

    def _pymare_estimator(self, study_mask):
        """Return the PyMARE estimator to fit over the current model's images."""
        kwargs = {
            **self._pymare_estimator_kwargs(),
            **self._pymare_weighting_kwargs(study_mask),
        }
        # Omitted when None so that each PyMARE estimator keeps its own default, which
        # differs: the ones that estimate tau^2 correct by default, WeightedLeastSquares
        # does not, because its tau^2 is supplied rather than estimated.
        if self.small_sample_correction is not None:
            kwargs["small_sample_correction"] = self.small_sample_correction
        return self._pymare_estimator_class(**kwargs)

    def _pymare_dataset(self, arrays, study_mask):
        """Return the PyMARE dataset for the current model.

        The default is a beta map and its sampling variance; estimators that take something
        else, or that derive their inputs first, override this.
        """
        beta_maps, varcope_maps = arrays
        return pymare.Dataset(
            y=beta_maps,
            v=varcope_maps,
            g=self._dependence(study_mask).labels,
        )

    def _tables(self):
        """Return the non-map outputs of a fit."""
        return {}

    def _fit_model(self, *arrays, study_mask):
        """Fit the model to the data."""
        est = self._pymare_estimator(study_mask)
        est.fit_dataset(self._pymare_dataset(arrays, study_mask))
        est_summary = est.summary()

        fe_stats = est_summary.get_fe_stats()
        return (
            fe_stats["z"].squeeze(),
            fe_stats["p"].squeeze(),
            _nlogp_to_logp_values(fe_stats["logp"].squeeze()),
            fe_stats["est"].squeeze(),
            fe_stats["se"].squeeze(),
            *(_EXTRA_MAP_SOURCES[name](est, est_summary) for name in self._extra_maps),
            self._dof_map(study_mask, arrays[0].shape[1], est_summary),
        )

    def _fit(self, dataset):
        self.dataset = dataset
        self._resolve_masker(dataset)

        maps = self._fit_over_bags(
            list(self._image_inputs),
            ["z", "p", "logp", "est", "se", *self._extra_maps, "dof"],
        )
        return maps, self._tables(), self._description_text()


@_fill_doc
class Fishers(IBMAEstimator):
    """An image-based meta-analytic test using t- or z-statistic images.

    .. versionchanged:: 0.21.0

        * New parameter: ``groupby``, identifying images contributed by the same participants.
        * The ``dof`` map now counts independent groups rather than images.
        * The null correlation between images is estimated per pair, on the voxels both are
          valid at, rather than on the voxels every image is valid at.
        * A dependence group whose images cancel is now excluded from the voxels where it
          cancels, with a warning naming the study.

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
    %(aggressive_mask)s
    %(groupby)s
    use_sample_size : :obj:`bool`, optional
        Whether to assign each study a total weighted-Fisher coefficient equal
        to its sample size. Repeated images divide that coefficient internally
        in PyMARE, so image multiplicity does not change the study's total
        weight, and a group whose images report different sample sizes is weighted by
        their mean. Default is False, preserving ordinary Fisher/Brown inference.
    two_sided : :obj:`bool`, optional
        If True, performs an unsigned t-test. Both positive and negative effects are considered;
        the null hypothesis is that the effect is zero. If False, only positive effects are
        considered as relevant. The null hypothesis is that the effect is zero or negative.
        Default is True.

    Notes
    -----
    Requires ``z`` images.

    When ``groupby`` finds a group holding more than one image, Fisher's chi-squared reference
    is replaced by the scaled one of :footcite:t:`brown1975method`, whose scale factor is
    estimated from the null correlation between the input maps :footcite:p:`kost2002combining`.

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

    %(liberal_mask_notes)s

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

    # Averaging across voxels invalidates the combination outright.
    _voxel_averaging = "reject"

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

    def _fit_model(self, stat_maps, *, study_mask, corr=None):
        """Fit the model to the data."""
        n_voxels = stat_maps.shape[1]

        est = pymare.estimators.FisherCombinationTest(mode=self._mode)

        # When studies contribute several images, Brown's method replaces
        # Fisher's chi-squared reference with a scaled one.
        groups = self._dependence(study_mask).labels
        sub_corr = None
        if groups is not None and corr is not None:
            sub_corr = corr[np.ix_(study_mask, study_mask)]

        weights = None
        if self.use_sample_size:
            weights = self._group_sample_sizes_for_mask(study_mask)[:, None]

        # Group labels and optional weights are per-image, so they are passed
        # as single columns rather than tiled across every voxel.
        pymare_dset = pymare.Dataset(y=stat_maps, n=weights, g=groups)
        est.fit_dataset(pymare_dset, corr=sub_corr)
        est_summary = est.summary()

        z_map, p_map, logp_map = self._combination_maps(est_summary)
        dof_map = self._dof_map(study_mask, n_voxels)

        return z_map, p_map, logp_map, dof_map

    def _fit(self, dataset):
        self.dataset = dataset
        self._resolve_masker(dataset)

        maps = self._fit_over_bags(
            ["z_maps"],
            ["z", "p", "logp", "dof"],
            corr=self.inputs_["corr_matrix"],
        )
        return maps, {}, self._description_text()


@_fill_doc
class Stouffers(IBMAEstimator):
    """A t-test on z-statistic images.

    .. versionchanged:: 0.21.0

        * New parameter: ``groupby``, identifying images contributed by the same participants.
        * The ``dof`` map now counts independent groups rather than images.
        * The null correlation between images is estimated per pair, on the voxels both are
          valid at, rather than on the voxels every image is valid at.
        * A dependence group whose images cancel is now excluded from the voxels where it
          cancels, with a warning naming the study, instead of ending the meta-analysis.

    Requires z-statistic images.

    This method is described in :footcite:t:`stouffer1949american`.

    .. versionchanged:: 0.3.0

        * New parameter: ``two_sided``, controls the type of test to be performed. In addition,
            the default is now set to True (two-sided), which differs from previous versions
            where only one-sided tests were performed.
        * Add correction for multiple contrasts within a study.
        * New parameter: ``normalize_contrast_weights`` to normalized the weights by the
            number of contrasts in each study. Removed again in 0.21.0; see ``groupby``.

    .. versionchanged:: 0.2.1

        * New parameter: ``aggressive_mask``, to control whether to use an aggressive mask.

    Parameters
    ----------
    %(aggressive_mask)s
    %(groupby)s
    use_sample_size : :obj:`bool`, optional
        Whether to use sample sizes for weights (i.e., "weighted Stouffer's") or not,
        as described in :footcite:t:`zaykin2011optimally`. A group whose images report
        different sample sizes is weighted by their mean. Default is False.
    two_sided : :obj:`bool`, optional
        If True, performs an unsigned t-test. Both positive and negative effects are considered;
        the null hypothesis is that the effect is zero. If False, only positive effects are
        considered as relevant. The null hypothesis is that the effect is zero or negative.
        Default is True.

    Notes
    -----
    Requires ``z`` images and optionally the sample size metadata field.

    When ``groupby`` finds a group holding more than one image, the images in a group are
    combined into one variance-standardized statistic and the sum's variance is inflated by
    the null correlation between them :footcite:p:`kost2002combining`.

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

    %(liberal_mask_notes)s

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

    # Averaging across voxels invalidates the combination outright.
    _voxel_averaging = "reject"

    _required_inputs = {"z_maps": ("image", "z")}

    def __init__(
        self,
        use_sample_size=False,
        two_sided=True,
        **kwargs,
    ):
        if "normalize_contrast_weights" in kwargs:
            raise TypeError(
                "normalize_contrast_weights was removed in 0.21.0. Repeated images are now "
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

    def _fit_model(self, stat_maps, *, study_mask, corr=None):
        """Fit the model to the data."""
        n_studies, n_voxels = stat_maps.shape

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
            weights *= np.sqrt(self._group_sample_sizes_for_mask(study_mask))

        # Weights and group labels are per-image, not per-voxel, so they are
        # passed as columns rather than tiled across the whole map.
        pymare_dset = pymare.Dataset(y=stat_maps, n=weights[:, None], g=groups)
        est.fit_dataset(pymare_dset, corr=sub_corr)
        est_summary = est.summary()

        z_map, p_map, logp_map = self._combination_maps(est_summary)
        dof_map = self._dof_map(study_mask, n_voxels)

        return z_map, p_map, logp_map, dof_map

    def _fit(self, dataset):
        self.dataset = dataset
        self._resolve_masker(dataset)

        maps = self._fit_over_bags(
            ["z_maps"],
            ["z", "p", "logp", "dof"],
            corr=self.inputs_["corr_matrix"],
        )
        return maps, {}, self._description_text()


@_fill_doc
class WeightedLeastSquares(_PyMARERegressionEstimator):
    """Weighted least-squares meta-regression.

    .. versionchanged:: 0.21.0

        * New parameters: ``groupby``, ``weight_scheme`` and ``rho``.
        * The ``dof`` map now reports Satterthwaite degrees of freedom for the CR2 standard
          errors. It is floating point rather than ``int32``, and varies by voxel.

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
    %(aggressive_mask)s
    %(groupby)s
    %(weighting)s
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
    %(cluster_robust_warning)s

    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will likely result in biased results. The extent of this bias is currently
    unknown.

    %(liberal_mask_notes)s

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`pymare.estimators.WeightedLeastSquares`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"beta_maps": ("image", "beta"), "varcope_maps": ("image", "varcope")}

    _pymare_estimator_class = pymare.estimators.WeightedLeastSquares

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

    def _pymare_estimator_kwargs(self):
        return {"tau2": self.tau2}

    def _tables(self):
        # tau2 is an assumed constant here, not a map, so it can't go into the results
        # dictionary.
        return {"level-estimator": pd.DataFrame(columns=["tau2"], data=[self.tau2])}


@_fill_doc
class DerSimonianLaird(_PyMARERegressionEstimator):
    """DerSimonian-Laird meta-regression estimator.

    .. versionchanged:: 0.21.0

        * New parameters: ``groupby``, ``weight_scheme`` and ``rho``.
        * The ``dof`` map now reports Satterthwaite degrees of freedom for the CR2 standard
          errors. It is floating point rather than ``int32``, and varies by voxel.

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
    %(aggressive_mask)s
    %(groupby)s
    %(weighting)s

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
    %(cluster_robust_warning)s

    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will likely result in biased results. The extent of this bias is currently
    unknown.

    %(liberal_mask_notes)s

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`pymare.estimators.DerSimonianLaird`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"beta_maps": ("image", "beta"), "varcope_maps": ("image", "varcope")}

    _pymare_estimator_class = pymare.estimators.DerSimonianLaird
    _extra_maps = ("tau2",)

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


@_fill_doc
class Hedges(_PyMARERegressionEstimator):
    """Hedges meta-regression estimator.

    .. versionchanged:: 0.21.0

        * New parameters: ``groupby``, ``weight_scheme`` and ``rho``.
        * The ``dof`` map now reports Satterthwaite degrees of freedom for the CR2 standard
          errors. It is floating point rather than ``int32``, and varies by voxel.

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
    %(aggressive_mask)s
    %(groupby)s
    %(weighting)s

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
    %(cluster_robust_warning)s

    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will likely result in biased results. The extent of this bias is currently
    unknown.

    %(liberal_mask_notes)s

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`pymare.estimators.Hedges`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"beta_maps": ("image", "beta"), "varcope_maps": ("image", "varcope")}

    _pymare_estimator_class = pymare.estimators.Hedges
    _extra_maps = ("tau2",)

    def _generate_description(self):
        description = (
            f"An image-based meta-analysis was performed with NiMARE {__version__} "
            "(RRID:SCR_017398; \\citealt{Salo2023}), on "
            f"{len(self.inputs_['id'])} beta and variance images using the Hedges "
            "method \\citep{hedges2014statistical}, in which tau-squared is estimated on a "
            "voxel-wise basis."
        )
        return description


@_fill_doc
class SampleSizeBasedLikelihood(_PyMARERegressionEstimator):
    """Method estimates with known sample sizes but unknown sampling variances.

    .. versionchanged:: 0.21.0

        * New parameters: ``groupby``, ``weight_scheme`` and ``rho``.
        * The ``dof`` map now reports Satterthwaite degrees of freedom for the CR2 standard
          errors. It is floating point rather than ``int32``, and varies by voxel.

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
    %(aggressive_mask)s
    %(groupby)s
    %(weighting)s
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
    %(cluster_robust_warning)s

    Likelihood-based estimators are not parallelized across voxels, so this
    method should not be used on full brains, unless you can submit your code
    to a job scheduler.

    %(liberal_mask_notes)s

    See Also
    --------
    :class:`pymare.estimators.SampleSizeBasedLikelihoodEstimator`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {
        "beta_maps": ("image", "beta"),
        "sample_sizes": ("metadata", "sample_sizes"),
    }

    _pymare_estimator_class = pymare.estimators.SampleSizeBasedLikelihoodEstimator
    _image_inputs = ("beta_maps",)
    _extra_maps = ("tau2", "sigma2")

    # Sampling variance is estimated from the maps themselves here, so averaging voxels
    # rescales the estimate along with the effect rather than biasing one against the other.
    _voxel_averaging = None

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

    def _pymare_estimator_kwargs(self):
        return {"method": self.method}

    def _pymare_dataset(self, arrays, study_mask):
        (beta_maps,) = arrays
        sample_sizes = self._sample_sizes_for_mask(study_mask)
        # PyMARE accepts a (K x 1) column here and broadcasts it, which would be far
        # cheaper. It is tiled anyway because this estimator optimizes numerically, and the
        # two array layouts land on answers that differ in the last few digits; the saving
        # does not justify moving published results, and this estimator is documented as
        # unsuitable for whole-brain runs where the memory would matter.
        return pymare.Dataset(
            y=beta_maps,
            n=np.tile(sample_sizes, (beta_maps.shape[1], 1)).T,
            g=self._dependence(study_mask).labels,
        )


@_fill_doc
class VarianceBasedLikelihood(_PyMARERegressionEstimator):
    """A likelihood-based meta-analysis method for estimates with known variances.

    .. versionchanged:: 0.21.0

        * New parameters: ``groupby``, ``weight_scheme`` and ``rho``.
        * The ``dof`` map now reports Satterthwaite degrees of freedom for the CR2 standard
          errors. It is floating point rather than ``int32``, and varies by voxel.

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
    %(aggressive_mask)s
    %(groupby)s
    %(weighting)s
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
    %(cluster_robust_warning)s

    Likelihood-based estimators are not parallelized across voxels, so this
    method should not be used on full brains, unless you can submit your code
    to a job scheduler.

    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will likely result in biased results. The extent of this bias is currently
    unknown.

    %(liberal_mask_notes)s

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`pymare.estimators.VarianceBasedLikelihoodEstimator`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"beta_maps": ("image", "beta"), "varcope_maps": ("image", "varcope")}

    _pymare_estimator_class = pymare.estimators.VarianceBasedLikelihoodEstimator
    _extra_maps = ("tau2",)

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

    def _pymare_estimator_kwargs(self):
        return {"method": self.method}


@_fill_doc
class PermutedOLS(IBMAEstimator):
    r"""An analysis with permuted ordinary least squares (OLS).

    .. versionchanged:: 0.21.0

        * New parameters: ``groupby`` and ``use_sample_size``.
        * The statistic is computed over one contribution per group, referred to Satterthwaite
          degrees of freedom rather than a count of images.
        * Nilearn's :func:`~nilearn.mass_univariate.permuted_ols` is no longer called, so
          exchangeability blocks work on every supported Nilearn version. The ``t`` map is
          unchanged.
        * New output: an uncorrected ``p`` map

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
    %(aggressive_mask)s
    %(groupby)s
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
    maps carries the same weight as one that uploaded a single map. Groups are sign-flipped as
    whole exchangeability blocks :footcite:p:`winkler2014permutation`. Equal weights reduce the
    statistic to the ordinary one-sample t over group means; sample-size weights make it the
    intercept-only CR2 statistic of :footcite:t:`hedges2010robust`, referred to Satterthwaite
    degrees of freedom :footcite:p:`tipton2015small`.

    :meth:`fit` produces a :class:`~nimare.results.MetaResult` object with the following maps:

    ============== ===============================================================================
    "t"            T-statistic map from one-sample test.
    "z"            Z-statistic map from one-sample test.
    "p"            P-value map from one-sample test.
    "dof"          Degrees of freedom map from one-sample test.
    ============== ===============================================================================

    Available correction methods: :func:`PermutedOLS.correct_fwe_montecarlo`

    Warnings
    --------
    With ``use_sample_size=True`` the degrees of freedom are Satterthwaite rather than a group
    count, and one dominant study can drag them well below it. PyMARE warns below about 4
    :footcite:p:`tipton2015small`.

    %(liberal_mask_notes)s

    References
    ----------
    .. footbibliography::

    See Also
    --------
    nilearn.mass_univariate.permuted_ols : The Nilearn function this implementation is derived
        from, and which it reproduces for the ungrouped, unweighted case.
    """

    _required_inputs = {"beta_maps": ("image", "beta")}

    # The sign-flip null is distribution-free, so label averaging biases it no more than it
    # biases the maps it is given.
    _voxel_averaging = None

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

        The block order is :attr:`DependenceModel.group_order`, which callers that need it
        read from :meth:`_dependence` rather than having it handed back here twice.

        ``groupby=False``, or a dataset in which no group holds more than one image, gives
        every image its own block, so collapsing is the identity and the ordinary one-sample
        test is recovered.
        """
        dependence = self._dependence(study_mask)
        if not self.use_sample_size:
            return dependence.blocks, None

        weights = dependence.per_group(self._sample_sizes_for_mask(study_mask))
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("Sample sizes must be finite positive values.")
        return dependence.blocks, weights

    def _fit_model(self, beta_maps, *, study_mask, n_perm=0, n_jobs=None, sign_flips=None):
        """Fit the model to the data.

        With ``n_perm`` the max-statistic null is left in ``null_distributions_`` rather than
        returned, so the maps this hands back are the same three whatever ``n_perm`` is.
        """
        n_voxels = beta_maps.shape[1]
        blocks, weights = self._blocks_and_weights(study_mask)

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
        # Everything reported comes off one nlogp, so the four maps cannot disagree.
        nlogp_map = t_to_nlogp(t_map, dof, tail="two" if self.two_sided else "one")
        z_map = t_to_z(t_map, dof)

        return (
            t_map,
            z_map,
            np.exp(nlogp_map),
            _nlogp_to_logp_values(nlogp_map),
            np.full(n_voxels, dof, dtype=float),
        )

    def _fit(self, dataset):
        self.dataset = dataset
        self._resolve_masker(dataset)

        maps = self._fit_over_bags(["beta_maps"], ["t", "z", "p", "logp", "dof"])
        return maps, {}, self._description_text()

    def correct_fwe_montecarlo(self, result, n_iters=5000, n_cores=1):
        """Perform FWE correction using the max-value permutation method.

        .. versionchanged:: 0.21.0

            One sign-flip null is now shared across every liberal-mask bag, so the
            max-statistic distribution describes the whole brain rather than one bag of it.
            The maximum is taken over z rather than t, so that bags backed by different
            numbers of images are on a common scale.

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

        global_labels = np.unique(self._dependence().blocks)
        rng = np.random.RandomState(self.random_state)
        global_sign_flips = rng.choice((-1.0, 1.0), size=(n_iters, global_labels.size))

        n_voxels = self.inputs_["beta_maps"].shape[1]
        observed_z = np.full(n_voxels, np.nan, dtype=float)
        h0_max_z = np.full(n_iters, -np.inf, dtype=float)
        for study_mask, voxel_mask, (values,) in self._model_bags(["beta_maps"]):
            dependence = self._dependence(study_mask)
            if not dependence.supports_inference:
                # Skipped by _fit too, so these voxels have no observed statistic to correct.
                continue

            # group_order is by first occurrence, which is the column order _permuted_ols
            # expects; map those onto the shared matrix's sorted columns.
            local_indices = np.searchsorted(global_labels, dependence.group_order)
            _, bag_z, _, _, bag_dof = self._fit_model(
                values,
                study_mask=study_mask,
                n_perm=n_iters,
                n_jobs=n_cores,
                sign_flips=global_sign_flips[:, local_indices],
            )
            observed_z[voxel_mask] = bag_z

            bag_null = self.null_distributions_["values_level-voxel_corr-fwe_method-montecarlo"]
            np.maximum(h0_max_z, t_to_z(bag_null, bag_dof[0]), out=h0_max_z)

        p_map = np.full(n_voxels, np.nan, dtype=float)
        valid = np.isfinite(observed_z)
        p_map[valid] = _empirical_max_p(observed_z[valid], h0_max_z, self.two_sided)

        sign = np.sign(observed_z)
        sign[sign == 0] = 1
        tail = "two" if self.two_sided else "one"
        z_map = p_to_z(p_map, tail=tail) * sign
        nlogp_map = -np.log10(p_map)

        self.null_distributions_ = {"values_level-voxel_corr-fwe_method-montecarlo": h0_max_z}
        maps = {
            "p_level-voxel": p_map,
            "z_level-voxel": z_map,
            "logp_level-voxel": nlogp_map,
        }
        description = (
            "Family-wise error rate correction was performed with a max-statistic null "
            "distribution generated by sign-flipping each group's contribution as one "
            "exchangeability block \\citep{winkler2014permutation}, following the permutation "
            "scheme of \\cite{freedman1983nonstochastic}. The maximum was taken over "
            "z-statistics, so that voxels backed by different numbers of images contribute "
            "on a common scale. "
            f"{n_iters} iterations were performed to generate the null distribution."
        )

        return maps, {}, description


@_fill_doc
class FixedEffectsHedges(_PyMARERegressionEstimator):
    """Fixed Effects Hedges meta-regression estimator.

    .. versionchanged:: 0.21.0

        * New parameters: ``groupby``, ``weight_scheme`` and ``rho``.
        * The ``dof`` map now reports Satterthwaite degrees of freedom for the CR2 standard
          errors. It is floating point rather than ``int32``, and varies by voxel.

    .. versionadded:: 0.4.0

    Provides the weighted least-squares estimate of the fixed effects using Hedge's g
    as the point estimate and the variance of bias-corrected Cohen's d as the variance
    estimate, and given known/assumed between-study variance tau^2.
    When tau^2 = 0 (default), the model is the standard inverse-weighted
    fixed-effects meta-regression.

    This method was described in :footcite:t:`bossier2019`.

    Parameters
    ----------
    %(aggressive_mask)s
    %(groupby)s
    %(weighting)s
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
    %(cluster_robust_warning)s

    Masking approaches which average across voxels (e.g., NiftiLabelsMaskers)
    will likely result in biased results. The extent of this bias is currently
    unknown.

    %(liberal_mask_notes)s

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`pymare.estimators.WeightedLeastSquares`:
        The PyMARE estimator called by this class.
    """

    _required_inputs = {"t_maps": ("image", "t"), "sample_sizes": ("metadata", "sample_sizes")}

    _pymare_estimator_class = pymare.estimators.WeightedLeastSquares
    _image_inputs = ("t_maps",)

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

    def _pymare_estimator_kwargs(self):
        return {"tau2": self.tau2}

    def _pymare_dataset(self, arrays, study_mask):
        (t_maps,) = arrays
        # One sample size per image, as a column. t_to_d and d_to_g are elementwise, so this
        # broadcasts across voxels; tiling it to the full (K x V) shape first would allocate
        # tens of megabytes at 2 mm to hold K distinct numbers.
        sample_sizes = self._sample_sizes_for_mask(study_mask)[:, None]

        # Hedges' g: the standardized mean, with the variance of bias-corrected Cohen's d.
        cohens_maps = t_to_d(t_maps, sample_sizes)
        hedges_maps, var_hedges_maps = d_to_g(cohens_maps, sample_sizes, return_variance=True)
        del cohens_maps

        return pymare.Dataset(
            y=hedges_maps,
            v=var_hedges_maps,
            g=self._dependence(study_mask).labels,
        )

    def _tables(self):
        # tau2 is an assumed constant here, not a map, so it can't go into the results
        # dictionary.
        return {"level-estimator": pd.DataFrame(columns=["tau2"], data=[self.tau2])}
