"""Coordinate-based meta-regression.

CBMR estimates a smooth activation-intensity function from reported coordinates. The model is
specified by a formula in which each term states its own spatial resolution; see
:mod:`nimare.meta.cbmr.terms` for the syntax and :mod:`nimare.meta.cbmr.predictor` for how the
linear predictor is assembled.

What CBMR consumes from a Studyset
----------------------------------
Exactly two things, both produced by ``_collect_inputs`` from a single narrowed selection:

``blocks_["coordinates"]``
    A :class:`~nimare.studyset.blocks.CoordinateBlock`. Used for ``group_of_point()``, which
    gives each focus's analysis position, ``ijk(affine)``, which is memoised and truncates
    exactly as :func:`nimare.utils.mm2vox` does, and ``space``/``space_categories`` for the
    mixed-space check.
``studyset_``
    The narrowed selection itself, for ``ids`` and ``annotations_df``. The formula's terms are
    built against that frame.

The point of taking both from one selection is that the block's rows *are* the studyset's
analyses, in order, so the foci matrix and the annotation frame index the same experiments
without a lookup. The alternative -- fetching coordinates and annotations separately and
aligning them afterwards -- is how one experiment's moderator values come to be attributed to
another's foci, which nothing downstream could detect.

Deliberately not used:

* ``inputs_["coordinates"]``, the per-focus frame the ``Coordinates`` requirement also renders.
  Nothing here reads it. It costs about 19 ms on a 20,000-focus studyset and is memoised, so it
  is not worth an opt-out.
* ``Studyset.sample_sizes()``. Sample size reaches CBMR as an ordinary annotation column named
  in a formula, like any other moderator, rather than through a dedicated accessor.
* ``Studyset.harmonized(target)``. CBMR *rejects* mixed coordinate spaces rather than projecting
  them, matching what it has always done; harmonizing is the caller's decision because it is
  lossy. :meth:`_CBMRInputs._validate_block_space` names the method to call.

``_focus_positions`` -- mapping a coordinate block onto masked-voxel indices -- is the one piece
here that is not CBMR-specific. :mod:`nimare.meta.cbma.base` needs the same projection and
currently does the ``mm2vox`` half itself, on the frame and uncached. It is left local for now
because CBMR is its only block-based consumer; if CBMA moves onto blocks, this is the primitive
to promote rather than duplicate.
"""

import logging
import re

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.sparse

from nimare import _version
from nimare.estimator import Estimator
from nimare.meta.cbmr._helpers import (
    DEFAULT_GROUP_NAME,
    DEFAULT_INCIDENCE_THRESHOLD,
    _uses_cuda,
    _validate_incidence_threshold,
)
from nimare.meta.cbmr._torch import torch
from nimare.meta.cbmr.basis import b_spline_bases
from nimare.meta.cbmr.predictor import experiment_totals
from nimare.meta.cbmr.results import CBMRResult
from nimare.meta.cbmr.terms import DERIVED_EXPOSURE_COLUMN, FormulaError
from nimare.utils import get_masker, get_masker_mask_image, get_template, seed_torch

LGR = logging.getLogger(__name__)
__version__ = _version.get_versions()["version"]


class _CBMRInputs(Estimator):
    """Shared input preparation for CBMR.

    Turns reported coordinates into the three things a model needs: an analysis mask, a spline
    basis evaluated at its voxels, and an experiment-by-voxel matrix of focus counts. Split from
    the estimator because it is substantial, independent of how the model is specified, and the
    place where a silent misalignment would be least detectable -- the formula binds terms
    against the annotation table, so its row order has to match the foci matrix exactly.

    The analysis mask is narrower than the ROI mask: voxels whose empirical focus incidence is at
    or below ``incidence_threshold`` are dropped, since they carry no information about the
    intensity there and would only widen the basis.
    """

    _required_inputs = {"coordinates": ("coordinates", None)}
    _group_column = "_cbmr_group"

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

    @staticmethod
    def _validate_block_space(block):
        """Reject a coordinate block that mixes declared spaces.

        Checked on the block rather than on the coordinate frame, so no frame has to exist. The
        store harmonizes only when a target space is set, and neither the view context nor the
        Coordinates requirement sets one by default, so mixed spaces would otherwise reach the
        mask projection unconverted and land in the wrong voxels.
        """
        codes = np.unique(np.asarray(block.space))
        if codes.size == 0:
            raise ValueError(
                "Coordinate space information is missing. Ensure the studyset's coordinates "
                "declare a space."
            )
        if codes.size > 1:
            categories = list(block.space_categories)
            named = [str(categories[c]) if c < len(categories) else str(c) for c in codes]
            raise ValueError(
                "Mixed coordinate spaces detected in the studyset (space contains more than one "
                f"value: {', '.join(named[:10])}{'...' if len(named) > 10 else ''}). Call "
                "Studyset.harmonized(target) to project them into one space before running a "
                "meta-analysis."
            )

    def _focus_positions(self, block, mask_img, mask_data, mask_lookup):
        """Map every focus onto ``(experiment row, masked-voxel column)``.

        Both come straight off the coordinate block. ``ijk`` is memoised there and documented to
        truncate exactly as :func:`nimare.utils.mm2vox` does, and ``group_of_point`` gives each
        focus's analysis position directly -- which *is* the experiment row, because the block is
        aligned to ``studyset_`` by construction. Foci outside the mask are dropped.
        """
        ijk = block.ijk(mask_img.affine)
        rows = block.group_of_point()
        shape = np.asarray(mask_data.shape, dtype=np.int64)

        in_bounds = np.all((ijk >= 0) & (ijk < shape), axis=1)
        flat = np.ravel_multi_index(ijk[in_bounds].T, mask_data.shape)
        in_mask = mask_data.ravel()[flat]

        kept_rows = rows[in_bounds][in_mask]
        kept_columns = mask_lookup[flat[in_mask]]
        n_dropped = int(len(ijk) - kept_rows.size)
        if n_dropped:
            LGR.info(
                "%d/%d coordinates fall outside of the mask. Removing them.", n_dropped, len(ijk)
            )
        return kept_rows, kept_columns

    @staticmethod
    def _foci_matrix(rows, columns, n_experiments, n_mask_voxels, dtype=np.float64):
        """Return the experiment-by-voxel focus-count matrix.

        Repeated ``(row, column)`` pairs accumulate, so an experiment reporting two foci in one
        voxel gives that cell a count of two. Experiments with no surviving foci keep an all-zero
        row rather than disappearing, which is what keeps the rows aligned to the annotations.
        """
        return scipy.sparse.csr_matrix(
            (np.ones(rows.size, dtype=dtype), (rows, columns)),
            shape=(n_experiments, n_mask_voxels),
            dtype=dtype,
        )

    def _threshold_mask_by_incidence(self, mask_img, mask_data, foci, n_experiments):
        """Narrow an ROI mask to voxels whose empirical focus incidence clears the threshold.

        Incidence is the fraction of experiments reporting at least one focus in a voxel. Voxels
        below the threshold carry no information about the intensity there and would only widen
        the spline basis, so they are dropped before it is built.
        """
        if n_experiments == 0:
            raise ValueError("CBMR requires at least one experiment.")

        incidence_rate = np.asarray((foci > 0).sum(axis=0)).ravel() / float(n_experiments)
        self.inputs_["empirical_incidence_rate_roi"] = incidence_rate

        if self.incidence_threshold is None:
            keep_voxels = np.ones(incidence_rate.size, dtype=bool)
        else:
            keep_voxels = incidence_rate > self.incidence_threshold

        if not np.any(keep_voxels):
            raise ValueError(
                "No voxels survived CBMR incidence filtering. Lower incidence_threshold or "
                "provide a less restrictive mask."
            )

        thresholded = np.zeros(mask_data.size, dtype=bool)
        thresholded[np.flatnonzero(mask_data.ravel())[keep_voxels]] = True
        self.inputs_["empirical_incidence_rate"] = incidence_rate[keep_voxels]
        self.inputs_["incidence_threshold"] = self.incidence_threshold
        return self._mask_image_from_data(thresholded.reshape(mask_data.shape), mask_img)

    def _preprocess_input(self, dataset):
        """Build the analysis mask, the spline basis, and the foci matrix.

        Everything is derived from ``blocks_["coordinates"]`` and ``studyset_``, which
        ``_collect_inputs`` produced together from one narrowed selection. That is what makes the
        foci rows and the annotation rows the same rows -- the alternative, fetching coordinates
        and annotations separately and reindexing one onto the other, is where a silent
        misattribution of moderator values would come from.

        The mask is built twice on purpose: once as the ROI, to measure focus incidence, and once
        narrowed to the voxels that clear the threshold, which is what the basis is evaluated at.
        """
        block = self.blocks_["coordinates"]
        self._validate_block_space(block)
        n_experiments = len(self.studyset_.ids)

        roi_masker = self._resolve_roi_masker(dataset)
        _, roi_mask_img = get_masker_mask_image(roi_masker)
        roi_mask_data = np.asanyarray(roi_mask_img.dataobj).astype(bool, copy=False)
        roi_lookup, n_roi_voxels = self._build_mask_lookup(roi_mask_data)

        roi_rows, roi_columns = self._focus_positions(
            block, roi_mask_img, roi_mask_data, roi_lookup
        )
        roi_foci = self._foci_matrix(roi_rows, roi_columns, n_experiments, n_roi_voxels)

        analysis_mask_img = self._threshold_mask_by_incidence(
            roi_mask_img, roi_mask_data, roi_foci, n_experiments
        )
        self.masker = get_masker(analysis_mask_img)
        _, mask_img = get_masker_mask_image(self.masker)
        mask_data, mask_lookup, n_mask_voxels = self._initialize_spatial_inputs(
            self.masker, mask_img
        )

        rows, columns = self._focus_positions(block, mask_img, mask_data, mask_lookup)
        self.inputs_["foci"] = self._foci_matrix(rows, columns, n_experiments, n_mask_voxels)


class CBMR(_CBMRInputs):
    """Coordinate-based meta-regression specified by a formula.

    .. versionadded:: 0.21.0

    Where the previous interface took ``group_categories``, ``moderators``,
    ``global_moderators``, ``voxelwise_moderators`` and a three-valued ``moderator_effect``
    switch, this takes one formula in which each term states its own spatial resolution::

        CBMR("~ s(diagnosis:drug_status)")                 # a map per cell
        CBMR("~ s(diagnosis) + sample_size")               # plus a scalar moderator
        CBMR("~ s(diagnosis) + sample_size + s(avg_age)")  # one of each
        CBMR("~ s(diagnosis) + exposure()")                # conditioned on each study's count

    ``s()`` crosses a term with the spline basis, making its coefficient a map; without it the
    term gets a single coefficient. That is the whole global-versus-voxelwise distinction, and
    stating it per term removes the need for a separate "mixed" mode. It also reaches designs
    the older interface could not express at all -- ``s(sample_size)`` for a spatially varying
    moderator pooled across groups, ``s(diagnosis) + s(drug_status)`` for additive spatial main
    effects, ``diagnosis:sample_size`` for a group-specific scalar slope.

    ``exposure()`` is the one term that is not a moderator. It conditions the fit on each
    experiment's own foci count, so every spatial term estimates a distribution over voxels
    rather than a rate and a contrast between groups asks where foci fall rather than how many.
    That is a different estimand, and it has consequences -- a non-spatial term has nothing left
    to estimate beside it, and neither do the overdispersed distributions, so both are refused
    rather than fitted to zero. ``exposure(column)`` carries a quantity of the user's own and
    has none of those consequences. See :mod:`nimare.meta.cbmr.terms`.

    Parameters
    ----------
    formula : :obj:`str` or :class:`~nimare.meta.cbmr.terms.Design`
        Model specification. See :mod:`nimare.meta.cbmr.terms` for the syntax, including why
        there is never a scalar intercept and how the spatial baseline is resolved.
    distribution : :obj:`str` or :class:`~nimare.meta.cbmr.distributions.Distribution`, optional
        Observation distribution: ``"poisson"``, ``"negativebinomial"`` or
        ``"clusterednegativebinomial"``. The overdispersion models need several experiments
        sharing each spatial map, so they cannot be combined with a continuously varying spatial
        term; :meth:`~nimare.meta.cbmr.distributions.Distribution.check_design` explains why.
        Default is ``"poisson"``.
    mask : :obj:`str`, :class:`~nibabel.nifti1.Nifti1Image`, or Nilearn masker, optional
        Region-of-interest mask. If None, the whole 2 mm MNI152 brain mask is used.
    incidence_threshold : :obj:`float` or None, optional
        Drop voxels whose empirical focus incidence is at or below this, after applying ``mask``.
        Use None to keep every voxel in ``mask``. Default is 0.001.
    spline_spacing : :obj:`int`, optional
        Knot spacing of the cubic B-spline bases, shared across x, y and z. Smaller means a wider
        basis and so more coefficients per ``s()`` term. Default is 10.
    n_iter, lr, tol : optional
        L-BFGS iteration cap, learning rate and stopping tolerance.
    device : :obj:`str`, optional
        ``"cpu"`` or ``"cuda"``. Default is ``"cpu"``.
    random_state : :obj:`int`, optional
        Seed for weight initialization. Default is None.

    Notes
    -----
    The per-term parameter budget is logged at fit time. Each ``s()`` term costs one basis width
    of coefficients per column -- 457 at the default spacing on the 2 mm mask, as much as
    another group's entire baseline map -- which the older single switch hid by promoting every
    moderator at once.
    """

    def __init__(
        self,
        formula,
        distribution="poisson",
        mask=None,
        incidence_threshold=DEFAULT_INCIDENCE_THRESHOLD,
        spline_spacing=10,
        n_iter=1000,
        lr=1.0,
        tol=1e-8,
        device="cpu",
        random_state=None,
        **kwargs,
    ):
        from nimare.meta.cbmr.distributions import resolve_distribution
        from nimare.meta.cbmr.terms import formula_to_design

        self.design = formula_to_design(formula)
        self.distribution = resolve_distribution(distribution)

        self.mask = mask
        self.masker = get_masker(mask) if mask is not None else None
        self.incidence_threshold = _validate_incidence_threshold(incidence_threshold)
        self.spline_spacing = spline_spacing
        self.n_iter = n_iter
        self.lr = lr
        self.tol = tol
        self.device = device
        if _uses_cuda(self.device) and not torch.cuda.is_available():
            LGR.debug("CUDA not found; using device 'cpu'.")
            self.device = "cpu"
        self.random_state = random_state
        # Grouping is expressed by the formula, so input preparation treats the studyset as one
        # undifferentiated group.
        self.group_categories = None
        self.groups = None
        super().__init__(**kwargs)

        self.bound_design = None
        self.predictor = None
        self.cbmr_model = None

    def _make_result(self, dataset, maps=None, tables=None, description=""):
        """Return a result that can test hypotheses over the fitted design."""
        masker = self.masker or dataset.masker
        return CBMRResult(self, mask=masker, maps=maps, tables=tables, description=description)

    def _experiment_annotations(self):
        """Return the annotations of the analyses CBMR is fitting, in foci-matrix row order.

        No reindexing. ``studyset_`` is the selection ``_collect_inputs`` narrowed to analyses
        with usable coordinates, and the coordinate block was resolved from that same selection,
        so its rows and this frame's rows are the same analyses in the same order. Fetching the
        two separately and aligning them afterwards is what would let one experiment's moderator
        values be attributed to another's foci -- a mistake nothing downstream could detect.
        """
        annotations = self.studyset_.annotations_df
        expected = list(self.studyset_.ids)
        if list(annotations["id"]) != expected:
            raise ValueError(
                "The annotations frame is not in studyset order, so its rows cannot be trusted "
                "to line up with the foci matrix. This is a bug in the studyset layer rather "
                "than in the formula."
            )

        # A copy, because the frame belongs to the studyset and the generated column below does
        # not: it depends on this fit's mask and incidence_threshold, so leaving it behind would
        # give a later fit with different settings a column whose meaning had silently changed.
        annotations = annotations.copy()
        annotations[DERIVED_EXPOSURE_COLUMN] = experiment_totals(self.inputs_["foci"])
        return annotations

    def _log_exposure(self, annotations):
        """Report what the exposure does to the experiments that have none."""
        exposure = self.predictor.exposure
        if exposure is None:
            return
        empty = int(np.sum(np.asarray(exposure) == 0))
        if not empty:
            return
        LGR.info(
            f"{empty}/{len(exposure)} experiments have an exposure of zero. They are retained "
            "and stay aligned with their annotations; under an exposure they contribute nothing, "
            "because a study reporting no foci inside the mask says nothing about where foci "
            "fall."
        )

    def _fit(self, dataset):
        """Fit the formula-specified model and summarize it into maps and tables."""
        from nimare.meta.cbmr.model import CBMRModel
        from nimare.meta.cbmr.predictor import CBMRPredictor
        from nimare.meta.cbmr.terms import bind

        seed_torch(self.random_state, self.device)

        annotations = self._experiment_annotations()
        self.annotations_ = annotations
        self.bound_design = bind(self.design, annotations)
        _reject_moderators_under_a_derived_exposure(self.bound_design)
        self.predictor = CBMRPredictor(self.bound_design, self.inputs_["coef_spline_bases"])
        self._log_exposure(annotations)

        n_bases = self.predictor.n_bases
        LGR.info(
            f"CBMR design {self.bound_design.design} over {self.predictor.n_voxels} voxels "
            f"and {self.predictor.patterns.n_experiments} experiments, "
            f"{self.predictor.patterns.n_patterns} distinct spatial map(s):\n"
            + self.bound_design.describe(n_bases)
        )

        foci = self.inputs_["foci"]
        self.cbmr_model = CBMRModel(self.predictor, self.distribution, device=self.device)
        self.cbmr_model.fit(foci, n_iter=self.n_iter, lr=self.lr, tol=self.tol)

        maps, tables = self._summarize()
        return maps, tables, self._description_text()

    def _summarize(self):
        """Turn the fitted model into result maps and tables.

        Reported per *term*, not per spatial pattern. A design with a continuously varying
        spatial term has as many patterns as experiments, and forty fitted intensity maps are
        not a useful answer; the informative object is the term's coefficient map. So a baseline
        term yields one intensity map per level, and any other spatial term yields its
        coefficient map -- a derivative of log intensity with respect to that column, which is
        what ``voxelwiseModeratorEffect_`` has always meant.
        """
        maps, tables = {}, {}
        bases = self.predictor.bases
        coefficients = self.cbmr_model.fitted_coefficients()
        errors = self.cbmr_model.standard_errors()

        for block in self.bound_design.blocks:
            if block.term.exposure:
                # No coefficient, so no estimate, no standard error and no table. Reporting a
                # row for it would invite a hypothesis to be written over a fixed quantity.
                continue
            name = str(block.term)
            values = np.atleast_2d(coefficients[name])
            error_values = np.atleast_2d(errors[name])

            if not block.term.spatial:
                # est/se, not coefficient/standard_error: NiMARE's canonical map and table
                # vocabulary is ["z", "p", "logp", "est", "se", "dof"], and contrasts use it too.
                tables[f"moderatorEffect_{_table_safe(name)}"] = pd.DataFrame(
                    {
                        "column": list(block.column_names),
                        "est": values.reshape(-1),
                        "se": error_values.reshape(-1),
                    }
                )
                continue

            log_intensity = values @ bases.T
            for index, column in enumerate(block.column_names):
                label = _label_from_column(column)
                if block.is_baseline:
                    maps[f"spatialIntensity_group-{label}"] = np.exp(log_intensity[index])
                    maps[f"logSpatialIntensity_group-{label}"] = log_intensity[index]
                elif block.term.is_sum_to_zero:
                    # A constrained factor is not a moderator: its coefficients are contrasts
                    # among levels, measuring how a level shifts the baseline map.
                    factor = block.term.expr.replace(":", "-")
                    maps[f"spatialFactorEffect_{factor}-{label}"] = log_intensity[index]
                else:
                    maps[f"voxelwiseModeratorEffect_{label}"] = log_intensity[index]

            tables[f"spatialCoefficient_{_table_safe(name)}"] = pd.DataFrame(
                values, index=list(block.column_names)
            )
            tables[f"spatialCoefficientSE_{_table_safe(name)}"] = pd.DataFrame(
                error_values, index=list(block.column_names)
            )

        overdispersion = self.cbmr_model.overdispersion()
        if overdispersion is not None:
            tables["overdispersion"] = pd.DataFrame({"overdispersion": overdispersion})
        return maps, tables

    def _generate_description(self):
        """Describe the fitted model."""
        distribution_citations = {
            "Poisson": (
                "the Poisson model \\citep{eisenberg1966general}, which treats voxel-wise foci "
                "counts as independent Poisson variables"
            ),
            "NegativeBinomial": (
                "the negative binomial model \\citep{barndorff1969negative}, which allows "
                "excess variance relative to Poisson through latent variation at each voxel"
            ),
            "ClusteredNegativeBinomial": (
                "the clustered negative binomial model \\citep{geoffroy2001poisson}, whose "
                "latent effect belongs to an experiment and is shared across the brain"
            ),
        }
        distribution_text = distribution_citations.get(
            self.distribution.name, f"the {self.distribution.name} model"
        )
        n_bases = self.predictor.n_bases
        exposure_text = ""
        exposure_terms = self.bound_design.design.exposure_terms
        if exposure_terms:
            term = exposure_terms[0]
            quantity = (
                "each experiment's own foci count inside the analysis mask"
                if term.is_derived_exposure
                else f"the annotation {term.expr!r}"
            )
            exposure_text = (
                f" The model was conditioned on {quantity}, which was carried as an exposure "
                "with a fixed coefficient rather than fitted, so each spatial term estimates a "
                "distribution over voxels rather than a rate."
            )
        return (
            f"A coordinate-based meta-regression with the design {self.bound_design.design} was "
            f"fitted with NiMARE {__version__}, using {distribution_text}. Spatial structure was "
            f"parameterized by cubic B-spline bases at spacing {self.spline_spacing}, giving "
            f"{n_bases} bases over {self.predictor.n_voxels} analysis-mask voxels, for "
            f"{self.bound_design.n_parameters(n_bases)} coefficients across "
            f"{self.predictor.patterns.n_experiments} experiments and "
            f"{self.predictor.patterns.n_patterns} distinct spatial map(s)." + exposure_text
        )


def _reject_moderators_under_a_derived_exposure(bound_design):
    """Refuse a non-spatial term alongside ``exposure()``, whose estimate is zero regardless.

    Conditioning on each experiment's own total fits that total exactly. A non-spatial column
    reaches the data only through those totals, so its score is identically zero and its maximum
    likelihood estimate is exactly zero whatever the data -- reported, with an ordinary standard
    error, as a confident null. Refusing beats reporting it.
    """
    derived = [block.term for block in bound_design.blocks if block.term.is_derived_exposure]
    if not derived:
        return
    moderators = [
        block.term
        for block in bound_design.blocks
        if not block.term.spatial and not block.term.exposure
    ]
    if not moderators:
        return
    names = ", ".join(str(term) for term in moderators)
    spatial = " + ".join(f"s({term.expr})" for term in moderators)
    raise FormulaError(
        f"{names} has no coefficient to estimate alongside exposure(). Conditioning on each "
        "experiment's foci count fits its total exactly, and a non-spatial term acts only on "
        "that total, so its estimate is exactly zero whatever the data -- which would still be "
        "reported with a standard error, and read as a confident null. Two ways to write what "
        "you probably meant:\n"
        f"  ~ ... + {spatial}\n"
        "      ask whether the moderator changes *where* foci fall, which conditioning leaves "
        "intact and which is the question an exposure model can answer;\n"
        "  or model the totals separately\n"
        "      they are an ordinary count regression on the per-experiment foci count with one "
        "fixed effect per spatial map, and CBMR's coefficient for a non-spatial term is "
        "numerically identical to it. Fit that alongside, and report both.\n"
        "An exposure of your own, exposure(some_column), does not fit the totals exactly and so "
        "is not refused."
    )


def _table_safe(name):
    """Make a rendered term usable as a result-table key."""
    return name.replace(" ", "").replace("(", "-").replace(")", "").replace(":", "-")


def _label_from_column(column):
    """Turn a patsy column name into a map label.

    ``diagnosis[schiz]:drug[yes]`` becomes ``schiz-yes``, so a formula design produces the same
    readable group labels the older ``group_categories`` interface did. Names without a level --
    a continuous covariate, or the intercept -- pass through unchanged.
    """
    parts = []
    for piece in column.split(":"):
        match = re.search(r"\[(?:T\.)?([^\]]+)\]", piece)
        parts.append(match.group(1) if match else piece)
    label = "-".join(parts)
    return DEFAULT_GROUP_NAME if label in ("1", "Intercept") else label
