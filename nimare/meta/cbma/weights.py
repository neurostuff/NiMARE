"""Per-study weights for coordinate-based meta-analysis."""

import logging

import numpy as np
import pandas as pd

from nimare.base import NiMAREBase
from nimare.utils import _add_metadata_to_dataframe

LGR = logging.getLogger(__name__)

#: Values of the ``inference_field`` metadata that mark a fixed-effects study-level model.
DEFAULT_FIXED_EFFECTS_LABELS = ("fixed", "ffx", "fixed-effects", "fixed effects", "fe")

_TRANSFORMS = {
    "sqrt": np.sqrt,
    "linear": lambda n: n,
    "none": np.ones_like,
}


class StudyWeights(NiMAREBase):
    r"""Relative weights for the contrasts entering a meta-analysis.

    .. versionadded:: 0.5.0

    Implements the weighting scheme of :footcite:t:`wager2009evaluating`, in which each
    study contrast map is weighted by the square root of its sample size and contrasts
    analysed with a fixed-effects study-level model are discounted:

    .. math::

        w_c \propto \delta_c \sqrt{N_c}

    The weights this object returns are *relative*. The Estimator rescales them over the
    contrasts actually being analysed, which is what makes leave-one-out analyses
    renormalise correctly.

    Parameters
    ----------
    source : {"sample_size", "uniform"}, :obj:`dict`, or array-like, default="sample_size"
        Where the per-contrast quantity comes from. ``"sample_size"`` reads the
        collection's ``sample_sizes`` (or ``sample_size``) metadata. A dict maps study ID
        to a weight; an array-like gives one weight per study in the order the Estimator
        collected them. Explicit values bypass ``transform`` and ``reduce``, so they are
        the route for the "other study quality measures" of
        :footcite:t:`wager2009evaluating`.
    transform : {"sqrt", "linear", "none"}, default="sqrt"
        Function applied to the sample size. ``"sqrt"`` is the published method.
        ``"linear"`` weights by sample size directly and is *not* what
        :footcite:t:`wager2009evaluating` describes.
    reduce : {"mean", "sum", "min", "max"}, default="mean"
        How to collapse a contrast's ``sample_sizes`` list to one number. NiMARE's
        converters write one entry per contrast, so this rarely matters; ``"mean"``
        matches what the ALE kernel does with the same field.
    inference_field : :obj:`str` or None, default=None
        Metadata field naming each contrast's study-level inference model. Contrasts whose
        value matches ``fixed_effects_labels`` are multiplied by
        ``fixed_effects_discount``. With no field, no contrast is discounted -- NiMARE has
        no fixed/random convention to infer from, and guessing wrong biases every voxel.
    fixed_effects_discount : :obj:`float`, default=0.75
        The :math:`\delta` applied to fixed-effects contrasts.
        :footcite:t:`wager2009evaluating` uses 0.75, which is a convention rather than an
        estimate.
    fixed_effects_labels : :obj:`tuple` of :obj:`str`, optional
        Values of ``inference_field`` that mark a fixed-effects model. Matched
        case-insensitively after stripping whitespace.
    on_missing : {"impute", "raise"}, default="impute"
        What to do with contrasts whose weight is missing, zero or negative.
        ``"impute"`` replaces them with the mean of the valid weights and warns, which is
        what the CANlab MATLAB implementation does; it keeps the weighted and unweighted
        analyses over the same study set. ``"raise"`` refuses instead.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        source="sample_size",
        transform="sqrt",
        reduce="mean",
        inference_field=None,
        fixed_effects_discount=0.75,
        fixed_effects_labels=DEFAULT_FIXED_EFFECTS_LABELS,
        on_missing="impute",
    ):
        if isinstance(source, str) and source not in ("sample_size", "uniform"):
            raise ValueError(
                f"Invalid source '{source}'. Use 'sample_size', 'uniform', a mapping of "
                "study ID to weight, or an array of weights."
            )
        if transform not in _TRANSFORMS:
            raise ValueError(
                f"Invalid transform '{transform}'. Must be one of {sorted(_TRANSFORMS)}."
            )
        if reduce not in ("mean", "sum", "min", "max"):
            raise ValueError(
                f"Invalid reduce '{reduce}'. Must be one of 'mean', 'sum', 'min', 'max'."
            )
        if on_missing not in ("impute", "raise"):
            raise ValueError(f"Invalid on_missing '{on_missing}'. Must be 'impute' or 'raise'.")
        if not np.isfinite(fixed_effects_discount) or fixed_effects_discount <= 0:
            raise ValueError("fixed_effects_discount must be a positive, finite number.")

        self.source = source
        self.transform = transform
        self.reduce = reduce
        self.inference_field = inference_field
        self.fixed_effects_discount = fixed_effects_discount
        self.fixed_effects_labels = tuple(fixed_effects_labels)
        self.on_missing = on_missing

        #: Number of contrasts the last call to ``raw_weights`` discounted. Read by the
        #: Estimator when it writes the methods description.
        self.n_fixed_effects_ = 0
        #: Number of contrasts the last call to ``raw_weights`` imputed a weight for.
        self.n_imputed_ = 0

    def raw_weights(self, dataset, ids):
        """Return one unnormalised, strictly positive weight per study ID.

        Parameters
        ----------
        dataset : :obj:`~nimare.nimads.Studyset` or :obj:`~nimare.dataset.Dataset`
            Collection the Estimator is fitting.
        ids : array-like of :obj:`str`
            Study IDs to weight.

        Returns
        -------
        :class:`pandas.Series`
            Weights indexed by study ID. The Estimator rescales these.
        """
        ids = list(ids)
        if not ids:
            raise ValueError("Cannot compute weights for an empty set of study IDs.")

        weights = self._base_weights(dataset, ids)
        weights = weights * self._discounts(dataset, ids)
        return self._resolve_missing(weights)

    def _base_weights(self, dataset, ids):
        """Return the transformed per-contrast quantity, before the fixed-effects discount."""
        if isinstance(self.source, str) and self.source == "uniform":
            return pd.Series(1.0, index=ids, dtype=float)

        if not isinstance(self.source, str):
            return self._explicit_weights(ids)

        sample_sizes = self._sample_sizes(dataset, ids)
        # A non-positive sample size is meaningless and would survive sqrt as 0; let
        # _resolve_missing deal with it uniformly alongside genuinely absent values.
        sample_sizes = sample_sizes.where(sample_sizes > 0)
        transformed = _TRANSFORMS[self.transform](sample_sizes.to_numpy(dtype=float))
        return pd.Series(transformed, index=ids, dtype=float)

    def _explicit_weights(self, ids):
        """Return caller-supplied weights, aligned to ``ids``."""
        if isinstance(self.source, dict):
            return pd.Series(self.source, dtype=float).reindex(ids)

        values = np.asarray(self.source, dtype=float).ravel()
        if values.size != len(ids):
            raise ValueError(
                f"Received {values.size} weights for {len(ids)} studies. Pass a mapping of "
                "study ID to weight if the collection's ordering is not known."
            )
        return pd.Series(values, index=ids, dtype=float)

    def _sample_sizes(self, dataset, ids):
        """Return one sample size per study ID, reduced from the collection's metadata."""
        reducer = {"mean": np.mean, "sum": np.sum, "min": np.min, "max": np.max}[self.reduce]
        frame = _add_metadata_to_dataframe(
            dataset,
            pd.DataFrame({"id": ids}),
            metadata_field=("sample_sizes", "sample_size"),
            target_column="sample_size",
            filter_func=reducer,
        )
        if "sample_size" not in frame.columns:
            raise ValueError(
                "Sample-size weighting was requested, but the collection has no "
                "'sample_sizes' or 'sample_size' metadata. Populate it, or pass explicit "
                "weights via StudyWeights(source=...)."
            )
        return pd.Series(
            pd.to_numeric(frame["sample_size"], errors="coerce").to_numpy(dtype=float),
            index=ids,
        )

    def _discounts(self, dataset, ids):
        """Return delta per study ID: the fixed-effects discount, or 1.0."""
        self.n_fixed_effects_ = 0
        if self.inference_field is None:
            return pd.Series(1.0, index=ids, dtype=float)

        available = set(dataset.get_metadata())
        if self.inference_field not in available:
            raise ValueError(
                f"inference_field '{self.inference_field}' is not a metadata field of this "
                f"collection. Available fields: {', '.join(sorted(available))}."
            )

        raw = dataset.get_metadata(field=self.inference_field, ids=ids)
        labels = {label.strip().lower() for label in self.fixed_effects_labels}

        discounts = np.ones(len(ids), dtype=float)
        n_labelled = 0
        for i_study, value in enumerate(raw):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            n_labelled += 1
            if str(value).strip().lower() in labels:
                discounts[i_study] = self.fixed_effects_discount

        n_discounted = int(np.sum(discounts != 1.0))
        self.n_fixed_effects_ = n_discounted
        if n_labelled < len(ids):
            LGR.warning(
                f"Field '{self.inference_field}' is set for {n_labelled} of {len(ids)} "
                "contrasts. The unlabelled contrasts are treated as random-effects and are "
                "not discounted, which weights them above any contrast that is."
            )
        LGR.info(
            f"Discounting {n_discounted} fixed-effects contrasts by "
            f"{self.fixed_effects_discount}."
        )
        return pd.Series(discounts, index=ids)

    def _resolve_missing(self, weights):
        """Impute or refuse weights that are missing, zero or negative."""
        invalid = ~np.isfinite(weights.to_numpy(dtype=float)) | (weights.to_numpy() <= 0)
        self.n_imputed_ = 0
        if not invalid.any():
            return weights

        missing_ids = list(weights.index[invalid])
        if invalid.all():
            raise ValueError(
                "No contrast has a usable weight. Check the collection's sample-size " "metadata."
            )
        if self.on_missing == "raise":
            shown = ", ".join(str(study_id) for study_id in missing_ids[:25])
            suffix = f", ... (+{len(missing_ids) - 25} more)" if len(missing_ids) > 25 else ""
            raise ValueError(
                f"{len(missing_ids)} contrasts have a missing or non-positive weight: "
                f"{shown}{suffix}. Populate their sample sizes, or pass "
                "StudyWeights(on_missing='impute') to give them the mean weight."
            )

        weights = weights.copy()
        self.n_imputed_ = len(missing_ids)
        imputed = float(np.mean(weights.to_numpy(dtype=float)[~invalid]))
        weights[invalid] = imputed
        shown = ", ".join(str(study_id) for study_id in missing_ids[:5])
        suffix = f", ... (+{len(missing_ids) - 5} more)" if len(missing_ids) > 5 else ""
        LGR.warning(
            f"{len(missing_ids)} contrasts have a missing or non-positive weight "
            f"({shown}{suffix}); imputing the mean weight of the remaining "
            f"{int((~invalid).sum())} ({imputed:.4g})."
        )
        return weights


def resolve_weighting(weighting):
    """Return a :class:`StudyWeights`, or None, from an Estimator's ``weighting`` argument."""
    if weighting is None or isinstance(weighting, StudyWeights):
        return weighting
    if isinstance(weighting, str):
        if weighting in ("sample_size", "uniform"):
            return StudyWeights(source=weighting)
        raise ValueError(
            f"Invalid weighting '{weighting}'. Use None, 'sample_size', or a StudyWeights "
            "instance."
        )
    raise TypeError(
        f"weighting must be None, a string, or a StudyWeights instance, not {type(weighting)}."
    )


def normalize_weights(raw_weights, n_studies):
    """Rescale relative weights so they sum to ``n_studies``.

    With uniform inputs every weight is exactly 1.0, so an unweighted analysis is
    untouched and the statistic stays the count of activating contrasts. Dividing the
    result by ``n_studies`` recovers the weighted proportion of
    :footcite:t:`wager2009evaluating`.

    References
    ----------
    .. footbibliography::
    """
    weights = np.asarray(raw_weights, dtype=np.float64).ravel()
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Study weights must sum to a positive, finite number.")
    if np.all(weights == weights[0]):
        # Short-circuit rather than relying on n * c / sum(c) landing on exactly 1.0, so
        # that a weighted analysis of an equal-N corpus is bit-identical to an unweighted
        # one.
        return np.ones_like(weights)
    return n_studies * weights / total
