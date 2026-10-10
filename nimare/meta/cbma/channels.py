r"""Evidence channels: what each kind of record is *entitled* to contribute.

Section 2.3 of the revised plan asks for observation channels "rather than coercing every record
into bounds", and section 5 calls the separation the first implementation priority. The reason is
that the three kinds of evidence a combined meta-analysis sees are not weaker and stronger
versions of one another -- they license different inferences, and the failure mode is a record
being read as though it licensed more than it does.

======================================  =========================================================
Channel                                 Allowed contribution
======================================  =========================================================
:attr:`~EvidenceChannel.IMAGE`          a direct effect likelihood
:attr:`~EvidenceChannel.QUALIFIED_TABLE`  a peak-selection likelihood, under a *stated* reporting
                                        event
:attr:`~EvidenceChannel.EXPLICIT_NONSIGNIFICANCE`  a scalar censoring interval
:attr:`~EvidenceChannel.POORLY_ANNOTATED_COORDINATES`  spatial features only -- never amplitude,
                                        never a below-threshold observation
:attr:`~EvidenceChannel.UNKNOWN_COVERAGE`  nothing
======================================  =========================================================

Two refusals are the point of the module.

**A location is not a magnitude.** Coordinates with unknown reporting provenance carry
information about where a literature looks, which is a spatial basis
(:mod:`nimare.meta.cbma.spatial`), and nothing about how large an effect was. Report *frequency*
is not effect magnitude, and treating it as one is the error that makes a coordinate-only
estimator look precise.

**An image and its own table are one study, not two.** They share subjects entirely, so counting
both as independent effect observations double-counts. :class:`EvidenceLedger` tracks study and
cohort identity and refuses it. Paired records remain useful for *calibrating a reporting model*,
where their dependence is the object of interest rather than a nuisance.

Notes
-----
**Standardisation is per design, not per sample size.** :func:`standardised_scale` covers the
one-sample/paired convention (:math:`\kappa = 1/\sqrt n`) and the independent equal-variance
two-group one (:math:`\kappa = \sqrt{1/n_1 + 1/n_2}`) and refuses everything else, because a
total sample size does not identify the conversion for a covariate-adjusted GLM, a mixed-effects
group map, a correlation or an F map. Section 2.1 is explicit about this, and a silent default
here would put an unstated design assumption underneath every downstream number.

**Hedges' correction acts on the estimator, not on the future design.** :func:`hedges_factor`
returns :math:`J(\nu)` for correcting an *observed* estimate toward its population standardised
effect. It must not be applied again to a planned study's non-centrality: the population effect
is already what :mod:`nimare.meta.cbma.planning` expects.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np
from scipy.special import gammaln

#: Designs whose standardisation is determined by the counts alone. Anything else is refused.
SUPPORTED_DESIGNS = ("one-sample", "paired", "two-sample")


class EvidenceChannel(enum.Enum):
    """The kind of evidence a record carries, and therefore what it may contribute.

    Attributes
    ----------
    IMAGE
        An unthresholded effect image whose sampling scale and contrast are understood.
    QUALIFIED_TABLE
        A coordinate table whose reporting event is known, or explicitly conditioned on a stated
        scenario. Contributes a peak-selection likelihood.
    EXPLICIT_NONSIGNIFICANCE
        A prespecified scalar test genuinely known to have been non-significant. The one record
        type that is a valid scalar censoring interval.
    POORLY_ANNOTATED_COORDINATES
        Locations with incomplete reporting knowledge. Spatial features only.
    UNKNOWN_COVERAGE
        Completeness or mask unknown. Contributes nothing, and in particular licenses no
        below-threshold observation anywhere.
    """

    IMAGE = "image"
    QUALIFIED_TABLE = "qualified_table"
    EXPLICIT_NONSIGNIFICANCE = "explicit_nonsignificance"
    POORLY_ANNOTATED_COORDINATES = "poorly_annotated_coordinates"
    UNKNOWN_COVERAGE = "unknown_coverage"


#: What each channel is allowed to contribute. Read by :meth:`EvidenceLedger.contributions`, so
#: a caller cannot quietly grant a channel more than the table above allows.
ALLOWED_CONTRIBUTIONS = {
    EvidenceChannel.IMAGE: frozenset({"effect_likelihood"}),
    EvidenceChannel.QUALIFIED_TABLE: frozenset({"selection_likelihood"}),
    EvidenceChannel.EXPLICIT_NONSIGNIFICANCE: frozenset({"censoring_interval"}),
    EvidenceChannel.POORLY_ANNOTATED_COORDINATES: frozenset({"spatial_basis"}),
    EvidenceChannel.UNKNOWN_COVERAGE: frozenset(),
}


def standardised_scale(design, sample_size=None, group_sizes=None):
    r"""Multiplier :math:`\kappa` linking a standardised effect to a t statistic.

    Parameters
    ----------
    design : {"one-sample", "paired", "two-sample"}
        Anything else raises. A covariate-adjusted GLM, a mixed-effects group map, a correlation
        or an F map is not convertible from counts, and guessing would place an unstated design
        assumption under every downstream number.
    sample_size : :obj:`int`, optional
        Observations, for the one-sample and paired conventions. A paired design standardises by
        the SD of the *differences*, which need not equal a standardised change computed against
        a raw-score SD -- the caller is asserting the former.
    group_sizes : pair of :obj:`int`, optional
        Per-group counts for the two-sample convention. A total is not enough: the same total
        splits into different :math:`\kappa`.
    """
    if design not in SUPPORTED_DESIGNS:
        raise ValueError(
            f"design must be one of {SUPPORTED_DESIGNS}; got {design!r}. A total sample size "
            "does not identify the standardisation for a covariate-adjusted GLM, a "
            "mixed-effects group map, a correlation or an F map, so no conversion is offered "
            "for them rather than a guessed one."
        )
    if design in ("one-sample", "paired"):
        if sample_size is None:
            raise ValueError(f"A {design!r} design needs sample_size.")
        count = np.asarray(sample_size, dtype=float)
        if np.any(count < 2):
            raise ValueError("A one-sample or paired design needs at least two observations.")
        return 1.0 / np.sqrt(count)
    if group_sizes is None:
        raise ValueError(
            "A two-sample design needs group_sizes: the same total splits into different "
            "standardisations, so a total cannot stand in for the split."
        )
    first, second = (np.asarray(size, dtype=float) for size in group_sizes)
    if np.any(first < 2) or np.any(second < 2):
        raise ValueError("Each group needs at least two observations.")
    return np.sqrt(1.0 / first + 1.0 / second)


def degrees_of_freedom(design, sample_size=None, group_sizes=None):
    """Degrees of freedom implied by the design, derived rather than assumed from a count."""
    if design in ("one-sample", "paired"):
        if sample_size is None:
            raise ValueError(f"A {design!r} design needs sample_size.")
        return np.asarray(sample_size, dtype=float) - 1.0
    if design == "two-sample":
        if group_sizes is None:
            raise ValueError("A two-sample design needs group_sizes.")
        first, second = (np.asarray(size, dtype=float) for size in group_sizes)
        return first + second - 2.0
    raise ValueError(f"design must be one of {SUPPORTED_DESIGNS}; got {design!r}.")


def hedges_factor(dof):
    r"""Hedges' small-sample correction :math:`J(\nu)`, for an **observed** estimate.

    .. math:: J(\nu) = \frac{\Gamma(\nu/2)}{\sqrt{\nu/2}\ \Gamma((\nu-1)/2)}

    This corrects an estimator toward the population standardised effect. It is **not** an extra
    multiplier on a planned study's non-centrality: by the time an effect reaches
    :mod:`nimare.meta.cbma.planning` it is already a population quantity, and applying the
    factor again would shrink a design target for a reason that has nothing to do with the
    design.
    """
    dof = np.asarray(dof, dtype=float)
    if np.any(dof <= 1):
        raise ValueError("Hedges' correction needs more than one degree of freedom.")
    return np.exp(gammaln(dof / 2.0) - 0.5 * np.log(dof / 2.0) - gammaln((dof - 1.0) / 2.0))


@dataclass(frozen=True)
class EvidenceRecord:
    """One record, its channel, and the identity that decides whether it is independent.

    Parameters
    ----------
    study_id : :obj:`str`
        The publication or analysis the record came from.
    channel : :class:`EvidenceChannel`
        What kind of evidence it is.
    cohort_id : :obj:`str`, optional
        The participant cohort. ``None`` means *unknown*, which is treated conservatively: an
        unknown cohort cannot be assumed distinct from any other. Section 8 of the plan is
        explicit that a missing cohort identifier requires investigation rather than an
        assumption of independence.
    reporting_event : :obj:`str`, optional
        For a qualified table, the reporting event or the named scenario it is conditioned on.
        Required, because "qualified" is the entire difference between this channel and
        poorly annotated coordinates.
    """

    study_id: str
    channel: EvidenceChannel
    cohort_id: str | None = None
    reporting_event: str | None = None
    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        """Refuse a table declared qualified with no reporting event to qualify it."""
        if self.channel is EvidenceChannel.QUALIFIED_TABLE and not self.reporting_event:
            raise ValueError(
                f"Study {self.study_id!r} is declared a QUALIFIED_TABLE with no "
                "reporting_event. A table with no stated reporting event is "
                "POORLY_ANNOTATED_COORDINATES, which contributes a spatial basis and not a "
                "selection likelihood."
            )


class EvidenceLedger:
    """Records, their channels, and the independence bookkeeping the plan requires.

    Refuses two things that look like more evidence than they are: an image counted alongside its
    own study's table as independent effect observations, and a contribution a channel is not
    entitled to.
    """

    def __init__(self, records=()):
        self._records = []
        for record in records:
            self.add(record)

    def add(self, record):
        """Add a record, refusing a second effect observation from the same study."""
        if not isinstance(record, EvidenceRecord):
            raise TypeError(f"expected an EvidenceRecord; got {type(record).__name__}.")
        if record.channel is EvidenceChannel.IMAGE:
            existing = [
                other
                for other in self._records
                if other.channel is EvidenceChannel.IMAGE and other.study_id == record.study_id
            ]
            if existing:
                raise ValueError(
                    f"Study {record.study_id!r} already contributes an image. One study is one "
                    "effect observation; a second image from the same study is the same "
                    "subjects measured again, not a second draw from the between-study "
                    "distribution."
                )
        self._records.append(record)
        return self

    def contributions(self):
        """Map each record to the contributions its channel allows, and nothing else."""
        return [(record, ALLOWED_CONTRIBUTIONS[record.channel]) for record in self._records]

    def may_contribute(self, record, contribution):
        """Whether ``record`` is entitled to make ``contribution``."""
        return contribution in ALLOWED_CONTRIBUTIONS[record.channel]

    def require(self, record, contribution):
        """Raise unless ``record`` is entitled to ``contribution``."""
        if not self.may_contribute(record, contribution):
            allowed = sorted(ALLOWED_CONTRIBUTIONS[record.channel]) or ["nothing"]
            raise ValueError(
                f"A {record.channel.value!r} record may contribute {allowed}, not "
                f"{contribution!r}. Study {record.study_id!r} would otherwise license an "
                "inference its provenance does not support."
            )

    def paired_studies(self):
        """Studies contributing both an image and a table, which are one study and not two."""
        images = {r.study_id for r in self._records if r.channel is EvidenceChannel.IMAGE}
        tables = {
            r.study_id
            for r in self._records
            if r.channel
            in (EvidenceChannel.QUALIFIED_TABLE, EvidenceChannel.POORLY_ANNOTATED_COORDINATES)
        }
        return sorted(images & tables)

    def effective_independent_units(self):
        r"""Count of independent units, treating an unknown cohort as *not* known to be distinct.

        Returns a dict with ``by_study``, ``by_cohort`` and ``unknown_cohort``. The plan asks for
        "effective independent cohort information, not merely peak/voxel counts", and the
        conservative reading of a missing identifier is the one taken here: records with no
        cohort are pooled into a single unit rather than counted separately, so the number cannot
        be inflated by absent metadata.
        """
        studies = {r.study_id for r in self._records}
        known = {r.cohort_id for r in self._records if r.cohort_id is not None}
        unknown = [r for r in self._records if r.cohort_id is None]
        return {
            "by_study": len(studies),
            "by_cohort": len(known) + (1 if unknown else 0),
            "unknown_cohort": len(unknown),
            "paired_studies": self.paired_studies(),
        }

    def __len__(self):
        """Count the records held."""
        return len(self._records)

    def __iter__(self):
        """Iterate the records in the order they were added."""
        return iter(self._records)
