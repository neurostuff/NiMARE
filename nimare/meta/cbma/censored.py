r"""Scalar interval-censored random-effects reference for the marginal effect.

This is the reference model the combined estimator is built on: a *scalar* Gaussian
random-effects meta-analysis in which an observation is an **event about a study's estimate**
rather than a value. A study's latent effect is :math:`\theta_i \sim N(m, \tau^2)` and its
estimate is :math:`Y_i \mid \theta_i \sim N(\theta_i, s_i^2)`, so marginally
:math:`Y_i \sim N(m, s_i^2 + \tau^2)` and an observation :math:`L_i < Y_i < U_i` contributes

.. math::
    \ell_i(m, \tau^2) = \log\left[
    \Phi\!\left(\frac{U_i - m}{\sqrt{s_i^2+\tau^2}}\right)
    - \Phi\!\left(\frac{L_i - m}{\sqrt{s_i^2+\tau^2}}\right)\right].

Both parameters are estimated jointly. Freezing :math:`\tau^2` at a between-study variance read
off a small image subset is not the same quantity whenever the effect distribution has a
component at zero: there
:math:`\operatorname{Var}(\theta) = \pi\tau_a^2 + \pi(1-\pi)\mu^2`, so dividing a total variance
by the prevalence overstates the active component's variance by exactly :math:`(1-\pi)\mu^2`.

Everything computed here is derived first in ``proofs/scalar_censored_reference.py`` of the
companion experiments repository: the marginalisation, the exact scores, the one-sided limits,
and the degenerate-interval limit that makes :meth:`ObservationState.EXACT` reduce to ordinary
random-effects meta-analysis. The one expression not in that file is the exact observation's own
score, which is the derivative of a Gaussian log-density; the tests check it against finite
differences alongside the censored ones.

**What this does not do.** It takes the bounds as given. Where the bounds come from -- how a
published table becomes an interval -- is the substantive problem and is not addressed here.
Bounds manufactured from a nearest-peak distance are not the bounds this likelihood assumes, and
no property proved of this estimator transfers to them. It is also scalar: one estimand, no
spatial dependence, no borrowing across voxels.

Notes
-----
**Unknown is not null.** :class:`ObservationState` separates the cases a coordinate table
actually produces, and two of them -- :attr:`ObservationState.UNKNOWN_COMPLETENESS` and
:attr:`ObservationState.OUTSIDE_MASK` -- contribute *nothing* to the likelihood rather than
being read as evidence of a small effect. A table whose completeness is unknown cannot support
the inference that an unlisted voxel was below threshold.

**A complete table is not a complete map.** ``ABSENT_COMPLETE_TABLE`` used to become a
voxelwise interval automatically. It no longer does: a complete list of local maxima above a
threshold is not a list of every suprathreshold voxel, so an unlisted location need not have
been below the cut. The voxelwise reading now has to be asserted explicitly, and the block
likelihood in :mod:`nimare.meta.cbma.blocks` is the right home for the event as it actually is.

**The reporting partition.** Report, silence, and anything in between are disjoint events whose
probabilities sum to one. Using :math:`1 - p_{\text{report}}` for the silence probability
overstates it by exactly the probability of whatever third event was dropped, and the
probabilities that would be correct among retained events are the conditional ones,
:math:`p_r/(p_r+p_s)`. This module therefore never forms a silence probability as a complement:
a silence is an interval like any other, and an event that is neither is simply not observed.
"""

from __future__ import annotations

import enum

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr
from scipy.stats import chi2, norm

#: Largest heterogeneity the optimiser will consider, as a multiple of the observed spread of
#: the finite bounds. A cap is needed because a likelihood with only one-sided observations can
#: be flat in ``tau2``, and an unbounded search then reports whichever value it stopped at.
_TAU2_CAP_MULTIPLE = 25.0

#: Starting values for ``tau2``, as fractions of the sampling variance scale. The mixture of an
#: interval likelihood and a boundary at zero makes a single start unsafe.
_TAU2_STARTS = (0.0, 0.25, 1.0, 4.0)

#: Retention is bounded away from zero: a retention of exactly zero prints nothing, so no
#: table could have been produced under it and the likelihood is undefined there.
_RETENTION_FLOOR = 1e-6

#: Starting values for an estimated retention. The likelihood is often nearly flat in it, so the
#: start that wins is informative about how flat.
_RETENTION_STARTS = (0.25, 0.5, 0.9)

#: Below this the interval mass is treated as numerically unusable and the observation is
#: refused rather than contributing a log of something indistinguishable from zero.
_MASS_FLOOR = 1e-300


class ObservationState(enum.Enum):
    """What a single record actually tells us about one study's effect at one location.

    The distinction between these is the point. Collapsing them loses the difference between
    "this study measured a small effect here" and "this study's table does not say".

    Attributes
    ----------
    IMAGE
        An unthresholded image was available, so the estimate is observed. Contributes the
        ordinary random-effects Gaussian term.
    EXACT
        A peak height reported to full precision, taken as the estimate itself. Identical to
        ``IMAGE`` in the likelihood; kept separate because its provenance differs and because a
        selected peak height is not an unbiased estimate of the effect at that location.

        **Rarely the right state for a coordinate table.** A printed height belongs to the
        cluster's maximum, not to the location being modelled; use ``CLUSTER_PEAK`` unless the
        location *is* the peak.
    CLUSTER_PEAK
        A peak height printed for a cluster that is taken to contain this location. The height
        is the cluster's *maximum*, so it bounds the effect here from above, and membership of a
        suprathreshold cluster bounds it from below by the study's threshold: the interval is
        :math:`(c, h]`. Reading such a report as ``EXACT`` instead is a misspecification with a
        known sign -- the exact record's maximiser is :math:`h` where the interval's is
        :math:`(c+h)/2`, so it pushes the estimate up by :math:`(h-c)/2` -- and measured on HCP
        pseudo-studies it is the whole of the positive bias at a wide coverage radius.
    ROUNDED
        A peak height reported to finite precision, so the estimate lies in a known interval.
    BOUNDED
        A known two-sided interval from some other source, such as a reported range.
    DIRECTION_ONLY
        A peak was reported with its sign but no usable magnitude, so the estimate is known only
        to have cleared the study's threshold in that direction: one-sided.
    NONSIGNIFICANT
        The study states explicitly that the effect here was not significant, which is a genuine
        two-sided interval inside its own threshold.
    NO_PEAK_NEARBY
        No peak is listed within the assumed reach of this location. The same interval as
        ``NONSIGNIFICANT`` and a **different premise**: the study said nothing about this
        location, and a cluster it did print may well contain the location with its peak
        further away than the assumed reach. That ambiguity is a retention event, so this state
        is retention-eligible where ``NONSIGNIFICANT`` is not, and the retention probability
        delivers exactly the two-branch mixture :math:`\kappa S + F` with
        :math:`\kappa = 1-\rho` (``proofs/what_a_silence_says.py``). Left at
        :math:`\rho = 1` it is the certain-silence reading, whose score is more negative than
        the correct one at every mean by :math:`\kappa f/[F(\kappa S + F)]`, which is why a
        corpus dominated by silences biases downward.
    ABSENT_COMPLETE_TABLE
        No peak is listed and the table is known to be complete, so the estimate did not clear
        the threshold. The same interval as ``NONSIGNIFICANT``, from a weaker premise.
    UNKNOWN_COMPLETENESS
        No peak is listed and the table's completeness is unknown. **Contributes nothing.**
    OUTSIDE_MASK
        The location was not analysed. **Contributes nothing.**
    """

    IMAGE = "image"
    EXACT = "exact"
    CLUSTER_PEAK = "cluster_peak"
    ROUNDED = "rounded"
    BOUNDED = "bounded"
    DIRECTION_ONLY = "direction_only"
    NONSIGNIFICANT = "nonsignificant"
    NO_PEAK_NEARBY = "no_peak_nearby"
    ABSENT_COMPLETE_TABLE = "absent_complete_table"
    UNKNOWN_COMPLETENESS = "unknown_completeness"
    OUTSIDE_MASK = "outside_mask"


#: States that carry no information about the effect. Listed once so that nothing has to
#: rediscover which they are.
UNINFORMATIVE_STATES = frozenset(
    {ObservationState.UNKNOWN_COMPLETENESS, ObservationState.OUTSIDE_MASK}
)

#: States whose presence in a table is itself subject to retention: the effect cleared the
#: threshold, and whether it was then printed is a separate event.
RETAINED_REPORT_STATES = frozenset(
    {
        ObservationState.EXACT,
        ObservationState.CLUSTER_PEAK,
        ObservationState.ROUNDED,
        ObservationState.BOUNDED,
        ObservationState.DIRECTION_ONLY,
    }
)

#: The states retention actually bites on. Both are absences whose cause is ambiguous between
#: "did not clear the threshold" and "cleared it and no row appeared here". For
#: ``ABSENT_COMPLETE_TABLE`` the second branch is suppression; for ``NO_PEAK_NEARBY`` it is a
#: printed cluster whose peak fell outside the assumed reach. The likelihood cannot tell them
#: apart -- they are the same term -- so one parameter covers both, and which mechanism it is
#: taken to describe is the caller's statement rather than the model's.
#:
#: ``NONSIGNIFICANT`` is deliberately absent: there the study *said* the effect was small, so
#: its presence is not a retention event. That is the whole distinction.
RETAINED_ABSENCE_STATES = frozenset(
    {ObservationState.ABSENT_COMPLETE_TABLE, ObservationState.NO_PEAK_NEARBY}
)


def retention_roles(states):
    """How retention enters each record: ``0`` not at all, ``1`` as a report, ``-1`` as absence.

    An :attr:`ObservationState.IMAGE` is unaffected, because an available image is available
    whatever the table printed. An :attr:`ObservationState.NONSIGNIFICANT` record is also
    unaffected: the study *said* the effect was not significant, so its presence is not a
    retention event -- that is the whole difference between it and
    :attr:`ObservationState.ABSENT_COMPLETE_TABLE`, which is the ambiguous case.
    """
    roles = np.zeros(len(states), dtype=int)
    for index, state in enumerate(_as_states(states)):
        if state in RETAINED_REPORT_STATES:
            roles[index] = 1
        elif state in RETAINED_ABSENCE_STATES:
            roles[index] = -1
    return roles


def _as_states(states):
    """Accept enum members or their values, and reject anything else by name."""
    out = []
    for item in states:
        if isinstance(item, ObservationState):
            out.append(item)
            continue
        try:
            out.append(ObservationState(item))
        except ValueError as error:
            raise ValueError(
                f"{item!r} is not an observation state; expected one of "
                f"{[state.value for state in ObservationState]}."
            ) from error
    return out


def bounds_from_states(
    states,
    values=None,
    thresholds=None,
    precisions=None,
    signs=None,
    absence_is_voxelwise=False,
):
    r"""Turn observation states into the interval each one implies.

    Parameters
    ----------
    states : sequence of :class:`ObservationState` or :obj:`str`
        One state per record.
    values : :obj:`numpy.ndarray`, optional
        Reported or observed estimate, on the effect-size scale. Required for ``IMAGE``,
        ``EXACT`` and ``ROUNDED``.
    thresholds : :obj:`numpy.ndarray`, optional
        The study's reporting threshold on the effect-size scale, as a positive magnitude.
        Required for ``DIRECTION_ONLY``, ``NONSIGNIFICANT`` and ``ABSENT_COMPLETE_TABLE``. It is
        supplied, never inferred from the smallest reported value: that value is an order
        statistic of the reported heights and so moves with how much signal the study had.
    precisions : :obj:`numpy.ndarray`, optional
        Full width of the rounding interval for ``ROUNDED`` records, in effect-size units.
    signs : :obj:`numpy.ndarray`, optional
        ``+1`` or ``-1`` for ``DIRECTION_ONLY`` records. A table giving unsigned magnitudes
        cannot use that state: an unsigned report constrains both tails and is a union of two
        intervals, which this likelihood does not represent.

        Also read, optionally, for ``NONSIGNIFICANT`` and ``ABSENT_COMPLETE_TABLE`` records,
        where it carries the *reporting protocol's* sidedness rather than a direction of effect.
        Under a one-sided protocol an absence means :math:`Y < c`, so a supplied ``+1`` leaves
        the lower bound at ``-inf``; left unsupplied, an absence is the two-sided
        :math:`|Y| < c`.

    Returns
    -------
    lower, upper : :obj:`numpy.ndarray`
        Interval bounds, with ``-inf``/``inf`` for one-sided records and both infinite for the
        uninformative states. ``lower == upper`` marks an exactly observed value.

    absence_is_voxelwise : :obj:`bool`, default=False
        Required to be ``True`` before an :attr:`ObservationState.ABSENT_COMPLETE_TABLE` record
        is turned into a voxelwise interval, because **that reading is usually wrong**.

        A complete table lists every *local maximum* above the threshold. It does not list every
        suprathreshold *voxel*: a voxel can exceed the cut and simply not be a local maximum, so
        "no peak was listed here" does not imply "the effect here was below the cut". Treating
        it as though it did fabricates a below-threshold observation, which is precisely the
        inference this module refuses for unknown completeness -- and it was doing it
        automatically for known completeness. Flagged by an external audit.

        Two honest routes exist. Use :attr:`ObservationState.NONSIGNIFICANT` where a
        *prespecified scalar* test is genuinely known to have been non-significant, which is a
        valid censoring interval. Or use :mod:`nimare.meta.cbma.blocks`, whose likelihood models
        "no local maximum in this region cleared the threshold" as the event it actually is.
        Passing ``absence_is_voxelwise=True`` asserts that the voxelwise reading holds for this
        corpus; it is never inferred.

    Raises
    ------
    ValueError
        If a state's required input is missing or not finite. A silently defaulted threshold
        would be an assumption entering through a gap rather than through the interface. Also if
        an ``ABSENT_COMPLETE_TABLE`` record appears without ``absence_is_voxelwise=True``.
    """
    states_checked = _as_states(states)
    if not absence_is_voxelwise and any(
        state is ObservationState.ABSENT_COMPLETE_TABLE for state in states_checked
    ):
        raise ValueError(
            "ABSENT_COMPLETE_TABLE records were supplied without absence_is_voxelwise=True. A "
            "complete table lists every local maximum above the threshold, not every "
            "suprathreshold voxel, so an unlisted voxel need not have been below the cut. Use "
            "NONSIGNIFICANT where a prespecified scalar test is known to be non-significant, or "
            "the block likelihood in nimare.meta.cbma.blocks, which models the reporting event "
            "as it is. Pass absence_is_voxelwise=True only to assert the voxelwise reading."
        )
    states = _as_states(states)
    count = len(states)
    lower = np.full(count, -np.inf)
    upper = np.full(count, np.inf)

    def required(array, name, index, state):
        if array is None:
            raise ValueError(f"{state.value!r} records need {name!r}, which was not supplied.")
        value = np.asarray(array, dtype=float).reshape(-1)[index]
        if not np.isfinite(value):
            raise ValueError(f"{state.value!r} record {index} has a non-finite {name!r}.")
        return value

    for index, state in enumerate(states):
        if state in UNINFORMATIVE_STATES:
            continue
        if state in (ObservationState.IMAGE, ObservationState.EXACT):
            point = required(values, "values", index, state)
            lower[index] = upper[index] = point
        elif state is ObservationState.CLUSTER_PEAK:
            height = required(values, "values", index, state)
            cut = abs(required(thresholds, "thresholds", index, state))
            direction = required(signs, "signs", index, state)
            if direction == 0:
                raise ValueError(
                    f"{state.value!r} record {index} has sign zero. A cluster peak bounds one "
                    "tail; without a direction the record is a union of two intervals, which "
                    "this likelihood cannot represent."
                )
            if abs(height) < cut:
                raise ValueError(
                    f"{state.value!r} record {index} reports a height of {height:g} inside its "
                    f"own threshold of {cut:g}. A printed peak below the cut it was selected by "
                    "is a table and a threshold that contradict each other, not something to "
                    "clip -- most often the height and the threshold are on different scales."
                )
            if direction > 0:
                lower[index], upper[index] = cut, abs(height)
            else:
                lower[index], upper[index] = -abs(height), -cut
        elif state is ObservationState.ROUNDED:
            point = required(values, "values", index, state)
            width = required(precisions, "precisions", index, state)
            if width <= 0:
                raise ValueError(f"{state.value!r} record {index} has a non-positive precision.")
            lower[index], upper[index] = point - width / 2.0, point + width / 2.0
        elif state is ObservationState.BOUNDED:
            lower[index] = required(values, "values", index, state)
            upper[index] = lower[index] + required(precisions, "precisions", index, state)
        elif state is ObservationState.DIRECTION_ONLY:
            cut = abs(required(thresholds, "thresholds", index, state))
            direction = required(signs, "signs", index, state)
            if direction > 0:
                lower[index] = cut
            elif direction < 0:
                upper[index] = -cut
            else:
                raise ValueError(
                    f"{state.value!r} record {index} has sign zero. An unsigned report "
                    "constrains both tails and is a union of two intervals, which this "
                    "likelihood cannot represent; use a signed contrast or drop the record."
                )
        else:  # NONSIGNIFICANT, NO_PEAK_NEARBY, ABSENT_COMPLETE_TABLE
            cut = abs(required(thresholds, "thresholds", index, state))
            # Absence is one-sided under a one-sided reporting protocol. A study that would have
            # reported only positive peaks and reported none tells us Y < c, not |Y| < c, and
            # reading it as the two-sided interval would invent a lower bound the protocol never
            # supports. The sidedness comes from ``signs`` because it is a property of the
            # protocol, not of the record.
            direction = 0.0
            if signs is not None:
                supplied = np.asarray(signs, dtype=float).reshape(-1)[index]
                direction = 0.0 if not np.isfinite(supplied) else float(supplied)
            if direction > 0:
                upper[index] = cut
            elif direction < 0:
                lower[index] = -cut
            else:
                lower[index], upper[index] = -cut, cut

    return lower, upper


def _interval_log_mass(lower_z, upper_z):
    """Log probability of a standardised interval, computed to avoid cancellation.

    ``ndtr(b) - ndtr(a)`` loses all its digits when both bounds sit in the same tail, so the
    upper tail is taken from the mirrored CDF there. Returns ``-inf`` where the mass underflows,
    which the caller turns into a refusal rather than a very large log-likelihood.
    """
    lower_z = np.asarray(lower_z, dtype=float)
    upper_z = np.asarray(upper_z, dtype=float)
    both_upper = lower_z > 0
    mass = np.where(
        both_upper,
        ndtr(-lower_z) - ndtr(-upper_z),
        ndtr(upper_z) - ndtr(lower_z),
    )
    one_sided_below = ~np.isfinite(lower_z) & np.isfinite(upper_z)
    one_sided_above = np.isfinite(lower_z) & ~np.isfinite(upper_z)
    with np.errstate(invalid="ignore"):
        out = np.where(mass > _MASS_FLOOR, np.log(np.clip(mass, _MASS_FLOOR, None)), -np.inf)
        out = np.where(one_sided_below, log_ndtr(upper_z), out)
        out = np.where(one_sided_above, log_ndtr(-lower_z), out)
    return out


def _retention_vector(retention, count):
    """Retention broadcast to the **full** record shape, before any subsetting.

    Broadcasting to a subset's length was a defect: with a per-record retention array, the
    values were matched to the censored records' positions rather than to their own, and a mixed
    exact/censored table raised a broadcasting error outright. Broadcast to every record, then
    index with the same mask the observations were indexed with.
    """
    values = np.broadcast_to(np.asarray(retention, dtype=float), (count,)).astype(float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0) or np.any(values > 1):
        raise ValueError("retention must be finite and in (0, 1]; a zero would print nothing.")
    return values


def _apply_retention(log_mass, rho, roles):
    """Turn interval log-probabilities into retention-aware ones.

    ``rho`` and ``roles`` must already be restricted to the same records as ``log_mass``.

    An absence is ``(1 - rho) + rho * mass`` rather than ``1 - rho * (1 - mass)``: algebraically
    the same, but the first form adds two non-negative numbers and the second subtracts nearly
    equal ones.
    """
    mass = np.exp(log_mass)
    out = np.where(roles == 1, log_mass + np.log(rho), log_mass)
    with np.errstate(divide="ignore"):
        absence = np.log(np.clip((1.0 - rho) + rho * mass, _MASS_FLOOR, None))
    return np.where(roles == -1, absence, out)


def censored_loglik(
    mean, between_variance, lower, upper, variances, *, retention=None, roles=None
):
    r"""Total log-likelihood of interval and exact observations.

    Exact records (``lower == upper``) contribute the Gaussian log-density, which the degenerate
    limit of the interval term equals up to an additive constant free of the parameters, so the
    two kinds of record can be summed. Records with both bounds infinite contribute nothing.

    Retention, if supplied, is the probability that an effect which cleared its threshold was
    actually printed. A record whose ``roles`` entry is ``1`` then contributes
    :math:`\rho \times \text{mass}`, and one whose entry is ``-1`` contributes
    :math:`(1-\rho) + \rho\,\text{mass}`, which is the retention-aware probability of *not*
    seeing a row. Leaving ``retention`` unset is the :math:`\rho = 1` special case, and that is
    not a neutral simplification: it asserts every exceedance is printed, and its bias is
    computable in closed form (see ``proofs/retention_in_the_reporting_model.py``).

    Returns ``-inf`` if any usable record has numerically zero probability, so an optimiser walks
    away from such a point rather than through it.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    total_variance = np.asarray(variances, dtype=float) + float(between_variance)
    if np.any(total_variance <= 0):
        return -np.inf
    scale = np.sqrt(total_variance)

    informative = np.isfinite(lower) | np.isfinite(upper)
    exact = np.isfinite(lower) & np.isfinite(upper) & (lower == upper)

    aligned_rho = aligned_roles = None
    if retention is not None:
        if roles is None:
            raise ValueError("Supplying retention requires roles; use retention_roles(states).")
        aligned_roles = np.asarray(roles, dtype=int).reshape(-1)
        if aligned_roles.size != lower.size:
            raise ValueError(
                f"roles must have one entry per record; got {aligned_roles.size} for "
                f"{lower.size} records."
            )
        aligned_rho = _retention_vector(retention, lower.size)

    total = 0.0
    if exact.any():
        terms = norm.logpdf(lower[exact], loc=mean, scale=scale[exact])
        if retention is not None:
            # An exactly reported peak height is still a *report*: its presence in the table is
            # a retention event, so it carries a factor of rho exactly as a censored report
            # does. Omitting it was a defect worth log(rho) per such record -- log(2) at
            # rho = 0.5 -- and it left the exact and censored limbs describing different models.
            # An IMAGE record has role 0 and is untouched: an available image is available
            # whatever the table printed.
            terms = terms + np.where(aligned_roles[exact] == 1, np.log(aligned_rho[exact]), 0.0)
        if not np.all(np.isfinite(terms)):
            return -np.inf
        total += float(terms.sum())

    censored = informative & ~exact
    if censored.any():
        with np.errstate(invalid="ignore"):
            lower_z = (lower[censored] - mean) / scale[censored]
            upper_z = (upper[censored] - mean) / scale[censored]
        terms = _interval_log_mass(lower_z, upper_z)
        if retention is not None:
            terms = _apply_retention(terms, aligned_rho[censored], aligned_roles[censored])
        if not np.all(np.isfinite(terms)):
            return -np.inf
        total += float(terms.sum())
    return total


def retention_score(mean, between_variance, lower, upper, variances, retention, roles):
    r"""Differentiate the log-likelihood with respect to the retention probability.

    A reported record contributes :math:`1/\rho`; an absent one contributes
    :math:`(\Delta - 1)/[(1-\rho) + \rho\Delta]` for its interval mass :math:`\Delta`. Both
    are verified symbolically in ``proofs/retention_in_the_reporting_model.py``, along with the
    fact that the second reduces to :math:`-S/(1-\rho S)` for a one-sided absence, so the
    general form is the same model as the directional one rather than a second one.

    Records retention does not apply to -- images, explicit nonsignificance, and the
    uninformative states -- contribute nothing.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    scale = np.sqrt(np.asarray(variances, dtype=float) + float(between_variance))
    roles = np.asarray(roles, dtype=int)

    if roles.size != lower.size:
        raise ValueError(
            f"roles must have one entry per record; got {roles.size} for {lower.size} records."
        )
    aligned_rho = _retention_vector(retention, lower.size)

    informative = _informative_mask(lower, upper)
    exact = np.isfinite(lower) & np.isfinite(upper) & (lower == upper)
    censored = informative & ~exact

    total = 0.0

    # Exactly reported records carry log(rho) in the likelihood, hence 1/rho in this score.
    # Omitting them left an analytic gradient that agreed with a *different* likelihood from the
    # one being maximised, which finite-difference checks against that same wrong likelihood
    # could not detect.
    exact_reports = exact & (roles == 1)
    if exact_reports.any():
        total += float(np.sum(1.0 / aligned_rho[exact_reports]))

    if censored.any():
        with np.errstate(invalid="ignore"):
            lower_z = (lower[censored] - mean) / scale[censored]
            upper_z = (upper[censored] - mean) / scale[censored]
        mass = np.exp(_interval_log_mass(lower_z, upper_z))
        selected = roles[censored]
        rho = aligned_rho[censored]

        report = selected == 1
        if report.any():
            total += float(np.sum(1.0 / rho[report]))
        absence = selected == -1
        if absence.any():
            denominator = np.clip(
                (1.0 - rho[absence]) + rho[absence] * mass[absence], _MASS_FLOOR, None
            )
            total += float(np.sum((mass[absence] - 1.0) / denominator))
    return total


def censored_score(mean, between_variance, lower, upper, variances, *, retention=None, roles=None):
    r"""Exact gradient of :func:`censored_loglik` in :math:`(m, \tau^2)`.

    For a censored record with standardised bounds :math:`a, b` and interval mass
    :math:`\Delta = \Phi(b) - \Phi(a)`,

    .. math::
        \partial_m \ell = -\frac{\varphi(b)-\varphi(a)}{\sigma\Delta},
        \qquad
        \partial_{\tau^2} \ell = -\frac{b\varphi(b)-a\varphi(a)}{2\sigma^2\Delta},

    both verified symbolically in ``proofs/scalar_censored_reference.py``. An exact record
    contributes the Gaussian score.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    total_variance = np.asarray(variances, dtype=float) + float(between_variance)
    scale = np.sqrt(total_variance)

    informative = np.isfinite(lower) | np.isfinite(upper)
    exact = np.isfinite(lower) & np.isfinite(upper) & (lower == upper)
    d_mean = 0.0
    d_between = 0.0

    if exact.any():
        residual = lower[exact] - mean
        variance = total_variance[exact]
        d_mean += float((residual / variance).sum())
        d_between += float((residual**2 / variance**2 - 1.0 / variance).sum() / 2.0)

    censored = informative & ~exact
    if censored.any():
        with np.errstate(invalid="ignore"):
            lower_z = (lower[censored] - mean) / scale[censored]
            upper_z = (upper[censored] - mean) / scale[censored]
        mass = np.exp(_interval_log_mass(lower_z, upper_z))
        mass = np.clip(mass, _MASS_FLOOR, None)
        # phi vanishes at an infinite bound, and so does z*phi; nan_to_num turns the inf*0 that
        # numpy produces there into the limit.
        with np.errstate(invalid="ignore"):
            density_low = np.nan_to_num(norm.pdf(lower_z), nan=0.0, posinf=0.0, neginf=0.0)
            density_high = np.nan_to_num(norm.pdf(upper_z), nan=0.0, posinf=0.0, neginf=0.0)
            weighted_low = np.nan_to_num(lower_z * density_low, nan=0.0, posinf=0.0, neginf=0.0)
            weighted_high = np.nan_to_num(upper_z * density_high, nan=0.0, posinf=0.0, neginf=0.0)
        mass_d_mean = -(density_high - density_low) / scale[censored]
        mass_d_between = -(weighted_high - weighted_low) / (2.0 * total_variance[censored])
        if retention is None:
            d_mean += float((mass_d_mean / mass).sum())
            d_between += float((mass_d_between / mass).sum())
        else:
            if roles is None:
                raise ValueError(
                    "Supplying retention requires roles; use retention_roles(states)."
                )
            aligned = np.asarray(roles, dtype=int).reshape(-1)
            if aligned.size != lower.size:
                raise ValueError(
                    f"roles must have one entry per record; got {aligned.size} for "
                    f"{lower.size} records."
                )
            selected_roles = aligned[censored]
            rho = _retention_vector(retention, lower.size)[censored]
            # A report carries a factor of rho, which is constant in (m, tau^2) and so drops
            # out of the score. An absence has probability (1-rho) + rho*mass, whose derivative
            # is rho times the mass derivative.
            denominator = np.where(selected_roles == -1, (1.0 - rho) + rho * mass, mass)
            numerator_scale = np.where(selected_roles == -1, rho, 1.0)
            denominator = np.clip(denominator, _MASS_FLOOR, None)
            d_mean += float((numerator_scale * mass_d_mean / denominator).sum())
            d_between += float((numerator_scale * mass_d_between / denominator).sum())
    return np.array([d_mean, d_between])


def _identification_condition(fit, lower, upper, variances, roles):
    """Condition number of the observed information in the three parameters, by differences.

    This is the practical form of the identification condition: a study's score direction is
    fixed by its threshold and its precision alone, so records sharing both contribute parallel
    gradients and the information collapses to rank one however many of them there are. Three
    parameters need three (threshold, precision) pairs whose score directions are not collinear,
    and collinearity is a single scalar equation on the design rather than a curiosity.

    A large condition number is not a failure. It is the statement that the fit is formally
    identified and badly determined, which is worth reporting rather than hiding behind a
    standard error that looks finite.
    """
    point = np.array([fit["mean"], fit["between_variance"], fit["retention"]], dtype=float)
    steps = np.array([1e-4, 1e-5, 1e-4])

    def gradient(parameters):
        rho = float(np.clip(parameters[2], _RETENTION_FLOOR, 1.0))
        partial = censored_score(
            parameters[0], parameters[1], lower, upper, variances, retention=rho, roles=roles
        )
        slope = retention_score(parameters[0], parameters[1], lower, upper, variances, rho, roles)
        return np.array([partial[0], partial[1], slope])

    hessian = np.zeros((3, 3))
    for index in range(3):
        forward, backward = point.copy(), point.copy()
        forward[index] += steps[index]
        backward[index] -= steps[index]
        if index == 1:
            backward[index] = max(backward[index], 0.0)
        if index == 2:
            forward[index] = min(forward[index], 1.0)
            backward[index] = max(backward[index], _RETENTION_FLOOR)
        span = forward[index] - backward[index]
        if span <= 0:
            return float("inf")
        hessian[:, index] = (gradient(forward) - gradient(backward)) / span
    information = -(hessian + hessian.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(information)
    if eigenvalues[0] <= 0:
        return float("inf")
    return float(eigenvalues[-1] / eigenvalues[0])


def _informative_mask(lower, upper):
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return np.isfinite(lower) | np.isfinite(upper)


def _variance_scale(lower, upper, variances):
    """Sampling-variance scale of the informative records only.

    Uninformative rows must not reach this. They set the optimiser's starting values through it,
    and a start that moves with the number of rows that contribute nothing would make
    "contributes nothing" true only to the optimiser's tolerance.
    """
    informative = _informative_mask(lower, upper)
    variances = np.asarray(variances, dtype=float)
    if not informative.any():
        return 0.0
    return float(np.mean(variances[informative]))


def _tau2_cap(lower, upper, variances):
    """Upper bound for the heterogeneity search, from the spread of the informative bounds."""
    informative = _informative_mask(lower, upper)
    finite = np.concatenate(
        [np.asarray(lower, dtype=float)[informative], np.asarray(upper, dtype=float)[informative]]
    )
    finite = finite[np.isfinite(finite)]
    scale = float(np.var(finite)) if finite.size > 1 else 0.0
    if informative.any():
        scale = max(scale, float(np.max(np.asarray(variances, dtype=float)[informative])))
    return _TAU2_CAP_MULTIPLE * max(scale, 1e-6)


def fit_censored(
    lower,
    upper,
    variances,
    *,
    retention=None,
    roles=None,
    estimate_retention=False,
    fixed_between_variance=None,
    max_starts=None,
):
    r"""Maximise the censored likelihood in :math:`(m, \tau^2)` from several starts.

    Parameters
    ----------
    lower, upper : :obj:`numpy.ndarray`
        Interval bounds, as returned by :func:`bounds_from_states`.
    variances : :obj:`numpy.ndarray`
        Sampling variance :math:`s_i^2` of each study's estimate.
    retention : :obj:`float` or :obj:`numpy.ndarray`, optional
        Probability that an effect clearing its threshold was actually printed. Supplied from
        external calibration. Leaving it unset is the :math:`\rho = 1` model.
    roles : :obj:`numpy.ndarray`, optional
        Which records retention applies to, from :func:`retention_roles`. Required whenever
        retention is supplied or estimated.
    estimate_retention : :obj:`bool`, default=False
        Estimate :math:`\rho` jointly with the other two parameters instead of taking it as
        given. Whether this is worth doing is a property of the design, not a preference:

        * a study's score direction is fixed by its threshold and its precision alone, so
          records sharing both contribute parallel gradients and the information collapses to
          rank one however many there are;
        * three parameters need three (threshold, precision) pairs whose directions are not
          collinear, which is why the returned ``condition_number`` matters more than the rank;
        * from report indicators *alone* the retention is barely estimable at realistic corpus
          sizes -- at 100 coordinate studies with no threshold spread its standard error is
          around 1.8 on a parameter confined to :math:`[0,1]`, falling to about 0.5 with
          threshold spread. Images change this, because they pin the magnitude and the
          heterogeneity and leave the report rate to carry only the retention.

        The trade against supplying a value: at 8 images and 100 coordinate studies, estimating
        retention widens the interval on the mean by about a third, while a supplied value wrong
        by 0.2 displaces the estimate by roughly 1.6 standard errors. Fitting wins there. With
        one image and many tables the misspecified fit has the *lower* root-mean-square error
        and no coverage at all, which is the trade the whole construction exists to expose.
    fixed_between_variance : :obj:`float`, optional
        Hold :math:`\tau^2` at this value instead of estimating it. Provided to *measure* the
        cost of freezing heterogeneity, not as a recommended mode.
    max_starts : :obj:`int`, optional
        Cap on the number of starting values. Defaults to all of them.

    Returns
    -------
    :obj:`dict`
        ``mean``, ``between_variance``, ``retention``, ``loglik``, ``n_informative``,
        ``converged``, ``at_zero_boundary`` (the fit sits at :math:`\tau^2 = 0`),
        ``at_variance_cap``, and ``valid``. ``mean`` is ``nan`` when no record is informative:
        there is no estimate then, and a zero would read as one.

        With ``estimate_retention=True`` the result also carries ``at_full_retention`` and
        ``condition_number``, the ratio of the largest to the smallest eigenvalue of the
        observed information in all three parameters. A large value is not a failure; it is the
        statement that the fit is formally identified and badly determined, which is worth
        reporting rather than hiding behind a standard error that looks finite.

    Notes
    -----
    Several starts are used because the interval likelihood is not globally concave in
    :math:`(m, \tau^2)` and a boundary at :math:`\tau^2 = 0` is often active. The returned
    ``converged`` flag reports the optimiser's own verdict at the best start; it is not a claim
    that the maximum is global.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    variances = np.asarray(variances, dtype=float)
    if not (lower.shape == upper.shape == variances.shape):
        raise ValueError(
            f"lower, upper and variances must have the same shape; got {lower.shape}, "
            f"{upper.shape} and {variances.shape}."
        )
    if np.any(variances <= 0):
        raise ValueError("Every sampling variance must be positive.")
    if np.any(lower > upper):
        raise ValueError("Every lower bound must be no greater than its upper bound.")

    informative = _informative_mask(lower, upper)
    failure = {
        "mean": np.nan,
        "between_variance": np.nan,
        "loglik": -np.inf,
        "n_informative": int(informative.sum()),
        "converged": False,
        "at_zero_boundary": False,
        "at_variance_cap": False,
        "valid": False,
    }
    if not informative.any():
        return failure

    with np.errstate(invalid="ignore"):
        centres = np.where(
            np.isfinite(lower) & np.isfinite(upper),
            (lower + upper) / 2.0,
            np.where(np.isfinite(lower), lower, upper),
        )
    start_mean = float(np.mean(centres[informative]))
    cap = _tau2_cap(lower, upper, variances)
    scale = _variance_scale(lower, upper, variances)

    if fixed_between_variance is not None:
        between = float(fixed_between_variance)
        if between < 0:
            raise ValueError("fixed_between_variance must be non-negative.")

        def objective(parameters):
            value = censored_loglik(
                parameters[0],
                between,
                lower,
                upper,
                variances,
                retention=retention,
                roles=roles,
            )
            gradient = censored_score(
                parameters[0],
                between,
                lower,
                upper,
                variances,
                retention=retention,
                roles=roles,
            )
            return -value, -gradient[:1]

        result = minimize(objective, x0=[start_mean], jac=True, method="L-BFGS-B")
        if not np.isfinite(result.fun):
            return failure
        return {
            "mean": float(result.x[0]),
            "between_variance": between,
            "loglik": float(-result.fun),
            "n_informative": int(informative.sum()),
            "converged": bool(result.success),
            "at_zero_boundary": between == 0.0,
            "at_variance_cap": False,
            "valid": True,
        }

    def objective(parameters):
        value = censored_loglik(
            parameters[0],
            parameters[1],
            lower,
            upper,
            variances,
            retention=retention,
            roles=roles,
        )
        if not np.isfinite(value):
            return np.inf, np.zeros(2)
        return -value, -censored_score(
            parameters[0],
            parameters[1],
            lower,
            upper,
            variances,
            retention=retention,
            roles=roles,
        )

    if estimate_retention:
        if roles is None:
            raise ValueError(
                "Estimating retention requires roles; use retention_roles(states). Without "
                "them nothing says which records a retention probability applies to."
            )

        def triple_objective(parameters):
            candidate = float(np.clip(parameters[2], _RETENTION_FLOOR, 1.0))
            value = censored_loglik(
                parameters[0],
                parameters[1],
                lower,
                upper,
                variances,
                retention=candidate,
                roles=roles,
            )
            if not np.isfinite(value):
                return np.inf, np.zeros(3)
            partial = censored_score(
                parameters[0],
                parameters[1],
                lower,
                upper,
                variances,
                retention=candidate,
                roles=roles,
            )
            slope = retention_score(
                parameters[0], parameters[1], lower, upper, variances, candidate, roles
            )
            return -value, -np.array([partial[0], partial[1], slope])

        best = None
        for fraction in _TAU2_STARTS:
            for start_retention in _RETENTION_STARTS:
                result = minimize(
                    triple_objective,
                    x0=[start_mean, min(fraction * scale, cap), start_retention],
                    jac=True,
                    method="L-BFGS-B",
                    bounds=[(None, None), (0.0, cap), (_RETENTION_FLOOR, 1.0)],
                )
                if not np.isfinite(result.fun):
                    continue
                if best is None or result.fun < best.fun:
                    best = result
        if best is None:
            return failure
        found = {
            "mean": float(best.x[0]),
            "between_variance": float(best.x[1]),
            "retention": float(best.x[2]),
            "loglik": float(-best.fun),
            "n_informative": int(informative.sum()),
            "converged": bool(best.success),
            "at_zero_boundary": bool(best.x[1] <= 0.0),
            "at_variance_cap": bool(best.x[1] >= cap * (1 - 1e-9)),
            "at_full_retention": bool(best.x[2] >= 1.0 - 1e-9),
            "valid": True,
        }
        found["condition_number"] = _identification_condition(
            found, lower, upper, variances, roles
        )
        return found

    starts = [(start_mean, min(fraction * scale, cap)) for fraction in _TAU2_STARTS]
    if max_starts is not None:
        starts = starts[: max(int(max_starts), 1)]

    best = None
    for start in starts:
        result = minimize(
            objective,
            x0=list(start),
            jac=True,
            method="L-BFGS-B",
            bounds=[(None, None), (0.0, cap)],
        )
        if not np.isfinite(result.fun):
            continue
        if best is None or result.fun < best.fun:
            best = result
    if best is None:
        return failure

    return {
        "mean": float(best.x[0]),
        "between_variance": float(best.x[1]),
        "retention": None if retention is None else float(np.mean(retention)),
        "loglik": float(-best.fun),
        "n_informative": int(informative.sum()),
        "converged": bool(best.success),
        "at_zero_boundary": bool(best.x[1] <= 0.0),
        "at_variance_cap": bool(best.x[1] >= cap * (1 - 1e-9)),
        "valid": True,
    }


def profile_interval(
    lower,
    upper,
    variances,
    *,
    level=0.95,
    retention=None,
    roles=None,
    estimate_retention=False,
    fixed_between_variance=None,
    search=None,
    grid=257,
):
    r"""Profile-likelihood interval for :math:`m`, with :math:`\tau^2` maximised out.

    Inverts nothing and needs no degrees of freedom, so it survives the weak identification that
    makes a Wald interval unreliable here. It still refers :math:`2\Delta\ell` to
    :math:`\chi^2_1`, which needs the regularity that a boundary at :math:`\tau^2 = 0` can
    break: ``touched_search_limit`` reports when a bound ran into the end of the search range
    rather than crossing the cutoff, and that frequency belongs in any summary that quotes these
    intervals.

    Parameters
    ----------
    retention, roles, estimate_retention
        Passed through to :func:`fit_censored` **and** used inside the profile, so the interval
        describes the same observation and reporting model that was fitted. With
        ``estimate_retention=True`` the retention is profiled out as a nuisance parameter.

    Returns
    -------
    :obj:`dict`
        ``lower``, ``upper``, ``level``, ``touched_search_limit`` and ``valid``. The bounds are
        ``nan``, never zero, where they could not be found.
    """
    fit = fit_censored(
        lower,
        upper,
        variances,
        retention=retention,
        roles=roles,
        estimate_retention=estimate_retention,
        fixed_between_variance=fixed_between_variance,
    )
    unavailable = {
        "lower": np.nan,
        "upper": np.nan,
        "level": float(level),
        "touched_search_limit": True,
        "valid": False,
    }
    if not fit["valid"]:
        return unavailable

    cutoff = float(chi2.ppf(level, 1)) / 2.0
    peak = fit["loglik"]

    variances = np.asarray(variances, dtype=float)
    if search is None:
        spread = np.sqrt(
            _variance_scale(lower, upper, variances) + max(fit["between_variance"], 0.0)
        )
        search = 12.0 * spread + 1.0

    def profile(mean):
        """Log-likelihood at ``mean``, with every other parameter of the *same* model profiled.

        The reporting model has to be the one the fit used. An earlier version took no retention
        argument at all, so a retention-aware fit was profiled against a rho = 1 likelihood and
        the resulting interval belonged to neither model.
        """
        if fixed_between_variance is not None and not estimate_retention:
            return censored_loglik(
                mean,
                fixed_between_variance,
                lower,
                upper,
                variances,
                retention=retention,
                roles=roles,
            )
        cap = _tau2_cap(lower, upper, variances)
        scale = _variance_scale(lower, upper, variances)
        best = -np.inf

        if not estimate_retention:
            for fraction in _TAU2_STARTS:
                result = minimize(
                    lambda parameter: (
                        -censored_loglik(
                            mean,
                            parameter[0],
                            lower,
                            upper,
                            variances,
                            retention=retention,
                            roles=roles,
                        ),
                        -censored_score(
                            mean,
                            parameter[0],
                            lower,
                            upper,
                            variances,
                            retention=retention,
                            roles=roles,
                        )[1:],
                    ),
                    x0=[min(fraction * scale, cap)],
                    jac=True,
                    method="L-BFGS-B",
                    bounds=[(0.0, cap)],
                )
                if np.isfinite(result.fun):
                    best = max(best, float(-result.fun))
            return best

        # Retention is a nuisance parameter here, so it is profiled out alongside tau^2 rather
        # than held at whatever the fit happened to find.
        def negative(parameter):
            candidate = float(np.clip(parameter[1], _RETENTION_FLOOR, 1.0))
            value = censored_loglik(
                mean, parameter[0], lower, upper, variances, retention=candidate, roles=roles
            )
            if not np.isfinite(value):
                return np.inf
            return -value

        for fraction in _TAU2_STARTS:
            for start_retention in _RETENTION_STARTS:
                result = minimize(
                    negative,
                    x0=[min(fraction * scale, cap), start_retention],
                    method="L-BFGS-B",
                    bounds=[(0.0, cap), (_RETENTION_FLOOR, 1.0)],
                )
                if np.isfinite(result.fun):
                    best = max(best, float(-result.fun))
        return best

    offsets = np.linspace(0.0, search, int(grid))
    bounds = []
    touched = False
    for direction in (-1.0, 1.0):
        crossing = np.nan
        previous = (0.0, peak)
        for offset in offsets[1:]:
            candidate = fit["mean"] + direction * offset
            value = profile(candidate)
            if not np.isfinite(value):
                break
            if peak - value >= cutoff:
                # Linear interpolation in the offset between the last two evaluations.
                span = offset - previous[0]
                gap_now, gap_before = peak - value, peak - previous[1]
                weight = (
                    0.0
                    if gap_now == gap_before
                    else (cutoff - gap_before) / (gap_now - gap_before)
                )
                crossing = fit["mean"] + direction * (previous[0] + weight * span)
                break
            previous = (offset, value)
        if not np.isfinite(crossing):
            touched = True
        bounds.append(crossing)

    return {
        "lower": float(bounds[0]),
        "upper": float(bounds[1]),
        "level": float(level),
        "touched_search_limit": touched,
        "valid": bool(np.isfinite(bounds[0]) and np.isfinite(bounds[1])),
    }


def practical_prevalence(mean, between_variance, threshold=0.0):
    r"""Share of studies whose own effect exceeds a stated threshold.

    .. math:: \pi_\delta^+ = \Phi\!\left(\frac{m - \delta}{\tau}\right)

    The design assessment proposes this in place of a structural-zero prevalence, and is right
    that it avoids an implausible exact-zero interpretation. It is also right that "it is not
    automatically identifiable because it has a better name": this is a function of the two
    parameters already being estimated, so it carries no information they do not, and it
    inherits the heterogeneity's uncertainty in full.

    It inherits something worse as well. Its derivative in the heterogeneity is
    :math:`(\delta - m)\varphi/\tau^2`, so **wherever the mean falls short of the threshold the
    prevalence increases with heterogeneity**: at a mean of 0.10 below a threshold of 0.30 it
    runs 0.000, 0.023, 0.159, 0.309, 0.401 as :math:`\tau` goes 0.05, 0.10, 0.20, 0.40, 0.80. A
    number that can be moved from nothing to two fifths by between-study noise alone must not be
    reported without the heterogeneity beside it, which is why :func:`practical_prevalence`
    takes the variance as an argument rather than reading it off a fit.

    Verified in ``proofs/practical_prevalence.py``.
    """
    between_variance = np.asarray(between_variance, dtype=float)
    if np.any(between_variance < 0):
        raise ValueError("The between-study variance cannot be negative.")
    scale = np.sqrt(between_variance)
    mean = np.asarray(mean, dtype=float)
    threshold = np.asarray(threshold, dtype=float)
    degenerate = scale <= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        value = ndtr((mean - threshold) / scale)
    # With no heterogeneity every study is the mean, so the prevalence is an indicator.
    return np.where(degenerate, (mean > threshold).astype(float), value)


def prevalence_times_conditional_mean_gap(mean, between_variance, threshold=0.0):
    r"""How far :math:`\pi_\delta^+\,\mathbb{E}[\theta\mid\theta>\delta]` sits from the mean.

    .. math::
        \pi_\delta^+\,\mathbb{E}[\theta\mid\theta>\delta] - m
        = \tau\,\varphi\!\left(\frac{m-\delta}{\tau}\right) - (1-\pi_\delta^+)\,m

    The assessment warns that "multiplying prevalence by the conditional mean does not recover
    the full marginal mean unless the complementary component has mean zero". This is that gap in
    closed form. It is exposed rather than merely documented so that anyone tempted to form the
    product can see what it costs first.

    Its **sign flips**: the product overstates the marginal mean where the density term dominates
    and understates it where the complementary mass does. It vanishes only as the threshold
    recedes to :math:`-\infty`, where every study counts as above it -- so at any finite
    threshold under a continuous effect distribution the assessment's "unless" is never met.
    """
    between_variance = np.asarray(between_variance, dtype=float)
    scale = np.sqrt(np.clip(between_variance, 0.0, None))
    mean = np.asarray(mean, dtype=float)
    threshold = np.asarray(threshold, dtype=float)
    prevalence = practical_prevalence(mean, between_variance, threshold)
    with np.errstate(divide="ignore", invalid="ignore"):
        density = norm.pdf((mean - threshold) / scale)
    density = np.where(scale > 0, density, 0.0)
    return scale * density - (1.0 - prevalence) * mean
