r"""Drive the censored likelihood over many locations at once, and say what that assumes.

:mod:`nimare.meta.cbma.censored` is scalar: one location, one record per study, one fit. A
meta-analysis wants a map. This module is the step between, deliberately kept separate from any
:class:`~nimare.base.Estimator` plumbing so that what it does is auditable on its own:

* it turns a study's table and threshold into one record per location, under a stated coverage
  reach, with the states those records actually justify -- a peak printed *at* a location is
  :attr:`~censored.ObservationState.EXACT` because there the height is the effect, a peak
  printed nearby is :attr:`~censored.ObservationState.CLUSTER_PEAK` because there the height is
  only an upper bound, and no peak within reach is
  :attr:`~censored.ObservationState.NO_PEAK_NEARBY` rather than a claim the study never made;
* it fits each location independently and returns the estimate, the profile interval and the
  per-location failure flags.

**Locations are fitted independently and that is a modelling choice, not an oversight.** The
map it returns is a field of independent scalar fits, so neighbouring estimates are as
correlated as the data make them and no more; nothing here borrows strength across space. The
measured reason is in ``nimare-experiments``: across 133 NeuroVault collections, group maps
correlate +0.017 on average, so a spatial basis built from other studies carries almost no
information, and regularising towards one would buy smoothness rather than accuracy. That is a
statement about a measured corpus, not a theorem.

**Everything it needs must be supplied.** The reach, the retention and each study's reporting
threshold are arguments with no defaults inferred from the data. The threshold especially: the
smallest reported height is an order statistic that moves with how much signal a study had, so
inferring it from the table makes the reporting model a function of the signal being measured.
Where an archive does not record it -- and the NIDM pain archive does not -- the honest output
is a sensitivity envelope over the assumption rather than a point.

**The reach stands in for an unreported cluster extent.** On HCP pseudo-studies the estimator's
bias changes sign between an 8 mm and a 12 mm reach, because a short reach reads a study that
printed the cluster containing a location as silent there while a long one reads a peak up to
20 mm away as the effect here. There is no safe default; there is a measured optimum on one
corpus, and the parameter belongs in whatever sensitivity analysis accompanies a result.
"""

from __future__ import annotations

import numpy as np

from nimare.meta.cbma.censored import (
    ObservationState,
    bounds_from_states,
    fit_censored,
    profile_interval,
    retention_roles,
)

#: Distances below this (in millimetres) count a printed peak as sitting *at* the location, so
#: its height is the effect there rather than a bound on it. Not a tunable: it exists only to
#: absorb floating-point error in a coordinate transform.
COINCIDENT_MM = 1e-6


def records_for_location(
    position,
    studies,
    reach,
    *,
    image_values=None,
    image_variances=None,
):
    r"""Build one record per study at a single location.

    Parameters
    ----------
    position : :obj:`numpy.ndarray`
        The location, in the same millimetre frame as each study's ``peaks``.
    studies : :obj:`list` of :obj:`dict`
        One entry per coordinate study, with ``peaks`` as an ``(n, 3)`` array of millimetre
        coordinates, ``heights`` as the matching effect-size magnitudes, ``threshold`` as that
        study's reporting threshold on the effect-size scale, and ``variance`` as the sampling
        variance of its estimate at this location.
    reach : :obj:`float`
        How far a printed peak is taken to speak for, in millimetres. Stands in for a cluster
        extent that papers do not report reliably; see the module docstring on why there is no
        safe default.
    image_values, image_variances : :obj:`numpy.ndarray`, optional
        Estimates and sampling variances from studies that supplied an unthresholded image.
        These enter as :attr:`~censored.ObservationState.IMAGE` records, which retention never
        touches: an available image is available whatever the table printed.

    Returns
    -------
    lower, upper, variances, roles : :obj:`numpy.ndarray`
        Ready for :func:`~censored.fit_censored`.
    """
    states, values, thresholds, variances = [], [], [], []
    if image_values is not None:
        image_values = np.asarray(image_values, dtype=float).reshape(-1)
        image_variances = np.asarray(image_variances, dtype=float).reshape(-1)
        if image_values.shape != image_variances.shape:
            raise ValueError(
                f"image_values and image_variances must match; got {image_values.shape} and "
                f"{image_variances.shape}."
            )
        for value, variance in zip(image_values, image_variances):
            states.append(ObservationState.IMAGE)
            values.append(value)
            thresholds.append(0.0)
            variances.append(variance)

    position = np.asarray(position, dtype=float).reshape(3)
    for study in studies:
        peaks = np.asarray(study["peaks"], dtype=float).reshape(-1, 3)
        heights = np.asarray(study["heights"], dtype=float).reshape(-1)
        if peaks.shape[0] != heights.shape[0]:
            raise ValueError(
                f"study {study.get('id', '?')!r} has {peaks.shape[0]} peaks and "
                f"{heights.shape[0]} heights."
            )
        nearby = np.array([], dtype=int)
        if peaks.size:
            distance = np.linalg.norm(peaks - position[None, :], axis=1)
            nearby = np.flatnonzero((distance <= reach) & (heights > 0))
        if nearby.size:
            chosen = nearby[np.argmax(heights[nearby])]
            at_location = float(distance[chosen]) <= COINCIDENT_MM
            states.append(ObservationState.EXACT if at_location else ObservationState.CLUSTER_PEAK)
            values.append(float(heights[chosen]))
        else:
            states.append(ObservationState.NO_PEAK_NEARBY)
            values.append(0.0)
        thresholds.append(float(study["threshold"]))
        variances.append(float(study["variance"]))

    lower, upper = bounds_from_states(
        states,
        values=np.asarray(values, dtype=float),
        thresholds=np.asarray(thresholds, dtype=float),
        signs=np.ones(len(states)),
    )
    return lower, upper, np.asarray(variances, dtype=float), retention_roles(states)


def fit_locations(
    positions,
    studies_at,
    reach,
    *,
    retention=1.0,
    fixed_between_variance=None,
    level=0.95,
    images_at=None,
):
    r"""Fit every location independently and return estimates, intervals and failures.

    Parameters
    ----------
    positions : :obj:`numpy.ndarray`
        ``(n, 3)`` millimetre coordinates of the locations to fit.
    studies_at : callable
        ``studies_at(index)`` returns the per-study dictionaries for location ``index``, as
        :func:`records_for_location` documents. A callable rather than an array because a
        study's sampling variance and threshold can vary by location and materialising every
        record for a whole mask at once is the one thing that makes this intractable.
    reach : :obj:`float`
        Millimetres a printed peak speaks for.
    retention : :obj:`float`, default=1.0
        Probability that an effect clearing its threshold produced a printed peak within the
        reach. At ``1.0`` a silence is read as certain, which is the model whose score is
        provably too negative; see ``proofs/what_a_silence_says.py``. It can be derived from the
        reporting rule and an assumed smoothness rather than fitted.
    fixed_between_variance : :obj:`float`, optional
        Hold :math:`\tau^2` at this value instead of estimating it. ``0.0`` is right only where
        every study really does draw from one population.
    level : :obj:`float`, default=0.95
        Confidence level of the profile interval.
    images_at : callable, optional
        ``images_at(index)`` returns ``(values, variances)`` for the image channel at that
        location, or ``None``.

    Returns
    -------
    :obj:`dict`
        Arrays ``estimate``, ``lower``, ``upper``, ``between_variance``, and the boolean
        ``valid``, ``converged`` and ``touched_search_limit``. Failures are returned rather than
        dropped: a summary computed over only the locations that converged is a summary of a
        subset chosen by the outcome.
    """
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    count = positions.shape[0]
    out = {
        "estimate": np.full(count, np.nan),
        "lower": np.full(count, np.nan),
        "upper": np.full(count, np.nan),
        "between_variance": np.full(count, np.nan),
        "valid": np.zeros(count, dtype=bool),
        "converged": np.zeros(count, dtype=bool),
        "touched_search_limit": np.zeros(count, dtype=bool),
    }
    shrink = (
        {}
        if fixed_between_variance is None
        else {"fixed_between_variance": float(fixed_between_variance)}
    )

    for index in range(count):
        images = images_at(index) if images_at is not None else None
        lower, upper, variances, roles = records_for_location(
            positions[index],
            studies_at(index),
            reach,
            image_values=None if images is None else images[0],
            image_variances=None if images is None else images[1],
        )
        # Retention is passed only where it can act. At 1.0 the mixture is the certain-silence
        # reading, so supplying it would change nothing while suggesting otherwise.
        extra = (
            {"retention": retention, "roles": roles}
            if (retention < 1.0 and (roles != 0).any())
            else {}
        )
        fit = fit_censored(lower, upper, variances, **shrink, **extra)
        interval = profile_interval(lower, upper, variances, level=level, **shrink, **extra)
        out["estimate"][index] = fit["mean"]
        out["between_variance"][index] = fit["between_variance"]
        out["valid"][index] = bool(fit["valid"])
        out["converged"][index] = bool(fit["converged"])
        out["lower"][index] = interval["lower"]
        out["upper"][index] = interval["upper"]
        out["touched_search_limit"][index] = bool(interval.get("touched_search_limit", False))
    return out
