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

**But the two are one knob, not two, and that changes what an envelope has to cover.** Write
:math:`\alpha(R)` for the chance a peak falls within the reach where the study's own value never
cleared its threshold and :math:`\beta(R)` for the chance none does where it did. The retention
the likelihood then wants is :math:`(1-\beta) + F\alpha/S` -- the false-silence complement
*plus the false reports it silently absorbs* -- so at the reach where the two errors are equally
numerous, which is the reach that leaves the estimating equation unbiased, the data asks for full
retention exactly. Shortening the reach buys a retention below one by precisely the false reports
the smaller ball no longer admits. So the honest envelope runs along a curve through the
``(reach, retention)`` plane rather than over its rectangle, and a pair chosen off that curve is
double-counting or double-discounting the same ambiguity. :func:`envelope_over_assumptions`
still evaluates the full grid, because locating the curve needs each study's own map and a
caller may not have one; where the maps exist the curve is the shorter and better-founded
sensitivity analysis.

**Neither setting has to be guessed when unthresholded maps are available.** The retention is a
parameter of a fixed likelihood, so pooling locations with a free mean at each locates it by
maximum likelihood -- one per-location fit cannot, since an absence term is strictly decreasing
in the retention and a location with no report maximises at the floor whatever the corpus does.
The reach is not a parameter of any one likelihood, since changing it changes which records
exist; it comes instead from the balance :math:`F\alpha(R) = S\beta(R)` above, measured on the
maps the caller already holds.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from nimare.meta.cbma.censored import (
    ObservationState,
    bounds_from_states,
    fit_censored,
    loglik_over_means,
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
    sided="two",
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
    sided : ``{"one", "two"}``, default="two"
        Which reporting protocol the tables came from, overridable per study with a ``sided``
        key. **Two-sided by default**, because the estimand is a signed effect size, an
        unthresholded map carries both signs, and on a corpus of 158 published faces tables
        reading the signs correctly moved the combined fit from rmse .4360 to .4066 against an
        external reference, cut the bias from -.108 to -.066, and raised the correlation from
        .332 to .417. At certain silence the gain is larger still, .5551 to .4355: a table's
        deactivations are informative, and discarding them is not a conservative choice. This is a
        property of the *protocol* and cannot be assumed globally: a paper that
        prints only activations really does have a one-sided rule, and its silence really does
        mean :math:`Y < c`.

        ``"one"``
            Only a positive peak is a report; a printed negative peak is discarded and its study
            enters as an absence bounded above by its threshold. Lossy but not false, since
            :math:`Y < c` contains the truth.
        ``"two"``
            A printed negative peak is a *signed report* on :math:`[h, -c)`, and an absence means
            :math:`|Y| < c`. Correct where the estimand is signed and the tables came from a
            two-sided rule, which is the usual case.

        **The two halves cannot be mixed.** A symmetric absence combined with discarded negative
        peaks asserts :math:`|Y| < c` where the table says :math:`Y \le -c` -- intervals that are
        *disjoint*, misstating the asserted value by one to two effect sizes at every such
        record. Conversely a one-sided absence under a two-sided rule is asymmetric, and its
        score at a null mean is :math:`-\varphi(c)/\Phi(c)` rather than zero, so every silent
        study pulls a null location downward and the pull accumulates with the corpus size. Both
        are derived in ``proofs/what_a_negative_peak_says.py``.

    Returns
    -------
    lower, upper, variances, roles : :obj:`numpy.ndarray`
        Ready for :func:`~censored.fit_censored`.
    """
    bundle = bundle_studies(studies, sided=sided)
    return records_from_bundle(
        bundle,
        position,
        reach,
        image_values=image_values,
        image_variances=image_variances,
    )


def bundle_studies(studies, sided="two"):
    r"""Flatten a study list into concatenated arrays, so geometry is one call and not one each.

    Profiling a whole-brain fit left record building at about 11 ms per location with 160
    studies -- comparable to the fit itself once that was vectorised -- and the reason is
    structural rather than arithmetic: :func:`records_for_location` loops the studies and takes a
    distance per study, so the cost is 160 numpy calls on six-element arrays, which is call
    overhead. Concatenating once turns that into one distance call over every peak, with the
    per-study selection done by segment reductions.

    Each study's protocol is baked in here as a per-peak eligibility mask, so a per-study
    ``sided`` override survives the flattening rather than being silently replaced by the
    call-level default.
    """
    peaks, heights, eligible, starts, counts = [], [], [], [], []
    thresholds, variances, offset = [], [], 0
    for study in studies:
        study_peaks = np.asarray(study["peaks"], dtype=float).reshape(-1, 3)
        study_heights = np.asarray(study["heights"], dtype=float).reshape(-1)
        if study_peaks.shape[0] != study_heights.shape[0]:
            raise ValueError(
                f"study {study.get('id', '?')!r} has {study_peaks.shape[0]} peaks and "
                f"{study_heights.shape[0]} heights."
            )
        protocol = str(study.get("sided", sided))
        if protocol not in ("one", "two"):
            raise ValueError(
                f"study {study.get('id', '?')!r} has sided={protocol!r}; use 'one' or 'two'."
            )
        peaks.append(study_peaks)
        heights.append(study_heights)
        eligible.append(
            study_heights > 0 if protocol == "one" else np.ones(study_heights.size, dtype=bool)
        )
        starts.append(offset)
        counts.append(study_peaks.shape[0])
        offset += study_peaks.shape[0]
        thresholds.append(float(study["threshold"]))
        variances.append(float(study["variance"]))
    return {
        "peaks": np.concatenate(peaks) if peaks else np.zeros((0, 3)),
        "heights": np.concatenate(heights) if heights else np.zeros(0),
        "eligible": np.concatenate(eligible) if eligible else np.zeros(0, dtype=bool),
        "starts": np.asarray(starts, dtype=int),
        "counts": np.asarray(counts, dtype=int),
        "thresholds": np.asarray(thresholds, dtype=float),
        "variances": np.asarray(variances, dtype=float),
        "sided": [str(study.get("sided", sided)) for study in studies],
        "n_studies": len(studies),
    }


def _chosen_peaks(bundle, position, reach):
    """Index of each study's chosen peak, or ``-1`` where it has none within the reach.

    Implements the same rule as :func:`records_for_location` -- the nearest eligible peak, with
    the larger magnitude breaking a tie -- by segment reductions rather than a loop, so exact
    distance ties still resolve by magnitude and not by table order.
    """
    total = bundle["heights"].size
    if total == 0 or bundle["n_studies"] == 0:
        return np.full(bundle["n_studies"], -1, dtype=int), np.zeros(0)

    distance = np.linalg.norm(bundle["peaks"] - np.asarray(position, dtype=float)[None, :], axis=1)
    usable = bundle["eligible"] & (distance <= reach)
    masked = np.where(usable, distance, np.inf)

    starts, counts = bundle["starts"], bundle["counts"]
    # ``reduceat`` needs every start index to be inside the array, and a study with no peaks has
    # a zero-length segment whose start can equal the length -- which raises rather than
    # returning an empty reduction. Reduce over the studies that *have* peaks and scatter the
    # answer back; a study with none is a silence by definition, not an edge case to patch after.
    present = np.flatnonzero(counts > 0)
    chosen = np.full(bundle["n_studies"], -1, dtype=int)
    if present.size == 0:
        return chosen, distance
    segment_starts = starts[present]
    segment_counts = counts[present]

    nearest = np.minimum.reduceat(masked, segment_starts)
    at_nearest = usable & (masked == np.repeat(nearest, segment_counts))
    magnitude = np.where(at_nearest, np.abs(bundle["heights"]), -np.inf)
    largest = np.maximum.reduceat(magnitude, segment_starts)
    winner = at_nearest & (magnitude == np.repeat(largest, segment_counts))
    positions = np.where(winner, np.arange(total), total)
    picked = np.minimum.reduceat(positions, segment_starts)
    chosen[present] = np.where(picked < total, picked, -1)
    return chosen, distance


def records_from_bundle(bundle, position, reach, *, image_values=None, image_variances=None):
    r"""Build one location's records from a pre-flattened study bundle.

    This is the single implementation: :func:`records_for_location` builds a one-off bundle and
    calls it, so the per-location convenience path and the batch path cannot drift apart. The
    only difference between them is *when* the flattening happens -- once per analysis instead of
    once per location, which is the whole saving.
    """
    states, values, thresholds, variances, signs = [], [], [], [], []
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
            signs.append(1.0)

    chosen, distance = _chosen_peaks(bundle, position, reach)
    for index in range(bundle["n_studies"]):
        pick = int(chosen[index])
        if pick >= 0:
            height = float(bundle["heights"][pick])
            at_location = float(distance[pick]) <= COINCIDENT_MM
            states.append(ObservationState.EXACT if at_location else ObservationState.CLUSTER_PEAK)
            values.append(height)
            signs.append(float(np.sign(height)) or 1.0)
        else:
            states.append(ObservationState.NO_PEAK_NEARBY)
            values.append(0.0)
            signs.append(1.0 if bundle["sided"][index] == "one" else 0.0)
        thresholds.append(float(bundle["thresholds"][index]))
        variances.append(float(bundle["variances"][index]))

    sign_array = np.asarray(signs, dtype=float)
    lower, upper = bounds_from_states(
        states,
        values=np.asarray(values, dtype=float),
        thresholds=np.asarray(thresholds, dtype=float),
        signs=np.where(sign_array == 0.0, np.nan, sign_array),
    )
    return lower, upper, np.asarray(variances, dtype=float), retention_roles(states)


#: Points in the mean grid the fast path evaluates, over a range set by the data and refined
#: quadratically around the maximum. Chosen by measuring the trade rather than by taste, against
#: the exact optimiser on 120 locations of a 160-study corpus:
#:
#: ===========  ==================  ==================
#: grid points  CPU ms per location  largest difference
#: ===========  ==================  ==================
#: 129          7.4                 6.2e-03
#: 257          6.6                 1.7e-03
#: 513          12.6                4.3e-04
#: 1025         23.5                2.1e-04
#: exact        72.9                --
#: ===========  ==================  ==================
#:
#: Those figures are from the first version, whose grid spanned twelve times the *least* precise
#: record's standard error. Scaling the span to the estimate's own precision instead changes the
#: trade entirely, because the points then lie where the likelihood varies:
#:
#: ===========  ===================  ==================
#: grid points  CPU ms per location  largest difference
#: ===========  ===================  ==================
#: 257          7.1                  2.8e-04
#: 513          10.9                  2.3e-04
#: ===========  ===================  ==================
#:
#: So 257 on the narrow span is both faster than 513 on the wide one and four times more
#: accurate, which is what it means for a grid to have been in the wrong place rather than too
#: coarse. 257 it is: 7.1 ms per location is 3.5 minutes for a 29,398-voxel mask on one core.
_GRID_POINTS = 257


def _grid_fit(lower, upper, variances, roles, *, between_variance, retention, cutoff, span=12.0):
    r"""Estimate and interval from a single vectorised pass over a grid of means.

    **Why this exists.** Profiling a whole-brain fit put 65-104 ms per location in the profile
    interval and 30-45 ms in the point fit, against 0.08 ms of record building, and both were
    nearly flat in the number of studies -- an exponent of +0.26 and +0.11 against the study
    count. Flat in the study count means the cost is Python-level iteration, not arithmetic over
    records: the optimiser's calls and the interval's grid loop, each re-entering a scalar
    likelihood. One call to :func:`~censored.loglik_over_means` evaluates the whole grid 8 to 40
    times faster than the equivalent loop, and the *same* curve yields both the maximiser and
    the likelihood-ratio crossings, so the interval costs nothing beyond the fit.

    The estimate is the grid's maximum refined by fitting a parabola through its two neighbours,
    which is exact for a locally quadratic log-likelihood -- and a log-likelihood is locally
    quadratic at an interior maximum, which is the same fact the curvature standard error rests
    on. Agreement with the exact optimiser is asserted in the tests rather than assumed, because
    a faster path that quietly disagrees is not an optimisation.
    """
    informative = np.isfinite(lower) | np.isfinite(upper)
    if not informative.any():
        return None
    finite = np.concatenate([lower[np.isfinite(lower)], upper[np.isfinite(upper)]])
    if finite.size == 0:
        return None
    centre = float(np.mean(finite))

    # The grid must span the plausible range of the *estimate*, whose scale is set by the most
    # precise record, not the least. Using the largest variance made the grid about seven times
    # wider than needed -- 12 times a coordinate study's standard error of 0.32 where the
    # estimate's own was 0.14 -- so most of the points fell where the likelihood is already
    # negligible. The widest record still sets a floor, since with interval records alone the
    # maximum can sit well away from the bounds' centre.
    precision = float(np.sqrt(np.min(variances[informative]) + max(between_variance, 0.0)))
    coarse = float(np.sqrt(np.max(variances[informative]) + max(between_variance, 0.0)))
    width = span * max(precision, 0.25 * coarse)

    # Widen once if the maximum lands on an edge: a narrower grid is only a saving while it still
    # contains the answer, and silently returning a boundary as an estimate is the failure mode
    # this whole module has been bitten by before.
    curve = usable = means = None
    for attempt in range(3):
        means = np.linspace(centre - width, centre + width, _GRID_POINTS)
        curve = loglik_over_means(
            means, between_variance, lower, upper, variances, retention=retention, roles=roles
        )
        usable = np.isfinite(curve)
        if not usable.any():
            return None
        best = int(np.nanargmax(np.where(usable, curve, -np.inf)))
        if 0 < best < means.size - 1:
            break
        width *= 4.0
    best = int(np.nanargmax(np.where(usable, curve, -np.inf)))
    peak = float(curve[best])

    estimate = float(means[best])
    if 0 < best < means.size - 1 and usable[best - 1] and usable[best + 1]:
        left, centre_value, right = curve[best - 1], curve[best], curve[best + 1]
        denominator = left - 2.0 * centre_value + right
        if denominator < 0:
            step = float(means[1] - means[0])
            shift = 0.5 * float(left - right) / float(denominator)
            # A parabola through three points puts its vertex within half a step of the middle
            # one; anything else means the curve is not locally quadratic there and the grid
            # maximum is the better answer.
            if abs(shift) <= 0.5:
                estimate = float(means[best] + shift * step)

    # The interval is the outermost pair of points where the curve is still within the cutoff of
    # its maximum, interpolated linearly between the bracketing grid points exactly as the
    # scalar search does.
    inside = np.flatnonzero(usable & (curve >= peak - cutoff))
    bounds = []
    touched = False
    for side, index in (("lower", inside[0]), ("upper", inside[-1])):
        step_out = -1 if side == "lower" else 1
        neighbour = index + step_out
        if neighbour < 0 or neighbour >= means.size or not usable[neighbour]:
            bounds.append(float(means[index]))
            touched = True
            continue
        gap_inside = peak - float(curve[index])
        gap_outside = peak - float(curve[neighbour])
        weight = (
            0.0
            if gap_outside == gap_inside
            else (cutoff - gap_inside) / (gap_outside - gap_inside)
        )
        bounds.append(float(means[index] + weight * (means[neighbour] - means[index])))
    return {
        "mean": estimate,
        "loglik": peak,
        "lower": bounds[0],
        "upper": bounds[1],
        "touched_search_limit": touched,
        "n_informative": int(informative.sum()),
    }


def _fit_one(index, position, studies_at, images_at, settings, cache):
    """Fit one location and return its row, shared by the serial loop and the workers.

    One implementation for both, so a parallel run cannot compute something a serial run would
    not. ``cache`` is a single-entry, caller-owned dictionary holding the flattened bundle for
    the study list last seen, which is the same object for every location in every caller here.

    **Not a module-level cache.** The first version kept one, keyed on ``id(studies)``, and the
    tests caught it immediately: a two-sided call and a one-sided call on the same study list
    returned identical estimates, because the second reused the first's bundle and the protocol
    is baked into it. A global keyed on an object's identity is also unsound on its own terms,
    since an id is reusable once the object is collected. Caller-owned state cannot leak between
    calls, and per-chunk state cannot leak between workers.
    """
    images = images_at(index) if images_at is not None else None
    here = studies_at(index)
    bundle = cache.get("bundle")
    if bundle is None or bundle[0] is not here:
        bundle = (here, bundle_studies(here, sided=settings["sided"]))
        cache["bundle"] = bundle
    lower, upper, variances, roles = records_from_bundle(
        bundle[1],
        position,
        settings["reach"],
        image_values=None if images is None else images[0],
        image_variances=None if images is None else images[1],
    )
    # Retention is passed only where it can act. At 1.0 the mixture is the certain-silence
    # reading, so supplying it would change nothing while suggesting otherwise.
    retention = settings["retention"]
    extra = (
        {"retention": retention, "roles": roles}
        if (retention < 1.0 and (roles != 0).any())
        else {}
    )
    shrink = settings["shrink"]
    between = float(shrink.get("fixed_between_variance") or 0.0)
    if settings["method"] == "grid":
        grid = _grid_fit(
            lower,
            upper,
            variances,
            roles,
            between_variance=between,
            retention=extra.get("retention"),
            cutoff=settings["cutoff"],
        )
        if grid is None:
            return None
        return {
            "estimate": grid["mean"],
            "between_variance": between,
            "valid": True,
            "converged": True,
            "lower": grid["lower"],
            "upper": grid["upper"],
            "touched_search_limit": bool(grid["touched_search_limit"]),
        }

    fit = fit_censored(lower, upper, variances, **shrink, **extra)
    interval = profile_interval(
        lower, upper, variances, level=settings["level"], **shrink, **extra
    )
    return {
        "estimate": fit["mean"],
        "between_variance": fit["between_variance"],
        "valid": bool(fit["valid"]),
        "converged": bool(fit["converged"]),
        "lower": interval["lower"],
        "upper": interval["upper"],
        "touched_search_limit": bool(interval.get("touched_search_limit", False)),
    }


def _fit_chunk(indices, positions, studies_at, images_at, settings):
    """Fit one contiguous block of locations and return its rows.

    A module-level function rather than a closure because joblib has to ship it to a worker.
    The *global* index is passed through, not the position within the chunk: ``studies_at`` and
    ``images_at`` are indexed by location, so renumbering them would hand each worker a
    different corpus than the serial path would -- silently, and only for locations after the
    first chunk.
    """
    rows = []
    cache = {}
    for index in indices:
        rows.append(
            (
                index,
                _fit_one(
                    int(index),
                    positions[int(index)],
                    studies_at,
                    images_at,
                    settings,
                    cache,
                ),
            )
        )
    return rows


def fit_locations(
    positions,
    studies_at,
    reach,
    *,
    retention=1.0,
    fixed_between_variance=None,
    level=0.95,
    images_at=None,
    sided="two",
    method="exact",
    n_jobs=1,
):
    r"""Fit every location independently and return estimates, intervals and failures.

    ``n_jobs`` splits the locations across processes. Every location is an independent fit, so
    this is embarrassingly parallel and the only reason it is not the default is that a process
    pool costs a second or two to start, which dominates a small run. ``0`` or a negative value
    means every core, following :func:`~nimare.utils._check_ncores`. Locations are dispatched in
    contiguous chunks rather than one task each: at a few milliseconds per location the dispatch
    overhead would otherwise be a large share of the work.

    ``method="grid"`` replaces the per-location optimiser and interval search with a single
    vectorised pass over a grid of means (:func:`_grid_fit`), which profiling identified as the
    whole cost of a whole-brain fit: 65-104 ms per location in the interval and 30-45 ms in the
    fit, both nearly flat in the study count, against 0.08 ms of record building. It requires a
    fixed between-study variance, since the grid is over the mean alone. ``"exact"`` keeps the
    optimiser and is the reference the fast path is checked against.

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
    if method not in ("exact", "grid"):
        raise ValueError(f"method must be 'exact' or 'grid'; got {method!r}.")
    n_jobs = int(n_jobs)
    if method == "grid" and fixed_between_variance is None:
        raise ValueError(
            "method='grid' profiles the mean on a grid at a fixed between-study variance; "
            "supply fixed_between_variance, or use method='exact' to profile it."
        )
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

    settings = {
        "reach": float(reach),
        "retention": float(retention),
        "sided": sided,
        "method": method,
        "level": float(level),
        "shrink": shrink,
        "cutoff": float(chi2.ppf(level, 1)) / 2.0,
    }

    if n_jobs == 1:
        rows = _fit_chunk(range(count), positions, studies_at, images_at, settings)
    else:
        from joblib import Parallel, delayed

        from nimare.utils import _check_ncores

        workers = _check_ncores(n_jobs)
        # Chunked rather than one task per location: at 7 ms a location the dispatch overhead
        # would otherwise be a large share of the work. Contiguous chunks also keep the bundle
        # cache useful inside each worker.
        chunks = np.array_split(np.arange(count), max(workers * 4, 1))
        gathered = Parallel(n_jobs=workers)(
            delayed(_fit_chunk)(chunk, positions, studies_at, images_at, settings)
            for chunk in chunks
            if chunk.size
        )
        rows = [row for chunk_rows in gathered for row in chunk_rows]

    for index, row in rows:
        if row is None:
            continue
        for key, value in row.items():
            out[key][int(index)] = value
    return out


def envelope_over_assumptions(
    positions,
    studies_at,
    reaches,
    retentions,
    *,
    fixed_between_variance=None,
    level=0.95,
    images_at=None,
):
    r"""Fit under every combination of reach and retention, and return the range.

    Neither the reach nor the retention can honestly ship as a default. The reach stands in for
    a cluster extent papers do not report, and the retention for the chance that an effect
    clearing its threshold produced a printed peak within that reach. Measured on HCP
    pseudo-studies, only a short reach with a low retention beats an equally regularised
    image-only baseline at all: 12 mm and 20 mm never do, at any retention. A single number from
    a single assumed pair would therefore be a number whose sign the assumption chose.

    **The grid is wider than the set of defensible pairs**, for the reason the module docstring
    gives: the retention the data wants at a given reach is fixed by that reach's false-report
    and false-silence rates, so the pairs worth reporting lie on a curve rather than filling the
    rectangle. Reporting the whole grid is the conservative choice when the maps needed to locate
    that curve are absent, and an over-statement of the range when they are present.

    **Every combination is evaluated, and that is not a performance oversight.** The implied
    magnitude in :func:`~nimare.meta.cbma.sensitivity.retention_envelope` is monotone in its
    retention, so its endpoints suffice. This estimate is *not* monotone in the reach: its bias
    runs -0.047, -0.026, +0.039, +0.245 across 4, 8, 12 and 20 mm, changing sign in the middle,
    so evaluating the corners would report a range that excludes the interior it brackets. Any
    reasoning by analogy to the other envelope is wrong here.

    Parameters
    ----------
    positions, studies_at, images_at, fixed_between_variance, level
        As :func:`fit_locations`.
    reaches, retentions : sequence of :obj:`float`
        The assumption grid. Both are swept in full.

    Returns
    -------
    :obj:`dict`
        ``lowest`` and ``highest`` estimates per location over the grid, ``interval_lower`` and
        ``interval_upper`` as the union of the profile intervals, ``spread`` as
        ``highest - lowest``, ``at_lowest`` and ``at_highest`` naming the ``(reach, retention)``
        that produced each extreme, ``grid`` as the pairs evaluated, and
        ``coverage_semantics``.

        The union of intervals is **not** a confidence interval and its width does not shrink
        with more studies: it is the set of answers the assumptions permit. Reporting it as one
        would attach a coverage statement that nothing here establishes.
    """
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    count = positions.shape[0]
    reaches = [float(value) for value in reaches]
    retentions = [float(value) for value in retentions]
    if not reaches or not retentions:
        raise ValueError("Both reaches and retentions must be non-empty.")

    lowest = np.full(count, np.inf)
    highest = np.full(count, -np.inf)
    interval_lower = np.full(count, np.inf)
    interval_upper = np.full(count, -np.inf)
    at_lowest = np.empty(count, dtype=object)
    at_highest = np.empty(count, dtype=object)

    grid = [(reach, retention) for reach in reaches for retention in retentions]
    for reach, retention in grid:
        fitted = fit_locations(
            positions,
            studies_at,
            reach,
            retention=retention,
            fixed_between_variance=fixed_between_variance,
            level=level,
            images_at=images_at,
        )
        estimate = fitted["estimate"]
        usable = np.isfinite(estimate)
        improve_low = usable & (estimate < lowest)
        improve_high = usable & (estimate > highest)
        lowest = np.where(improve_low, estimate, lowest)
        highest = np.where(improve_high, estimate, highest)
        # Assigned per index rather than by mask: a list of tuples assigned into a boolean
        # slice of an object array is read as a 2-D array by numpy and refused.
        for index in np.flatnonzero(improve_low):
            at_lowest[index] = (reach, retention)
        for index in np.flatnonzero(improve_high):
            at_highest[index] = (reach, retention)
        interval_lower = np.where(
            np.isfinite(fitted["lower"]) & (fitted["lower"] < interval_lower),
            fitted["lower"],
            interval_lower,
        )
        interval_upper = np.where(
            np.isfinite(fitted["upper"]) & (fitted["upper"] > interval_upper),
            fitted["upper"],
            interval_upper,
        )

    unreached = ~np.isfinite(lowest)
    lowest = np.where(unreached, np.nan, lowest)
    highest = np.where(~np.isfinite(highest), np.nan, highest)
    interval_lower = np.where(np.isfinite(interval_lower), interval_lower, np.nan)
    interval_upper = np.where(np.isfinite(interval_upper), interval_upper, np.nan)
    return {
        "lowest": lowest,
        "highest": highest,
        "spread": highest - lowest,
        "interval_lower": interval_lower,
        "interval_upper": interval_upper,
        "at_lowest": at_lowest,
        "at_highest": at_highest,
        "grid": grid,
        "coverage_semantics": (
            "The range of estimates the assumptions tried permit, and nothing more. It is not a "
            "confidence interval, its width does not shrink with more studies, and it carries no "
            "coverage statement unless the grid is asserted to contain the truth -- which this "
            "function never infers."
        ),
    }
