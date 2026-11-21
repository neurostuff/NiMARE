r"""The two objects the design document asks for: a reporting calibration, and an estimator.

The document's build order names ``ReportingCalibration.fit(...)`` and ``MixedEffectSize(...)``.
Both were previously built as functions, on the reasoning that a calibration object with nothing
to calibrate against would be a shell. That reasoning no longer holds: there are now two
selectors that read data rather than an assumption, so the object has something to do.

:class:`ReportingCalibration`
    Estimates the reporting model from studies that supply **both** an unthresholded map and
    their own table. Two settings, two different kinds of estimation, which is why they cannot
    share a route:

    * the **retention** is a parameter of a fixed likelihood, so it comes from a profile pooled
      across locations with a free mean at each. Pooled, not per-location: an absence term is
      strictly decreasing in the retention, so a location with no report maximises at the floor
      whatever the corpus retains, and on a sparse corpus most locations have no report.
    * the **reach** is not a parameter of any one likelihood -- changing it changes which records
      exist, so likelihoods at two reaches describe different data. It comes instead from the
      balance :math:`F\alpha(R) = S\beta(R)` between false reports and false silences, the
      condition that leaves the estimating equation unbiased.

    The two are then one knob rather than two: :math:`\hat\rho = (1-\beta) + F\alpha/S`, so at
    the balanced reach the data asks for full retention exactly, and a shorter reach buys a
    retention below one by precisely the false reports the smaller ball no longer admits. The
    object reports the whole curve for that reason, and a caller who wants a sensitivity analysis
    should walk it rather than take the product of two ranges.

:class:`MixedEffectSize`
    Fits the combined likelihood at supplied settings. It does **not** default them: a reach and
    a retention chosen without data are an assumption choosing the answer's level, and the
    estimate is not monotone in the reach, so even bracketing them requires evaluating the
    interior. Pass a fitted :class:`ReportingCalibration` to take its settings, or supply them
    and own the assumption.

Neither class adds a model. They are the documented entry points to
:mod:`~nimare.meta.cbma.censored` and :mod:`~nimare.meta.cbma.driver`, and every number they
return comes from those.
"""

from __future__ import annotations

import numpy as np

from nimare.meta.cbma.censored import fit_censored
from nimare.meta.cbma.driver import fit_locations, records_for_location

#: Retention grid the pooled profile is evaluated on, denser near the floor where a sparse
#: corpus's per-location fits pile up.
RETENTION_GRID = (
    0.02,
    0.05,
    0.08,
    0.12,
    0.18,
    0.25,
    0.32,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.97,
    1.0,
)

#: Half the chi-square(1) 95% point: the profile-likelihood cutoff for one parameter.
_CUTOFF = 1.9207


def _pooled_retention_profile(records, grid):
    """Profile the summed log-likelihood over one shared retention, means free per location."""
    curve = np.full(len(grid), np.nan)
    for index, candidate in enumerate(grid):
        total, used = 0.0, 0
        for lower, upper, variances, roles in records:
            if not np.any(np.asarray(roles) != 0):
                continue
            fit = fit_censored(
                lower,
                upper,
                variances,
                roles=roles,
                retention=float(candidate),
                fixed_between_variance=0.0,
            )
            if not fit.get("valid"):
                continue
            total += float(fit["loglik"])
            used += 1
        curve[index] = total if used else np.nan
    if not np.any(np.isfinite(curve)):
        return None
    best = int(np.nanargmax(np.where(np.isfinite(curve), curve, -np.inf)))
    inside = np.nonzero(curve >= curve[best] - _CUTOFF)[0]
    return {
        "grid": [float(value) for value in grid],
        "curve": [float(value) for value in curve],
        "retention": float(grid[best]),
        "interval": (float(grid[inside[0]]), float(grid[inside[-1]])),
        "at_grid_edge": bool(best in (0, len(grid) - 1)),
    }


def _error_rates(maps, reaches):
    r"""False-report and false-silence rates at each candidate reach, from maps and their tables.

    ``alpha(R) = P(peak within R | Y < c)`` and ``beta(R) = P(no peak within R | Y >= c)``, both
    conditional on a study's own value at a location, which is why this needs maps: a corpus of
    tables alone cannot run it. Peaks are restricted to the positive ones, since a printed
    negative peak is not a report for the positive tail.
    """
    from scipy.spatial import cKDTree

    distances, exceedances = [], []
    for study in maps:
        heights = np.asarray(study["heights"], dtype=float).reshape(-1)
        peaks = np.asarray(study["peaks"], dtype=float).reshape(-1, 3)
        if peaks.shape[0] != heights.size:
            raise ValueError(
                f"study {study.get('id', '?')!r} has {peaks.shape[0]} peaks and "
                f"{heights.size} heights."
            )
        positive = heights > 0
        if not positive.any():
            continue
        tree = cKDTree(peaks[positive])
        distance, _ = tree.query(np.asarray(study["positions"], dtype=float))
        distances.append(distance)
        exceedances.append(np.asarray(study["values"], dtype=float) >= float(study["threshold"]))
    if not distances:
        return None
    distance = np.concatenate(distances)
    exceed = np.concatenate(exceedances).astype(bool)
    survival = float(exceed.mean())
    below = ~exceed
    # Both rates are conditional, so an empty class makes one of them undefined. Refusing is the
    # only honest option: a mean over nothing is nan, and a nan propagates into a "crossing" that
    # looks like an answer. The same shape of defect once turned an empty threshold bin into the
    # claim that a suprathreshold voxel is never reported.
    if not exceed.any() or not below.any():
        raise ValueError(
            f"The supplied maps put {int(exceed.sum())} of {exceed.size} locations above their "
            "threshold and the rest below, so one of the two error rates is conditional on an "
            "empty set and the balance condition has nothing to solve."
        )
    alpha = np.array([float((distance[below] <= value).mean()) for value in reaches])
    beta = np.array([float((distance[exceed] > value).mean()) for value in reaches])
    return {
        "reaches": np.asarray(reaches, dtype=float),
        "alpha": alpha,
        "beta": beta,
        "exceedance": survival,
        "imbalance": (1.0 - survival) * alpha - survival * beta,
    }


class ReportingCalibration:
    r"""Estimate the reach and the retention from studies that supply maps and their own tables.

    Parameters
    ----------
    reaches : sequence of :obj:`float`, optional
        Candidate reaches in millimetres for the balance condition. The default spans 0 to 24 mm
        at a quarter-millimetre step, which resolves the crossing without pretending to more
        precision than the rates behind it carry.
    retention_grid : sequence of :obj:`float`, optional
        Retentions for the pooled profile.

    Attributes
    ----------
    reach_ : :obj:`float` or :obj:`None`
        Where the two error counts balance. ``None`` if the imbalance never changes sign, which
        means one error dominates at every reach -- a finding about the corpus rather than a
        value to substitute.
    retention_ : :obj:`float` or :obj:`None`
        The pooled profile's maximiser at ``reach_``, once :meth:`fit_retention` has run.
    implied_retention_ : :obj:`float` or :obj:`None`
        ``1 - beta`` at the balanced reach: the retention the reach itself implies, which is a
        second measurement of the same quantity from the maps rather than from the report rates.
    """

    def __init__(self, reaches=None, retention_grid=RETENTION_GRID):
        self.reaches = (
            np.arange(0.0, 24.01, 0.25) if reaches is None else np.asarray(reaches, dtype=float)
        )
        self.retention_grid = tuple(float(value) for value in retention_grid)
        self.rates_ = None
        self.reach_ = None
        self.retention_ = None
        self.retention_interval_ = None
        self.implied_retention_ = None
        self.profile_ = None

    def fit(self, maps):
        """Locate the balanced reach from each map's own values and its own table.

        Parameters
        ----------
        maps : :obj:`list` of :obj:`dict`
            One entry per study that supplies a map, with ``positions`` an ``(n, 3)`` array of
            millimetre coordinates, ``values`` that study's effect size at each, ``peaks`` its
            printed peak coordinates, ``heights`` their effect sizes, and ``threshold`` its
            reporting threshold on the effect-size scale -- supplied, never inferred from the
            smallest printed height.
        """
        rates = _error_rates(maps, self.reaches)
        if rates is None:
            raise ValueError("No supplied map printed a positive peak, so no reach is estimable.")
        self.rates_ = rates
        signs = np.sign(rates["imbalance"])
        changes = np.nonzero(np.diff(signs) != 0)[0]
        if changes.size == 0:
            self.reach_ = None
            return self
        index = int(changes[0])
        low, high = rates["imbalance"][index], rates["imbalance"][index + 1]
        weight = 0.0 if high == low else -low / (high - low)
        self.reach_ = float(
            rates["reaches"][index]
            + weight * (rates["reaches"][index + 1] - rates["reaches"][index])
        )
        self.implied_retention_ = 1.0 - float(
            np.interp(self.reach_, rates["reaches"], rates["beta"])
        )
        return self

    def fit_retention(self, positions, studies_at, images_at=None, reach=None):
        """Pool the locations and profile the shared retention at a reach.

        Parameters
        ----------
        positions : :obj:`numpy.ndarray`
            Locations to pool over, one row of millimetre coordinates each.
        studies_at : callable
            ``studies_at(index)`` returning the coordinate studies at that location, in the form
            :func:`~nimare.meta.cbma.driver.records_for_location` takes.
        images_at : callable, optional
            ``images_at(index)`` returning ``(values, variances)`` from studies with maps.
        reach : :obj:`float`, optional
            Defaults to :attr:`reach_`, so :meth:`fit` runs first.
        """
        reach = self.reach_ if reach is None else float(reach)
        if reach is None:
            raise ValueError("No reach: call fit() first, or supply one.")
        records = []
        for index, position in enumerate(np.asarray(positions, dtype=float)):
            images = images_at(index) if images_at is not None else (None, None)
            records.append(
                records_for_location(
                    position,
                    studies_at(index),
                    reach,
                    image_values=images[0],
                    image_variances=images[1],
                )
            )
        profile = _pooled_retention_profile(records, self.retention_grid)
        if profile is None:
            raise ValueError(
                "No location produced a usable fit, so the retention is not profiled."
            )
        self.profile_ = profile
        self.retention_ = profile["retention"]
        self.retention_interval_ = profile["interval"]
        return self

    def curve(self):
        r"""Return the ``(reach, retention)`` pairs the data implies, as a sensitivity path.

        Returns the retention each candidate reach implies through
        :math:`(1-\beta) + F\alpha/S`, clipped to one. This is the set worth walking in a
        sensitivity analysis: the rectangle spanned by a reach range and a retention range
        contains pairs that double-count the same ambiguity.
        """
        if self.rates_ is None:
            raise ValueError("Call fit() first.")
        survival = self.rates_["exceedance"]
        if survival <= 0:
            raise ValueError("No location in the supplied maps cleared its threshold.")
        implied = (1.0 - self.rates_["beta"]) + (1.0 - survival) * self.rates_["alpha"] / survival
        return {
            "reaches": self.rates_["reaches"].tolist(),
            "retentions": np.clip(implied, 0.0, 1.0).tolist(),
        }


class MixedEffectSize:
    """Fit the combined coordinate-and-image likelihood at supplied settings.

    Parameters
    ----------
    reach : :obj:`float` or :class:`ReportingCalibration`
        Millimetres a printed peak is taken to speak for, or a fitted calibration to take it
        from. There is no default: the estimate is not monotone in the reach, so a default would
        be an assumption choosing the answer's level.
    retention : :obj:`float`, optional
        Probability that an effect clearing its threshold was printed. Taken from the calibration
        when one is supplied and this is left unset; ``1.0`` asserts every exceedance is printed,
        which is not a neutral simplification.
    fixed_between_variance : :obj:`float`, optional
        Hold the between-study variance rather than profiling it.
    level : :obj:`float`, default=0.95
        Interval level.
    """

    def __init__(self, reach, retention=None, fixed_between_variance=None, level=0.95):
        if isinstance(reach, ReportingCalibration):
            calibration = reach
            if calibration.reach_ is None:
                raise ValueError(
                    "The supplied calibration has no reach: its error counts never balanced, so "
                    "it cannot furnish one."
                )
            self.reach = float(calibration.reach_)
            if retention is None:
                retention = (
                    calibration.retention_
                    if calibration.retention_ is not None
                    else calibration.implied_retention_
                )
            self.calibration = calibration
        else:
            self.reach = float(reach)
            self.calibration = None
        if retention is None:
            raise ValueError(
                "A retention is required. Supply one, or fit a ReportingCalibration and pass it, "
                "rather than leaving the reporting model to a default."
            )
        self.retention = float(retention)
        self.fixed_between_variance = fixed_between_variance
        self.level = float(level)
        self.estimate_ = None
        self.interval_ = None
        self.failures_ = None

    def fit(self, positions, studies_at, images_at=None):
        """Fit every location and keep estimates, intervals and failures.

        Returns ``self``. Failures are returned rather than dropped, since a location that could
        not be fitted is information about the corpus.
        """
        out = fit_locations(
            positions,
            studies_at,
            self.reach,
            retention=self.retention,
            fixed_between_variance=self.fixed_between_variance,
            level=self.level,
            images_at=images_at,
        )
        self.estimate_ = out["estimate"]
        self.interval_ = (out["lower"], out["upper"])
        self.failures_ = ~out["valid"]
        self.result_ = out
        return self
