r"""Reporting uncertainty as scenarios and envelopes, not as a free parameter.

Section 5 of the revised plan: "Do not use a free per-study retention curve to absorb the
differences between predicted and observed tables. Start with a restricted reporting family",
and "report separate scenario results or an envelope; do not average arbitrarily chosen
scenarios and call the average calibrated."

For the scalar positive-tail illustration a report rate :math:`q` under retention :math:`\rho`
and threshold :math:`c` inverts to

.. math:: m(\rho) = c + \sigma\,\Phi^{-1}\!\left(\frac{q}{\rho}\right),

which is **strictly decreasing** in :math:`\rho`: a literature that prints less of what it finds
must have had a larger effect to leave the same trace. So an envelope over a retention range is
determined by its endpoints and :func:`retention_envelope` evaluates only those. Derived in
``proofs/reporting_sensitivity_envelope.py``.

Three properties decide how the output may be described.

**It diverges.** As :math:`\rho \to q^+` the implied magnitude is unbounded, because a corpus can
print a small fraction of a huge effect or all of a modest one and leave the same rate. At
:math:`q=.1,\ c=.6,\ \sigma=.25`, a floor of :math:`\rho=.3` gives a width of :math:`.18` and a
floor of :math:`.101` gives :math:`.87`. An envelope reaching down to the observed rate is
infinite, and that is the identification speaking rather than the arithmetic failing.

**Its width ignores the data.** The width is
:math:`\sigma[\Phi^{-1}(q/\rho_{\min}) - \Phi^{-1}(q/\rho_{\max})]` -- the scenario set and the
threshold, with no sample size in it. A confidence interval narrows as evidence accumulates;
this does not, at any number of studies. :class:`ReportingEnvelope` therefore refuses to be
called an interval, and its ``coverage_semantics`` says so in the object: one if the true
retention lies in the set, zero if it does not, and nothing in between.

**Averaging it is a prior, not a calibration.** :func:`scenario_average` will compute a weighted
average, and returns alongside it the Jensen gap against the magnitude at the average retention
and the weights it used, because reporting the number without the weights asserts a prior
silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


def implied_magnitude(report_rate, retention, threshold, sampling_sd):
    r""":math:`m(\rho) = c + \sigma\,\Phi^{-1}(q/\rho)`, the magnitude a report rate implies.

    Raises when ``retention`` is below ``report_rate``: a corpus cannot print a larger fraction
    than it retains, so such a scenario is not conservative but impossible.
    """
    rate = np.asarray(report_rate, dtype=float)
    keep = np.asarray(retention, dtype=float)
    if np.any(rate <= 0) or np.any(rate >= 1):
        raise ValueError("The report rate must lie strictly between zero and one.")
    if np.any(keep <= 0) or np.any(keep > 1):
        raise ValueError("Retention must lie in (0, 1].")
    if np.any(keep < rate):
        raise ValueError(
            "A retention below the observed report rate is impossible, not conservative: a "
            "corpus cannot print a larger fraction of its findings than it keeps. The implied "
            "magnitude diverges as retention approaches the rate from above."
        )
    return float(threshold) + float(sampling_sd) * norm.ppf(rate / keep)


@dataclass(frozen=True)
class ReportingEnvelope:
    """An envelope over a retention range, and the semantics it is allowed to claim.

    Attributes
    ----------
    lower, upper : :obj:`float`
        The implied magnitudes at the **highest** and **lowest** retention respectively, since
        the mapping is decreasing.
    width : :obj:`float`
        ``upper - lower``. Carries no sample size, at any number of studies.
    coverage_semantics : :obj:`str`
        Fixed text. An envelope is not a confidence interval and this field exists so the
        distinction survives being passed around.
    """

    lower: float
    upper: float
    retention_range: tuple
    report_rate: float
    threshold: float
    sampling_sd: float
    scenario_set_complete: bool = False

    @property
    def width(self):
        """Width of the envelope, which no amount of data reduces."""
        return float(self.upper - self.lower)

    @property
    def coverage_semantics(self):
        """State what the envelope does and does not guarantee."""
        if self.scenario_set_complete:
            return (
                "Coverage is one if the asserted scenario set really contains the reporting "
                "mechanism, and zero otherwise. There is nothing in between, and the width "
                "does not shrink with more studies. This is not a confidence interval."
            )
        return (
            "The scenario set has not been asserted complete, so no coverage statement is "
            "available at all -- only the range of magnitudes consistent with the scenarios "
            "tried. This is not a confidence interval and a wide one is not conservative."
        )

    def as_dict(self):
        """Envelope with its semantics attached, so the two cannot be separated downstream."""
        return {
            "lower": float(self.lower),
            "upper": float(self.upper),
            "width": self.width,
            "retention_range": tuple(float(value) for value in self.retention_range),
            "report_rate": float(self.report_rate),
            "coverage_semantics": self.coverage_semantics,
            "is_confidence_interval": False,
        }


def retention_envelope(
    report_rate, threshold, sampling_sd, retention_range, *, scenario_set_complete=False
):
    r"""Envelope of implied magnitudes over a retention range, from its endpoints.

    Only the endpoints are evaluated, because :math:`m(\rho)` is strictly decreasing in
    :math:`\rho` and the interior therefore adds nothing. ``scenario_set_complete`` is an
    assertion the caller makes about their own scenario set; it is never inferred, and without it
    the envelope carries no coverage statement.
    """
    low, high = (float(value) for value in retention_range)
    if not 0.0 < low <= high <= 1.0:
        raise ValueError("retention_range must be an increasing pair inside (0, 1].")
    # Decreasing in retention, so the *highest* retention gives the *lowest* magnitude.
    lower = implied_magnitude(report_rate, high, threshold, sampling_sd)
    upper = implied_magnitude(report_rate, low, threshold, sampling_sd)
    return ReportingEnvelope(
        lower=lower,
        upper=upper,
        retention_range=(low, high),
        report_rate=float(report_rate),
        threshold=float(threshold),
        sampling_sd=float(sampling_sd),
        scenario_set_complete=bool(scenario_set_complete),
    )


def scenario_average(report_rate, threshold, sampling_sd, retentions, weights=None):
    r"""Weighted average of implied magnitudes, returned **with** its Jensen gap and weights.

    The plan's instruction is not to average scenarios and call the result calibrated. This will
    average them, and returns:

    ``average``
        the weighted mean of :math:`m(\rho)`;
    ``at_average_retention``
        :math:`m` evaluated at the weighted mean retention, which is a different number;
    ``jensen_gap``
        their difference, non-zero because :math:`\Phi^{-1}(q/\rho)` is not affine in
        :math:`\rho`;
    ``weights``
        echoed back, because an average without its weights asserts a prior silently;
    ``is_posterior_mean``
        always ``False``. It is a posterior mean only if the weights were a justified prior, and
        nothing here can establish that.
    """
    values = np.asarray(retentions, dtype=float).reshape(-1)
    if values.size < 2:
        raise ValueError("Averaging needs at least two scenarios.")
    if weights is None:
        weights = np.full(values.size, 1.0 / values.size)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.size != values.size or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must be non-negative, sum to one, and match the scenarios.")

    implied = np.array(
        [implied_magnitude(report_rate, value, threshold, sampling_sd) for value in values]
    )
    average = float(weights @ implied)
    at_average = implied_magnitude(report_rate, float(weights @ values), threshold, sampling_sd)
    return {
        "average": average,
        "at_average_retention": at_average,
        "jensen_gap": float(average - at_average),
        "weights": weights.copy(),
        "scenarios": values.copy(),
        "is_posterior_mean": False,
    }


def envelope_width_is_data_independent(
    report_rate, threshold, sampling_sd, retention_range, study_counts=(10, 100, 1000)
):
    """Demonstrate that the envelope width is the same at every study count.

    A test rather than a computation: the width has no sample size in it, and this returns the
    widths at several counts so a caller can see they are identical. Offered because the
    temptation to read a wide envelope as a conservative interval is strong, and the clearest
    rebuttal is that it does not move when the data grow.
    """
    envelope = retention_envelope(report_rate, threshold, sampling_sd, retention_range)
    return {int(count): envelope.width for count in study_counts}
