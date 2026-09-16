r"""Marginal effect size from images corrected by a coordinate-based prediction.

The target is the marginal population effect :math:`m(v) = \mathbb{E}_{P^*}[\theta_i(v)]`: the mean
standardized effect over studies drawn from a stated target population, including any whose effect
at :math:`v` is null. It is what an image-based meta-analysis estimates, and it is identified by
representative unthresholded images without any latent structure.

Coordinate tables enter as a *control variate* rather than as observations. A predictor turns a
study's table into a guess at its effect image; the guess is allowed to be biased and is never
trusted. With :math:`n` studies that supplied images and :math:`N` coordinate-only studies,

.. math:: \hat m_\lambda(v) = \bar Y_I(v) + \lambda_v\,[\bar f_C(v) - \bar f_I(v)],

where :math:`\bar Y_I` averages the images, :math:`\bar f_I` averages the predictor over the same
studies, and :math:`\bar f_C` averages it over the coordinate-only studies. The predictor's error
is measured on the image cohort and subtracted, so a systematically wrong predictor cancels
instead of propagating. See ``proofs/marginal_control_variate.py`` in the companion experiments
repository for the verified algebra.

What this buys, and its ceiling:

.. math::
    \operatorname{Var}(\hat m_\lambda) = \frac{\operatorname{Var}(Y - \lambda f)}{n}
    + \frac{\lambda^2 \operatorname{Var}(f)}{N},
    \qquad
    \lambda^*_v = \frac{\operatorname{Cov}(Y, f)}{(1 + n/N)\operatorname{Var}(f)},
    \qquad
    \frac{V_*}{V_{\text{images}}} = 1 - \frac{\rho^2}{1 + n/N}.

As :math:`N \to \infty` the ratio tends to :math:`1 - \rho^2`. **Coordinates never substitute for
images**: with a predictor correlating 0.5 with the truth, no quantity of tables reduces the
variance below three quarters of the image-only value.

Warnings
--------
Two assumptions are load-bearing and neither is checkable from the fit alone.

``lambda`` and the predictor must not be fitted on the studies they correct. Fitting either on the
image cohort and then applying it there reintroduces exactly the bias the construction removes.
Supply a coefficient measured elsewhere, or accept the pooled estimate and read the cross-fitted
diagnostic.

The cohorts must be exchangeable, so that :math:`\mathbb{E}f_C = \mathbb{E}f_I`. Image sharing is
not random: if studies that share images differ from those that do not, the correction is biased
by that difference. ``cohort_shift`` and its standard error are reported so the assumption can be
inspected, but a shift indistinguishable from zero is not evidence of exchangeability.
"""

from __future__ import annotations

import numpy as np

COEFFICIENT_MODES = ("pooled", "voxelwise")


def optimal_coefficient(values, predictions, n_images, n_coordinates, mode="pooled"):
    r"""Variance-minimising :math:`\lambda`, from the image cohort's covariance.

    Parameters
    ----------
    values, predictions : :obj:`numpy.ndarray` of shape (n_images, n_voxels)
        Per-study effect estimates and the predictor evaluated on the same studies.
    n_images, n_coordinates : :obj:`int`
        Cohort sizes. Their ratio shrinks the coefficient: the correction borrows precision from
        the coordinate cohort, so a small one is trusted less.
    mode : {"pooled", "voxelwise"}
        ``"pooled"`` uses one coefficient for the whole volume, computed from every
        (study, voxel) pair. ``"voxelwise"`` fits each voxel separately, which is unbiased for
        the per-voxel optimum and, at the study counts this estimator is for, very noisy.

    Returns
    -------
    :obj:`numpy.ndarray`
        A scalar array under ``"pooled"``, one value per voxel under ``"voxelwise"``. Zero
        wherever the predictor does not vary, since there is nothing to correct with.
    """
    if mode not in COEFFICIENT_MODES:
        raise ValueError(f"mode must be one of {COEFFICIENT_MODES}; got {mode!r}.")
    if n_images < 2:
        raise ValueError(
            "Estimating lambda needs at least two image studies, because it is a covariance. "
            "With one image, pass a coefficient measured on external data instead."
        )

    centred_y = values - values.mean(axis=0, keepdims=True)
    centred_f = predictions - predictions.mean(axis=0, keepdims=True)
    shrink = 1.0 + n_images / n_coordinates

    if mode == "pooled":
        var_f = float((centred_f**2).sum())
        if var_f <= 0:
            return np.zeros(())
        return np.asarray(float((centred_y * centred_f).sum()) / (shrink * var_f))

    var_f = (centred_f**2).sum(axis=0)
    cov = (centred_y * centred_f).sum(axis=0)
    return np.divide(cov, shrink * var_f, out=np.zeros_like(cov), where=var_f > 0)


def control_variate_mean(
    values, predictions_image, predictions_coordinate, coefficient, n_coordinates=None
):
    r"""Correct the image mean with the coordinate prediction, and report what qualifies it.

    ``n_coordinates`` defaults to the number of rows in ``predictions_coordinate``. Pass it
    explicitly only when the predictor has been averaged elsewhere.

    Returns
    -------
    :obj:`dict`
        ``estimate``
            :math:`\hat m_\lambda`, one value per voxel.
        ``se``
            Its standard error from the two-cohort variance, ``inf`` where the image cohort is
            too small to supply one rather than zero, which would read as certainty.
        ``se_images_only``
            The same for :math:`\bar Y_I` alone, so the correction can be priced.
        ``variance_ratio``
            ``se**2 / se_images_only**2``. Below one where the correction helped.
        ``correlation``
            The image cohort's correlation between the effect and the prediction, which is what
            the achievable ratio depends on.
        ``cohort_shift``, ``cohort_shift_se``
            :math:`\bar f_C - \bar f_I` and its standard error. The exchangeability assumption
            says this is zero in expectation.
        ``valid``
            Where every quantity above is finite and the predictor varied.
    """
    values = np.atleast_2d(np.asarray(values, dtype=float))
    predictions_image = np.atleast_2d(np.asarray(predictions_image, dtype=float))
    predictions_coordinate = np.atleast_2d(np.asarray(predictions_coordinate, dtype=float))
    n_images = values.shape[0]
    if predictions_image.shape != values.shape:
        raise ValueError(
            f"predictions_image must match values in shape; got {predictions_image.shape} "
            f"against {values.shape}."
        )
    n_coord = int(n_coordinates if n_coordinates is not None else predictions_coordinate.shape[0])
    if n_coord < 1:
        raise ValueError("The correction needs at least one coordinate-only study.")

    lam = np.asarray(coefficient, dtype=float)
    mean_y = values.mean(axis=0)
    mean_f_image = predictions_image.mean(axis=0)
    mean_f_coord = predictions_coordinate.mean(axis=0)
    shift = mean_f_coord - mean_f_image
    estimate = mean_y + lam * shift

    # Variances from the image cohort. With one image there is no within-cohort spread to read,
    # so every variance is unavailable; say so with inf rather than 0.
    if n_images < 2:
        infinite = np.full(mean_y.shape, np.inf)
        return {
            "estimate": estimate,
            "se": infinite,
            "se_images_only": infinite.copy(),
            "variance_ratio": np.full(mean_y.shape, np.nan),
            "correlation": np.full(mean_y.shape, np.nan),
            "cohort_shift": shift,
            "cohort_shift_se": infinite.copy(),
            "valid": np.zeros(mean_y.shape, dtype=bool),
        }

    var_y = values.var(axis=0, ddof=1)
    var_f = predictions_image.var(axis=0, ddof=1)
    cov = ((values - mean_y) * (predictions_image - mean_f_image)).sum(axis=0) / (n_images - 1)

    var_corrected = var_y - 2.0 * lam * cov + lam**2 * var_f
    variance = np.clip(var_corrected, 0.0, None) / n_images + lam**2 * var_f / n_coord
    se = np.sqrt(variance)
    se_images = np.sqrt(var_y / n_images)

    denominator = np.sqrt(var_y * var_f)
    correlation = np.divide(
        cov, denominator, out=np.full(cov.shape, np.nan), where=denominator > 0
    )
    ratio = np.divide(
        variance, var_y / n_images, out=np.full(variance.shape, np.nan), where=var_y > 0
    )
    # The shift is a difference of two independent cohort means of the same quantity.
    shift_se = np.sqrt(var_f * (1.0 / n_images + 1.0 / n_coord))

    valid = np.isfinite(se) & (se > 0) & (var_f > 0) & (var_y > 0)
    return {
        "estimate": estimate,
        "se": np.where(valid, se, np.inf),
        "se_images_only": np.where(var_y > 0, se_images, np.inf),
        "variance_ratio": ratio,
        "correlation": correlation,
        "cohort_shift": shift,
        "cohort_shift_se": shift_se,
        "valid": valid,
    }


def achievable_ratio(correlation, n_images, n_coordinates):
    r"""Give the best variance ratio a predictor of this quality can reach.

    The ceiling is :math:`1 - \rho^2/(1 + n/N)`.

    Worth computing before fitting anything. It is the honest ceiling: the estimate cannot beat
    it, and as the coordinate cohort grows it stops improving at :math:`1 - \rho^2`.
    """
    rho = np.asarray(correlation, dtype=float)
    return 1.0 - rho**2 / (1.0 + n_images / n_coordinates)
