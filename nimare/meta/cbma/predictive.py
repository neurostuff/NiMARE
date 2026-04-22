"""Predictive cutoff helpers for ALE family-wise error correction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, skew


class PredictiveCutoffError(ValueError):
    """Raised when predictive cutoff inputs are unsupported or out of distribution."""


def _load_xgboost():
    """Import xgboost lazily so ALE predictive correction remains optional."""
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "ALE predictive cutoff correction requires the optional dependency "
            "'xgboost'. Install it to use FWECorrector(method='predictive')."
        ) from exc
    return xgb


def _safe_stat(func, values):
    """Evaluate a univariate statistic and coerce NaNs from degenerate inputs to zero."""
    value = func(values)
    if np.isnan(value):
        return 0.0
    return float(value)


def feature_extraction(nexp, nsub, nfoci):
    """Extract the feature vector used by the JALE cutoff-prediction models."""
    nsub = np.asarray(nsub, dtype=float)
    nfoci = np.asarray(nfoci, dtype=float)

    if nsub.ndim != 1 or nfoci.ndim != 1 or nsub.shape[0] != nfoci.shape[0]:
        raise PredictiveCutoffError(
            "Predictive cutoff features require one-dimensional subject and focus counts "
            "with matching lengths."
        )
    if nsub.shape[0] != nexp:
        raise PredictiveCutoffError(
            f"Expected {nexp} experiments, but received {nsub.shape[0]} subject counts."
        )
    if np.any(nsub <= 0):
        raise PredictiveCutoffError(
            "Predictive cutoff requires strictly positive subject counts for every experiment."
        )
    if np.any(nfoci < 0):
        raise PredictiveCutoffError(
            "Predictive cutoff requires nonnegative focus counts for every experiment."
        )
    if nexp > 150:
        raise PredictiveCutoffError(
            "Predictive cutoff is out of the JALE model's supported range: "
            f"n_experiments={nexp} exceeds 150."
        )
    if np.max(nsub) > 300:
        raise PredictiveCutoffError(
            "Predictive cutoff is out of the JALE model's supported range: "
            f"max subjects per experiment ({np.max(nsub):.0f}) exceeds 300."
        )
    if np.max(nfoci) > 150:
        raise PredictiveCutoffError(
            "Predictive cutoff is out of the JALE model's supported range: "
            f"max foci per experiment ({np.max(nfoci):.0f}) exceeds 150."
        )

    ratios = nfoci / nsub
    hi_foci = np.sum(nfoci[nsub > 20])
    mi_foci = np.sum(nfoci[(nsub > 15) & (nsub <= 20)])
    li_foci = np.sum(nfoci[(nsub > 10) & (nsub <= 15)])
    vi_foci = np.sum(nfoci[nsub <= 10])

    features = np.c_[
        nexp,
        np.sum(nsub),
        np.mean(nsub),
        np.median(nsub),
        np.std(nsub),
        np.max(nsub),
        np.min(nsub),
        _safe_stat(skew, nsub),
        _safe_stat(kurtosis, nsub),
        np.sum(nfoci),
        np.mean(nfoci),
        np.median(nfoci),
        np.std(nfoci),
        np.max(nfoci),
        np.min(nfoci),
        _safe_stat(skew, nfoci),
        _safe_stat(kurtosis, nfoci),
        np.mean(ratios),
        np.std(ratios),
        np.max(ratios),
        np.min(ratios),
        np.sum(nfoci) / nexp,
        hi_foci,
        mi_foci,
        li_foci,
        vi_foci,
    ]
    return features.astype(float, copy=False)


def _model_path(model_name):
    """Return the packaged model path for one predictive-cutoff regressor."""
    return Path(__file__).resolve().parents[2] / "resources" / "ml_models" / model_name


def predict_cutoffs(nexp, nsub, nfoci, include_tfce=False):
    """Predict vFWE and cFWE cutoffs from ALE dataset metadata."""
    xgb = _load_xgboost()
    features = feature_extraction(nexp, nsub, nfoci)
    dmatrix = xgb.DMatrix(features)

    model_names = {
        "vfwe": "vFWE_model.xgb",
        "cfwe": "cFWE_model.xgb",
    }
    if include_tfce:
        model_names["tfce"] = "tfce_model.xgb"

    cutoffs = {}
    for key, model_name in model_names.items():
        model = xgb.Booster()
        model.load_model(_model_path(model_name))
        predicted = float(np.ravel(model.predict(dmatrix))[0])
        cutoffs[key] = round(predicted) if key == "cfwe" else predicted

    return cutoffs
