"""Does the age of the participants show up in where a study reports peaks?

A continuous target, so a regressor rather than a classifier, and restricted to
healthy task fMRI so it is not reading diagnosis instead of age.
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))

from cohort import WORK  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import GroupKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import cohort  # noqa: E402

SEED, ATLAS = 13, "DiFuMo-256"
df = cohort.load_cohort()
mask = cohort.healthy_task_mask(df).values & df.age.notna().values
idx = np.flatnonzero(mask)
sub = df.iloc[idx]
X = cohort.load_features(ATLAS, idx)
y = sub.age.values
groups = sub.study_id.values
print(f"healthy task-fMRI analyses with a mean age: {len(idx):,} "
      f"from {sub.study_id.nunique():,} studies", flush=True)
print(f"age: median {np.median(y):.1f}, IQR {np.percentile(y, 25):.1f}-"
      f"{np.percentile(y, 75):.1f}, range {y.min():.1f}-{y.max():.1f}\n", flush=True)

models = {
    "mean (baseline)": DummyRegressor(strategy="mean"),
    "focus count": "peaks",
    "ridge on regions": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 12))),
    "gradient boosting": HistGradientBoostingRegressor(max_iter=300, random_state=SEED),
}
cv = GroupKFold(5)
rows = []
for name, model in models.items():
    features = sub.n_peaks.values.reshape(-1, 1).astype(float) if model == "peaks" else X
    est = make_pipeline(StandardScaler(),
                        RidgeCV(alphas=np.logspace(-1, 4, 12))) if model == "peaks" else model
    pred = cross_val_predict(est, features, y, cv=cv, groups=groups)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    rows.append({"model": name, "r2": 1 - ss_res / ss_tot,
                 "mae": float(np.mean(np.abs(y - pred))),
                 "r": float(np.corrcoef(y, pred)[0, 1]) if pred.std() > 0 else 0.0})
    print(f"{name:<20} R2 {rows[-1]['r2']:+.3f}   MAE {rows[-1]['mae']:.1f} yr   "
          f"r {rows[-1]['r']:+.3f}", flush=True)

res = pd.DataFrame(rows)
res.to_parquet(f"{WORK}/results_age.parquet")

# Which regions carry it?
best = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 12))).fit(X, y)
import json
meta = json.load(open(f"{WORK}/ops/meta.json"))
names = meta[ATLAS]["names"]
w = pd.Series(best[-1].coef_, index=names)
print("\nregions weighted toward OLDER samples:")
print(w.sort_values(ascending=False).head(8).round(3).to_string())
print("\nregions weighted toward YOUNGER samples:")
print(w.sort_values().head(8).round(3).to_string())
