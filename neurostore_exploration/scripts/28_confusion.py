"""Redo the domain confusion with scale-free scores.

One-vs-rest decision functions are not on a common scale, so an argmax over
them favours whichever model has the widest spread. Converting each model's
out-of-fold scores to within-model percentile ranks removes that.
"""
import warnings; warnings.filterwarnings("ignore")
import sys; from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))

from cohort import WORK  # noqa: E402
import numpy as np, pandas as pd
import cohort, nsnorm
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
pd.set_option("display.width", 220)

W, SEED, ATLAS = cohort.W, 13, "DiFuMo-256"
df = cohort.load_cohort()
idx = np.flatnonzero(cohort.healthy_task_mask(df).values)
sub = df.iloc[idx]
X = cohort.load_features(ATLAS, idx)
groups = sub.study_id.values

scores = {}
for d in nsnorm.DOMAINS:
    y = sub["dom_" + d].values.astype(int)
    scores[d] = cross_val_predict(
        make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=3000, class_weight="balanced",
                                         random_state=SEED)),
        X, y, cv=GroupKFold(5), groups=groups, method="decision_function")
    print(f"scored {d}", flush=True)

S = np.column_stack([rankdata(scores[d]) / len(idx) for d in nsnorm.DOMAINS])
single = sub.n_domains.values == 1
labels = np.column_stack([sub["dom_" + d].values for d in nsnorm.DOMAINS])[single]
true = labels.argmax(axis=1)
pred = S[single].argmax(axis=1)

conf = pd.crosstab(pd.Series([nsnorm.DOMAINS[i] for i in true], name="labelled"),
                   pd.Series([nsnorm.DOMAINS[i] for i in pred], name="top-ranked"),
                   normalize="index").reindex(index=nsnorm.DOMAINS,
                                              columns=nsnorm.DOMAINS)
print(f"\nsingle-domain analyses: {int(single.sum()):,}")
print("row = the label the extractor gave; column = the highest-ranked model (%)")
print((conf * 100).round(1).to_string())
print(f"\ntop-1 agreement: {(true == pred).mean():.1%}  (chance {1/10:.1%})")
print("\nper-domain recall:")
rec = pd.Series(np.diag(conf.values), index=nsnorm.DOMAINS).sort_values(ascending=False)
print((rec * 100).round(1).to_string())
print("\nsingle-domain analyses available per domain:")
print(pd.Series(true).value_counts().rename(lambda i: nsnorm.DOMAINS[i]).to_string())
conf.to_parquet(f"{W}/results_confusion.parquet")
