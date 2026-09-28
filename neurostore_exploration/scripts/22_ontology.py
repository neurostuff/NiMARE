"""Study A2: read the fitted domain models back as brain maps and ask what
similarity structure the ten domains have when defined by coordinates alone."""
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))
import numpy as np, pandas as pd
import cohort, nsnorm
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

W, SEED, ATLAS = cohort.W, 13, "DiFuMo-256"
meta = json.load(open(f"{W}/ops/meta.json"))

def _unique(labels):
    """Atlas region names repeat, so make them unique before they index a frame."""
    seen, out = {}, []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        out.append(label if seen[label] == 1 else f"{label} #{seen[label]}")
    return out

names = _unique(meta[ATLAS]["names"])

df = cohort.load_cohort()
mask = cohort.healthy_task_mask(df).values
idx = np.flatnonzero(mask)
sub = df.iloc[idx]
X = cohort.load_features(ATLAS, idx)
groups = sub.study_id.values
print(f"full healthy task-fMRI sample: {X.shape[0]:,} analyses, "
      f"{sub.study_id.nunique():,} studies, {X.shape[1]} regions", flush=True)

def pipe():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, class_weight="balanced",
                                            random_state=SEED))

weights, aucs, oof = {}, {}, {}
from sklearn.metrics import roc_auc_score
for d in nsnorm.DOMAINS:
    y = sub["dom_" + d].values.astype(int)
    p = cross_val_predict(pipe(), X, y, cv=GroupKFold(5), groups=groups,
                          method="decision_function")
    aucs[d] = roc_auc_score(y, p)
    oof[d] = p
    model = pipe().fit(X, y)
    weights[d] = model[-1].coef_.ravel()
    print(f"{d:<32} AUC {aucs[d]:.3f}   (n_pos={y.sum():,})", flush=True)

Wm = pd.DataFrame(weights, index=names).T
Wm.to_parquet(f"{W}/results_domain_weights.parquet")
pd.Series(aucs).to_frame("auc").to_parquet(f"{W}/results_domain_auc_full.parquet")

print("\n" + "=" * 90)
print("Similarity of the ten domains, as their coordinate signatures define them")
print("(Pearson r between one-vs-rest weight maps)")
print("=" * 90)
C = Wm.T.corr()
print(C.round(2).to_string())
C.to_parquet(f"{W}/results_domain_similarity.parquet")

d = 1 - C.values
np.fill_diagonal(d, 0.0)
Z = linkage(squareform(d, checks=False), method="average")
dn = dendrogram(Z, labels=list(C.index), no_plot=True)
print("\nhierarchical ordering:", " | ".join(dn["ivl"]))

print("\n" + "=" * 90)
print("Top regions per domain (largest positive weights)")
print("=" * 90)
for dom in nsnorm.DOMAINS:
    top = Wm.loc[dom].sort_values(ascending=False).head(5)
    print(f"\n{dom}  (AUC {aucs[dom]:.3f})")
    for region, w in top.items():
        print(f"    {w:+.3f}  {region}")

print("\n" + "=" * 90)
print("FORWARD vs REVERSE INFERENCE")
print("how often a region is reported, against how diagnostic it is of a domain")
print("=" * 90)
# Forward: how often each region is touched at all. Reverse: how much the
# one-vs-rest models lean on it.
freq = (X > 0).mean(axis=0)
selectivity = np.abs(Wm.values).max(axis=0)
spread = np.abs(Wm.values).mean(axis=0)
fr = pd.DataFrame({"region": names, "reported_in": freq,
                   "peak_weight": selectivity, "mean_weight": spread})
r = np.corrcoef(fr.reported_in, fr.peak_weight)[0, 1]
print(f"corr(how often reported, how diagnostic) = {r:+.3f}")
print("\nMOST REPORTED regions (the usual suspects):")
print(fr.sort_values("reported_in", ascending=False).head(10)
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nMOST DIAGNOSTIC regions:")
print(fr.sort_values("peak_weight", ascending=False).head(10)
      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
fr.to_parquet(f"{W}/results_forward_reverse.parquet")

print("\n" + "=" * 90)
print("MULTI-LABEL CONFUSION: where do the domain models disagree with the labels?")
print("=" * 90)
# Among single-domain analyses, which domain does each model rank highest?
single = sub.n_domains.values == 1
scores = np.column_stack([oof[d] for d in nsnorm.DOMAINS])[single]
true = np.array([[sub["dom_" + d].values[single][i] for d in nsnorm.DOMAINS]
                 for i in range(single.sum())]).argmax(axis=1)
pred = scores.argmax(axis=1)
conf = pd.crosstab(pd.Series([nsnorm.DOMAINS[i] for i in true], name="labelled"),
                   pd.Series([nsnorm.DOMAINS[i] for i in pred], name="top-scoring"),
                   normalize="index")
conf = conf.reindex(index=nsnorm.DOMAINS, columns=nsnorm.DOMAINS).fillna(0)
print(f"single-domain analyses: {int(single.sum()):,}")
print((conf * 100).round(1).to_string())
conf.to_parquet(f"{W}/results_confusion.parquet")
print(f"\ntop-1 agreement: {(true == pred).mean():.1%} (chance {1/len(nsnorm.DOMAINS):.1%})")
