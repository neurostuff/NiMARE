"""Study A: how well do reported coordinates predict cognitive domain,
and how does that depend on the spatial granularity of the atlas?"""
import warnings; warnings.filterwarnings("ignore")
import sys, json, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))
import numpy as np, pandas as pd
import cohort, nsnorm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

W = cohort.W
SEED = 13
MAX_ROWS = 20000

df = cohort.load_cohort()
mask = cohort.healthy_task_mask(df).values
print(f"healthy task-fMRI analyses with a domain: {mask.sum():,} "
      f"from {df.loc[mask,'study_id'].nunique():,} studies", flush=True)

# Subsample whole studies so the sweep is affordable and stays grouped.
rng = np.random.default_rng(SEED)
studies = df.loc[mask, "study_id"].unique()
idx = np.flatnonzero(mask)
if idx.size > MAX_ROWS:
    rng.shuffle(studies)
    take, chosen = 0, []
    for s in studies:
        chosen.append(s)
        take += int((df.study_id == s).values[idx].sum())
        if take >= MAX_ROWS:
            break
    idx = idx[np.isin(df.study_id.values[idx], chosen)]
sub = df.iloc[idx]
groups = sub.study_id.values
print(f"sweep sample: {len(idx):,} analyses from {sub.study_id.nunique():,} studies", flush=True)
for d in nsnorm.DOMAINS:
    print(f"  {d:<32} {int(sub['dom_'+d].sum()):>6,}  ({sub['dom_'+d].mean():.1%})")

meta = json.load(open(f"{W}/ops/meta.json"))
order = sorted(meta, key=lambda k: meta[k]["n_regions"])

def evaluate(X, y, groups):
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    )
    return cross_val_score(pipe, X, y, cv=GroupKFold(5), groups=groups,
                           scoring="roc_auc", n_jobs=1)

records = []
# Baseline: how much of the signal is just "how many foci were reported"?
n_peaks = sub.n_peaks.values.reshape(-1, 1).astype(float)
for d in nsnorm.DOMAINS:
    y = sub["dom_" + d].values.astype(int)
    s = evaluate(n_peaks, y, groups)
    records.append({"atlas": "n_peaks only", "n_regions": 1, "domain": d,
                    "auc": s.mean(), "sd": s.std()})
    print(f"[baseline] {d:<32} AUC {s.mean():.3f}", flush=True)

for name in order:
    X = cohort.load_features(name, idx)
    t0 = time.time()
    for d in nsnorm.DOMAINS:
        y = sub["dom_" + d].values.astype(int)
        s = evaluate(X, y, groups)
        records.append({"atlas": name, "n_regions": meta[name]["n_regions"],
                        "domain": d, "auc": s.mean(), "sd": s.std()})
    mean_auc = np.mean([r["auc"] for r in records if r["atlas"] == name])
    print(f"{name:<14} ({meta[name]['n_regions']:>3} regions)  mean AUC {mean_auc:.3f}"
          f"   [{time.time()-t0:.0f}s]", flush=True)
    del X

res = pd.DataFrame(records)
res.to_parquet(f"{W}/results_domain_sweep.parquet")
print("\n" + "=" * 80)
print(res.pivot_table(index="domain", columns="atlas", values="auc").round(3).to_string())
