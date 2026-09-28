"""Study B: is there a 'clinical' signature in reported coordinates, or is it
study-design confound? And does it transfer across diagnoses?"""
import warnings; warnings.filterwarnings("ignore")
import sys, time
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
from sklearn.metrics import roc_auc_score

W, SEED, ATLAS = cohort.W, 13, "DiFuMo-256"
df = cohort.load_cohort()
X_all = np.load(f"{W}/feat/{ATLAS}.npy", mmap_mode="r")

def pipe():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
    )

def auc(mask, y=None, feats=None, label=""):
    idx = np.flatnonzero(mask)
    sub = df.iloc[idx]
    target = (sub.group == "patients").astype(int).values if y is None else y
    X = np.asarray(X_all[idx]) if feats is None else feats
    s = cross_val_score(pipe(), X, target, cv=GroupKFold(5),
                        groups=sub.study_id.values, scoring="roc_auc")
    print(f"{label:<52} n={len(idx):>6,}  patients={target.mean():>5.1%}  "
          f"AUC {s.mean():.3f} +/- {s.std():.3f}", flush=True)
    return s.mean(), len(idx)

print("=" * 96)
print("B. THE CONFOUND LADDER: patients vs healthy, tightening the sample")
print("=" * 96)
has_grp = df.group.isin(["healthy", "patients"])
rows = []
rows.append(("1. any modality, any design", *auc(has_grp, label="1. any modality, any design")))
m2 = has_grp & (df.modality == "fMRI-BOLD")
rows.append(("2. + fMRI-BOLD only", *auc(m2, label="2. + fMRI-BOLD only")))
m3 = m2 & (df.resting == 0.0)
rows.append(("3. + task only (no resting state)", *auc(m3, label="3. + task only (no resting state)")))
m4 = m3 & (df.n_domains > 0)
rows.append(("4. + has a cognitive domain label", *auc(m4, label="4. + has a cognitive domain label")))

print("\n-- design-only baselines on sample 4 (no coordinates at all) --")
idx4 = np.flatnonzero(m4.values)
sub4 = df.iloc[idx4]
design_feats = sub4[["dom_" + d for d in nsnorm.DOMAINS]].values.astype(float)
design_feats = np.column_stack([design_feats, sub4.n_peaks.values])
auc(m4, feats=design_feats, label="   domain labels + focus count ONLY")
auc(m4, feats=sub4.n_peaks.values.reshape(-1, 1).astype(float),
    label="   focus count ONLY")

print("\n" + "=" * 96)
print("B2. WITHIN EACH COGNITIVE DOMAIN (task fMRI only)")
print("=" * 96)
within = []
for d in nsnorm.DOMAINS:
    m = m3 & df["dom_" + d]
    if m.sum() < 800 or df.loc[m, "study_id"].nunique() < 40:
        continue
    a, n = auc(m, label=f"   {d}")
    within.append({"domain": d, "auc": a, "n": n})
pd.DataFrame(within).to_parquet(f"{W}/results_clinical_within_domain.parquet")

print("\n" + "=" * 96)
print("B3. CROSS-DIAGNOSIS TRANSFER (train on one disorder, test on another)")
print("=" * 96)
base = m3.values
healthy = np.flatnonzero(base & (df.group == "healthy").values)
counts = df.iloc[np.flatnonzero(base & (df.group == "patients").values)].dx.value_counts()
dxs = [d for d, c in counts.items() if c >= 500 and d not in ("Other", "Healthy")]
print(f"diagnoses with >=500 task-fMRI patient analyses: {dxs}\n", flush=True)

rng = np.random.default_rng(SEED)
models, holdout = {}, {}
for d in dxs:
    pat = np.flatnonzero(base & (df.dx == d).values & (df.group == "patients").values)
    # Match the healthy comparison set in size, drawn once and shared.
    ctl = rng.choice(healthy, size=min(len(pat) * 2, len(healthy)), replace=False)
    idx = np.concatenate([pat, ctl])
    y = np.concatenate([np.ones(len(pat)), np.zeros(len(ctl))])
    st = df.study_id.values[idx]
    # Hold out a quarter of studies so transfer is tested out-of-sample.
    ustudies = np.unique(st); rng.shuffle(ustudies)
    test_st = set(ustudies[: max(1, len(ustudies) // 4)])
    is_test = np.array([s in test_st for s in st])
    X = np.asarray(X_all[idx])
    model = pipe().fit(X[~is_test], y[~is_test])
    models[d] = model
    holdout[d] = (X[is_test], y[is_test], len(pat))
    print(f"  {d:<28} patients={len(pat):>5,}  held-out rows={is_test.sum():>5,}", flush=True)

mat = pd.DataFrame(index=dxs, columns=dxs, dtype=float)
for tr in dxs:
    for te in dxs:
        Xte, yte, _ = holdout[te]
        if len(np.unique(yte)) < 2:
            continue
        mat.loc[tr, te] = roc_auc_score(yte, models[tr].decision_function(Xte))
mat.to_parquet(f"{W}/results_transfer.parquet")
print("\nrows = trained on, columns = tested on (held-out studies), AUC:")
print(mat.round(3).to_string())
print(f"\nmean diagonal (within-disorder):  {np.nanmean(np.diag(mat.values)):.3f}")
off = mat.values.copy(); np.fill_diagonal(off, np.nan)
print(f"mean off-diagonal (cross-disorder): {np.nanmean(off):.3f}")
pd.DataFrame(rows, columns=["step", "auc", "n"]).to_parquet(f"{W}/results_clinical_ladder.parquet")
