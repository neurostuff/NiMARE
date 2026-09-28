"""Study A3: decoding tops out around AUC 0.63. Which ingredient is the limit --
the atlas, the linear model, the label noise, or the coordinates themselves?"""
import warnings; warnings.filterwarnings("ignore")
import sys, time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))
import numpy as np, pandas as pd
import cohort, nsnorm
from nimare.extract import fetch_neurostore
from nimare.meta.kernel import MKDAKernel
from nimare.ml import MAKernel
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

W, SEED, ATLAS = cohort.W, 13, "DiFuMo-256"
MAX_ROWS = 20000
df = cohort.load_cohort()
mask = cohort.healthy_task_mask(df).values

rng = np.random.default_rng(SEED)
idx_all = np.flatnonzero(mask)
studies = df.study_id.values[idx_all]
uniq = np.unique(studies); rng.shuffle(uniq)
take, chosen, counts = 0, [], pd.Series(studies).value_counts()
for s in uniq:
    chosen.append(s); take += counts[s]
    if take >= MAX_ROWS: break
idx = idx_all[np.isin(studies, chosen)]
sub = df.iloc[idx]; groups = sub.study_id.values
print(f"sample: {len(idx):,} analyses / {sub.study_id.nunique():,} studies", flush=True)

Xa = cohort.load_features(ATLAS, idx)

print("\nbuilding MA maps for the unreduced comparison...", flush=True)
ss = fetch_neurostore(version="nightly")
bunch = ss.to_bunch()
rows = np.load(f"{W}/feat/rows.npy")
peaks = bunch.data[:, bunch.voxel_columns][rows][idx]
maps = MAKernel(MKDAKernel(r=10), source_masker=bunch.masker).fit_transform(peaks)
print(f"maps: {maps.shape} nnz={maps.nnz:,}", flush=True)

def score(X, y, est, cv_groups):
    s = cross_val_score(est, X, y, cv=GroupKFold(5), groups=cv_groups, scoring="roc_auc")
    return s.mean(), s.std()


def score_once(X, y, est, cv_groups):
    """One grouped holdout, for arms too expensive to fit five times.

    Refitting a truncated SVD of the whole voxel matrix once per fold costs far
    more than the comparison is worth, so this arm gets a single split, and the
    atlas arm is repeated on that same split for a fair comparison.
    """
    s = cross_val_score(est, X, y,
                        cv=GroupShuffleSplit(1, test_size=0.25, random_state=SEED),
                        groups=cv_groups, scoring="roc_auc")
    return s.mean(), s.std()

lin = lambda: make_pipeline(StandardScaler(),
    LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED))

records = []
single = sub.n_domains.values == 1
print(f"\nsingle-domain analyses in sample: {int(single.sum()):,}\n", flush=True)

for d in nsnorm.DOMAINS:
    y = sub["dom_" + d].values.astype(int)
    if y.sum() < 100: continue
    row = {"domain": d, "n_pos": int(y.sum())}

    row["atlas_linear"] = score(Xa, y, lin(), groups)[0]
    row["atlas_gbm"] = score(
        Xa, y,
        HistGradientBoostingClassifier(max_iter=200, class_weight="balanced",
                                       random_state=SEED), groups)[0]
    row["svd128_voxels"] = score_once(
        maps, y,
        make_pipeline(TruncatedSVD(128, random_state=SEED), StandardScaler(),
                      LogisticRegression(max_iter=3000, class_weight="balanced",
                                         random_state=SEED)), groups)[0]
    row["atlas_same_split"] = score_once(Xa, y, lin(), groups)[0]
    # Label noise: keep only analyses the extractor gave exactly one domain.
    ys, Xs, gs = y[single], Xa[single], groups[single]
    row["atlas_linear_singleonly"] = score(Xs, ys, lin(), gs)[0] if ys.sum() >= 40 else np.nan
    records.append(row)
    print(f"{d:<30} atlas {row['atlas_linear']:.3f} | gbm {row['atlas_gbm']:.3f} | "
          f"svd128 {row['svd128_voxels']:.3f} vs atlas {row['atlas_same_split']:.3f} "
          f"(same split) | single-label {row['atlas_linear_singleonly']:.3f}",
          flush=True)

res = pd.DataFrame(records)
res.to_parquet(f"{W}/results_ceiling.parquet")
print("\n" + "=" * 86)
print(res.set_index("domain").round(3).to_string())
print("\nmeans:")
print(res[["atlas_linear", "atlas_gbm", "svd128_voxels", "atlas_same_split",
           "atlas_linear_singleonly"]].mean().round(3).to_string())
