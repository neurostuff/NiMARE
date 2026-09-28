"""At what spatial scale does the information in reported peaks live?

The atlas sweep varied how finely the brain is divided. This varies the other
spatial scale: how far a reported peak is allowed to spread before it is read.
"""
import warnings; warnings.filterwarnings("ignore")
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))

from cohort import WORK  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import cohort  # noqa: E402
import nsnorm  # noqa: E402
from nimare.extract import fetch_neurostore  # noqa: E402
from nimare.meta.kernel import ALEKernel, KDAKernel, MKDAKernel  # noqa: E402
from nimare.ml import MAKernel, clear_map_cache  # noqa: E402

SEED, ATLAS, MAX_ROWS = 13, "DiFuMo-256", 12000

df = cohort.load_cohort()
idx_all = np.flatnonzero(cohort.healthy_task_mask(df).values)
rng = np.random.default_rng(SEED)
studies = df.study_id.values[idx_all]
counts = pd.Series(studies).value_counts()
uniq = np.unique(studies); rng.shuffle(uniq)
take, chosen = 0, []
for s in uniq:
    chosen.append(s); take += counts[s]
    if take >= MAX_ROWS:
        break
idx = idx_all[np.isin(studies, chosen)]
sub = df.iloc[idx]
groups = sub.study_id.values
print(f"sample: {len(idx):,} analyses / {sub.study_id.nunique():,} studies", flush=True)

ss = fetch_neurostore(version="nightly")
bunch = ss.to_bunch()
rows = np.load(f"{WORK}/feat/rows.npy")
peaks = bunch.data[:, bunch.voxel_columns][rows][idx]

operator = np.load(f"{WORK}/ops/{ATLAS}.npy")

KERNELS = [("MKDA r=5", MKDAKernel(r=5)), ("MKDA r=10", MKDAKernel(r=10)),
           ("MKDA r=15", MKDAKernel(r=15)), ("MKDA r=20", MKDAKernel(r=20)),
           ("MKDA r=30", MKDAKernel(r=30)), ("KDA r=10", KDAKernel(r=10)),
           ("ALE (sample-size)", ALEKernel())]

records = []
for name, kernel in KERNELS:
    t0 = time.time()
    try:
        maps = MAKernel(kernel, source_masker=bunch.masker).fit_transform(peaks)
    except Exception as exc:
        print(f"{name:<20} skipped: {type(exc).__name__}: {str(exc)[:70]}", flush=True)
        continue
    X = np.asarray(maps @ operator, dtype=np.float32)
    density = float((X != 0).mean())
    for domain in nsnorm.DOMAINS:
        y = sub["dom_" + domain].values.astype(int)
        if y.sum() < 100:
            continue
        s = cross_val_score(
            make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=3000, class_weight="balanced",
                                             random_state=SEED)),
            X, y, cv=GroupKFold(5), groups=groups, scoring="roc_auc")
        records.append({"kernel": name, "domain": domain, "auc": s.mean(),
                        "density": density})
    mean_auc = np.mean([r["auc"] for r in records if r["kernel"] == name])
    print(f"{name:<20} mean AUC {mean_auc:.3f}  "
          f"(nnz/row {maps.nnz / maps.shape[0]:.0f}, {time.time() - t0:.0f}s)", flush=True)
    del maps, X
    clear_map_cache()

res = pd.DataFrame(records)
res.to_parquet(f"{WORK}/results_kernel_scale.parquet")
print("\n" + "=" * 86)
print(res.pivot_table(index="domain", columns="kernel", values="auc").round(3).to_string())
