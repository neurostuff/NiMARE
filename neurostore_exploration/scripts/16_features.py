import warnings; warnings.filterwarnings("ignore")
import sys, os, time, json, gc
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))

from cohort import WORK  # noqa: E402
import numpy as np, pandas as pd
from scipy import sparse
import nsnorm
from nimare.extract import fetch_neurostore
from nimare.meta.kernel import MKDAKernel
from nimare.ml import MAKernel, clear_map_cache

W = WORK
os.makedirs(f"{W}/feat", exist_ok=True)

ss = fetch_neurostore(version="nightly")
b = ss.to_bunch()
ids = np.asarray(b.ids)
print(f"bunch rows: {len(ids):,}", flush=True)

# ---- QC: drop analyses reporting voxel indices as millimetres -------------
_, per = nsnorm.flag_coordinates(ss.coordinates)
suspect = set(per.index[per.voxel_indices])
short = np.asarray([i.split("-", 1)[1] if "-" in i else i for i in ids])
keep = ~np.isin(short, list(suspect))
n_peaks = np.asarray(b.data[:, b.voxel_columns].sum(axis=1)).ravel()
keep &= n_peaks > 0
print(f"QC: dropped {int((~keep).sum()):,} rows "
      f"({len(suspect):,} voxel-index analyses + empties); keeping {int(keep.sum()):,}", flush=True)

rows = np.flatnonzero(keep)
np.save(f"{W}/feat/rows.npy", rows)
pd.DataFrame({"contrast_id": short[rows], "full_id": ids[rows],
              "study_id": np.asarray(b.groups)[rows],
              "n_peaks": n_peaks[rows]}).to_parquet(f"{W}/feat/rows.parquet")

X = b.data[:, b.voxel_columns][rows]
kernel = MAKernel(MKDAKernel(r=10), source_masker=b.masker)
kernel.fit(X[:2])

meta = json.load(open(f"{W}/ops/meta.json"))
CHUNK = 8192
for name, info in sorted(meta.items(), key=lambda kv: kv[1]["n_regions"]):
    out = f"{W}/feat/{name}.npy"
    if os.path.exists(out):
        print(f"{name}: cached", flush=True); continue
    t0 = time.time()
    op = (sparse.load_npz(f"{W}/ops/{name}.npz") if info["kind"] == "sparse"
          else np.load(f"{W}/ops/{name}.npy"))
    feats = np.empty((X.shape[0], info["n_regions"]), dtype=np.float32)
    for start in range(0, X.shape[0], CHUNK):
        stop = min(start + CHUNK, X.shape[0])
        maps = kernel.transform(X[start:stop])
        red = maps @ op
        feats[start:stop] = np.asarray(red.todense() if sparse.issparse(red) else red,
                                       dtype=np.float32)
        del maps, red
    np.save(out, feats)
    print(f"{name:<14} {feats.shape} in {time.time()-t0:6.1f}s  "
          f"({feats.nbytes/1e6:.0f} MB, {(feats!=0).mean():.1%} nonzero)", flush=True)
    del op, feats; gc.collect()
clear_map_cache()
print("features done")
