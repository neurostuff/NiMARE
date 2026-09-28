import warnings; warnings.filterwarnings("ignore")
import sys, time, pickle; from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))

from cohort import WORK  # noqa: E402
import numpy as np
from scipy import sparse
from nimare.extract import fetch_neurostore
from nimare.meta.kernel import MKDAKernel
from nimare.ml import MAKernel
from fastreduce import atlas_operator
from atlases import ATLASES

ss = fetch_neurostore(version="nightly")
b = ss.to_bunch()
ref = MAKernel(MKDAKernel(r=10), source_masker=b.masker).fit_transform(b.data[:4, b.voxel_columns])

ops = {}
for name, fetch in ATLASES.items():
    t = time.time()
    try:
        atlas = fetch()
        op, names = atlas_operator(atlas, b.masker, ref)
        dens = "sparse" if sparse.issparse(op) else "dense"
        ops[name] = (op, names)
        print(f"{name:<14} {op.shape[1]:>4} regions  [{dens}]  {time.time()-t:6.1f}s  "
              f"names={'yes' if names is not None else 'no'}", flush=True)
    except Exception as e:
        print(f"{name:<14} FAILED: {type(e).__name__}: {str(e)[:110]}", flush=True)

with open("{WORK}/operators.pkl", "wb") as fh:
    pickle.dump(ops, fh)
print(f"\nsaved {len(ops)} operators")
