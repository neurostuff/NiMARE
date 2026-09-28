import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from nimare.extract import fetch_neurostore
pd.set_option("display.width", 200)
ss = fetch_neurostore(version="nightly")
co = ss.coordinates

box = (co.x.abs() > 90) | (co.y < -130) | (co.y > 90) | (co.z < -80) | (co.z > 100)
bad = co[box]
print(f"{len(bad):,} implausible foci in {bad.contrast_id.nunique():,} analyses\n")

# Per-analysis: is the WHOLE analysis bad (systematic) or just a stray focus (typo)?
n_bad = bad.groupby("contrast_id").size()
n_all = co.groupby("contrast_id").size()
frac_bad = (n_bad / n_all.reindex(n_bad.index)).sort_values()
print("fraction of an affected analysis's foci that are implausible:")
print(frac_bad.describe(percentiles=[.1,.25,.5,.75,.9]).to_string())
print(f"analyses where ALL foci implausible: {(frac_bad==1).sum():,} "
      f"({(frac_bad==1).mean():.0%} of affected)")

# Hypothesis A: voxel indices (all positive, 0..~190 range) rather than mm
whole = frac_bad[frac_bad == 1].index
w = co[co.contrast_id.isin(whole)]
per = w.groupby("contrast_id")[["x","y","z"]].agg(["min","max"])
allpos = w.groupby("contrast_id").apply(lambda d: (d[["x","y","z"]] >= 0).all().all())
inrange = w.groupby("contrast_id").apply(
    lambda d: bool(((d[["x","y","z"]] >= 0) & (d[["x","y","z"]] <= 200)).all().all()))
print(f"\nfully-implausible analyses: {len(whole):,}")
print(f"  all coords non-negative:        {int(allpos.sum()):,}")
print(f"  all coords within [0,200] too:  {int(inrange.sum()):,}  <-- looks like VOXEL INDICES")

print("\nexample of a voxel-index-looking analysis:")
ex = inrange[inrange].index[0]
print(co[co.contrast_id == ex][["x","y","z"]].head(6).to_string(index=False))

# Hypothesis B: magnitude blow-ups (digit concatenation / unit errors)
huge = co[(co.x.abs() > 500) | (co.y.abs() > 500) | (co.z.abs() > 500)]
print(f"\nextreme (|coord| > 500 mm): {len(huge):,} foci in {huge.contrast_id.nunique():,} analyses")
print(huge[["x","y","z"]].head(8).to_string(index=False))

# How much of the corpus would a simple in-brain filter drop?
print("\n" + "="*70)
print("IMPACT OF FILTERING")
print("="*70)
ok = ~box
keep_an = co[ok].contrast_id.nunique()
print(f"foci kept: {ok.sum():,} / {len(co):,} ({ok.mean():.3%})")
print(f"analyses retaining >=1 focus: {keep_an:,} / {co.contrast_id.nunique():,}")
