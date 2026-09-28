"""Non-independence: the same coordinate table reported in more than one place.

A meta-analysis assumes its analyses are independent. Identical foci sets
appearing under different studies break that.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, hashlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))
import numpy as np, pandas as pd
import nsnorm
from nimare.extract import fetch_neurostore
pd.set_option("display.width", 200)

ss = fetch_neurostore(version="nightly")
co = nsnorm.clean_coordinates(ss.coordinates)

def fingerprint(g):
    pts = np.round(g[["x", "y", "z"]].values.astype(float), 1)
    pts = pts[np.lexsort(pts.T[::-1])]
    return hashlib.md5(pts.tobytes()).hexdigest()

fp = co.groupby("contrast_id").apply(fingerprint).rename("fp").reset_index()
fp = fp.merge(co[["contrast_id", "study_id"]].drop_duplicates(), on="contrast_id")
n_foci = co.groupby("contrast_id").size().rename("n_foci")
fp = fp.merge(n_foci, on="contrast_id")

print("=" * 82)
print("IDENTICAL COORDINATE SETS")
print("=" * 82)
print(f"analyses: {len(fp):,}   distinct coordinate sets: {fp.fp.nunique():,}")

# Only count sets with enough foci to make coincidence implausible.
sized = fp[fp.n_foci >= 4]
grp = sized.groupby("fp").agg(n_analyses=("contrast_id", "size"),
                              n_studies=("study_id", "nunique"),
                              n_foci=("n_foci", "first"))
rep = grp[grp.n_analyses > 1]
cross = grp[grp.n_studies > 1]
print(f"\namong the {len(sized):,} analyses reporting >=4 foci:")
print(f"  coordinate sets appearing more than once: {len(rep):,}")
print(f"  ... spanning more than one study:         {len(cross):,}")
print(f"  analyses involved in a cross-study repeat:{int(cross.n_analyses.sum()):,} "
      f"({cross.n_analyses.sum()/len(sized):.2%})")
print("\nlargest cross-study repeats (foci set size x how many studies):")
print(cross.sort_values(["n_studies", "n_foci"], ascending=False).head(10).to_string())

print("\n" + "=" * 82)
print("WITHIN-STUDY REPEATS (the same table entered as several analyses)")
print("=" * 82)
within = sized.groupby(["study_id", "fp"]).size()
within = within[within > 1]
print(f"  study/coordinate-set pairs repeated: {len(within):,}")
print(f"  analyses involved:                   {int(within.sum()):,}")

print("\n" + "=" * 82)
print("EXACT DUPLICATE FOCI WITHIN ONE ANALYSIS (before cleaning)")
print("=" * 82)
raw = ss.coordinates
dup = raw.duplicated(subset=["contrast_id", "x", "y", "z"], keep="first")
print(f"  {int(dup.sum()):,} repeated foci in {raw[dup].contrast_id.nunique():,} analyses")
