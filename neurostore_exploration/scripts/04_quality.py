import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from nimare.extract import fetch_neurostore
pd.set_option("display.width", 200)

ss = fetch_neurostore(version="nightly")
co = ss.coordinates
print("=" * 78); print("COORDINATE SPACE LABELS"); print("=" * 78)
print(co.space.value_counts(dropna=False).to_string())

print("\n-- foci per analysis --")
per = co.groupby("contrast_id").size()
print(per.describe(percentiles=[.05,.25,.5,.75,.95,.99]).to_string())
print(f"analyses with coordinates: {per.size:,} of {len(ss.ids):,} "
      f"({per.size/len(ss.ids):.1%})")
print(f"analyses with 1 focus: {(per==1).sum():,}   >100 foci: {(per>100).sum():,}")

print("\n-- coordinate value ranges (mm) --")
print(co[["x","y","z"]].describe(percentiles=[.001,.01,.5,.99,.999]).to_string())

# Out-of-brain / impossible coordinates
oob = (co.x.abs() > 90) | (co.y < -130) | (co.y > 90) | (co.z < -80) | (co.z > 100)
print(f"\nimplausible coords (outside a generous MNI box): {int(oob.sum()):,} "
      f"({oob.mean():.3%}) in {co[oob].contrast_id.nunique():,} analyses")
print(co[oob][["x","y","z","space"]].head(8).to_string())

# integer vs subvoxel
frac = ((co[["x","y","z"]] % 1) != 0).any(axis=1)
print(f"\nnon-integer coordinates: {frac.mean():.1%}")

# exact duplicate foci within an analysis
dup = co.duplicated(subset=["contrast_id","x","y","z"], keep="first")
print(f"duplicate foci within analysis: {int(dup.sum()):,} ({dup.mean():.2%}) "
      f"in {co[dup].contrast_id.nunique():,} analyses")

# all-zero / origin foci
origin = (co.x==0)&(co.y==0)&(co.z==0)
print(f"origin (0,0,0) foci: {int(origin.sum()):,}")

# Left-right asymmetry in reported x  -- reporting/lateralization check
print(f"\nx<0 (left): {(co.x<0).mean():.3%} | x>0 (right): {(co.x>0).mean():.3%} "
      f"| x==0 (midline): {(co.x==0).mean():.3%}")

print("\n" + "=" * 78); print("STUDY-LEVEL DUPLICATION"); print("=" * 78)
st = ss.metadata
print("analyses per study:", st.groupby("study_id").size().describe().to_string())
# same paper ingested twice?
names = st.drop_duplicates("study_id")["study_name"]
print(f"\ndistinct study_ids: {names.size:,}; distinct study_names: {names.nunique():,}")
dupn = names[names.duplicated(keep=False)].value_counts()
print(f"study_names attached to >1 study_id: {dupn.size:,}")
print(dupn.head(8).to_string())
