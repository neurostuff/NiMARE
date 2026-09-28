"""Descriptive meta-science: who is studied, how big, and where does the
literature point? Plus the annotation contradictions worth knowing about."""
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))
import numpy as np, pandas as pd
import cohort, nsnorm
pd.set_option("display.width", 200)

df = cohort.load_cohort()
task = cohort.clinical_task_mask(df)

print("=" * 88)
print("ANNOTATION CONTRADICTION: group_name says 'patients', diagnosis reads healthy")
print("=" * 88)
from nimare.extract import fetch_neurostore
ann = fetch_neurostore(version="nightly").annotations_df
G = "ParticipantDemographicsExtractor.groups[0]."
bad = (ann[G + "group_name"] == "patients") & (
    nsnorm.canonical_diagnosis(ann[G + "diagnosis"]) == "Healthy")
print(f"{int(bad.sum()):,} analyses affected ({bad.mean():.2%} of the release)")
print("\nthe diagnosis strings behind it:")
print(ann.loc[bad, G + "diagnosis"].value_counts().head(12).to_string())

print("\n" + "=" * 88)
print("WHO IS STUDIED: sample size and sex, by diagnosis (task fMRI)")
print("=" * 88)
sub = df[task & (df.group == "patients")]
g = sub.groupby("dx").agg(
    n_analyses=("dx", "size"),
    n_studies=("study_id", "nunique"),
    median_n=("n_subj", "median"),
    mean_age=("age", "mean"),
    pct_female=("female_frac", "mean"),
)
g = g[g.n_analyses >= 300].sort_values("n_analyses", ascending=False)
g["pct_female"] = (g.pct_female * 100).round(1)
print(g.round(1).to_string())

print("\n" + "=" * 88)
print("SAMPLE SIZE: healthy vs patient studies (task fMRI)")
print("=" * 88)
print(df[task].groupby("group").n_subj.describe(
    percentiles=[.25, .5, .75, .95]).round(1).to_string())

print("\n" + "=" * 88)
print("WHO IS STUDIED, by cognitive domain (healthy task fMRI)")
print("=" * 88)
h = df[cohort.healthy_task_mask(df)]
rows = []
for d in nsnorm.DOMAINS:
    s = h[h["dom_" + d]]
    rows.append({"domain": d, "n": len(s), "median_n_subj": s.n_subj.median(),
                 "mean_age": s.age.mean(), "pct_female": s.female_frac.mean() * 100,
                 "median_foci": s.n_peaks.median()})
r = pd.DataFrame(rows).sort_values("pct_female")
print(r.round(1).to_string(index=False))

print("\n" + "=" * 88)
print("STATISTICAL POWER: what fraction of analyses come from small samples?")
print("=" * 88)
n = df[task].n_subj.dropna()
for thr in (10, 16, 20, 30, 50, 100):
    print(f"  n <= {thr:<4}: {(n <= thr).mean():>6.1%} of analyses")
print(f"\nmedian group size: {n.median():.0f}   mean: {n.mean():.1f}")
