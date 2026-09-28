"""Does the cross-disorder transfer split along the psychiatric / neurological line?"""
import warnings; warnings.filterwarnings("ignore")
import sys; from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))

from cohort import WORK  # noqa: E402
import numpy as np, pandas as pd, cohort
pd.set_option("display.width", 200)

T = pd.read_parquet(f"{cohort.W}/results_transfer.parquet").astype(float)
PSYCH = ["Schizophrenia", "Major depression", "Bipolar disorder", "Anxiety disorders",
         "PTSD", "ADHD", "Substance use", "Autism spectrum"]
NEURO = ["Stroke / vascular", "Parkinson's disease", "Epilepsy", "Tinnitus / sensory",
         "Chronic pain"]
block = {d: ("psychiatric" if d in PSYCH else "neurological") for d in T.index}

rows = []
for tr in T.index:
    for te in T.columns:
        if tr == te:
            continue
        rows.append({"train": tr, "test": te, "auc": T.loc[tr, te],
                     "pair": f"{block[tr]} -> {block[te]}"})
long = pd.DataFrame(rows)
print("=" * 70)
print("CROSS-DISORDER TRANSFER BY BLOCK (diagonal excluded)")
print("=" * 70)
print(long.groupby("pair").auc.agg(["mean", "std", "size"]).round(3).to_string())

within_self = pd.Series({d: T.loc[d, d] for d in T.index})
print("\nwithin-disorder (held-out) AUC:")
print(within_self.sort_values(ascending=False).round(3).to_string())

print("\n" + "=" * 70)
print("TRANSFER-IN score: how well OTHER disorders' models detect this one")
print("=" * 70)
arr = np.array(T.values, dtype=float, copy=True)
np.fill_diagonal(arr, np.nan)
off = pd.DataFrame(arr, index=T.index, columns=T.columns)
summary = pd.DataFrame({
    "self": within_self,
    "detected_by_others": off.mean(axis=0),
    "detects_others": off.mean(axis=1),
    "block": pd.Series(block),
}).sort_values("detected_by_others", ascending=False)
print(summary.round(3).to_string())
summary.to_parquet(f"{cohort.W}/results_transfer_blocks.parquet")

print("\nbest cross-disorder pairs:")
print(long.nlargest(8, "auc")[["train", "test", "auc"]].round(3).to_string(index=False))
