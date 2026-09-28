"""Figures for the exploration report."""
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import cohort, nsnorm

W = cohort.W
OUT = "/home/user/NiMARE/neurostore_exploration/figures"

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9992"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
BLUES = LinearSegmentedColormap.from_list("seq_blue", SEQ)
DIVERGING = LinearSegmentedColormap.from_list(
    "div", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#8d1f1e"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": MUTED,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "grid.color": "#e8e7e3", "grid.linewidth": 0.8,
})

def finish(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path, flush=True)

# --- 1. granularity curve -------------------------------------------------
sweep = pd.read_parquet(f"{W}/results_domain_sweep.parquet")
atl = sweep[sweep.atlas != "n_peaks only"]
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.set_axisbelow(True); ax.yaxis.grid(True)
for d, g in atl.groupby("domain"):
    g = g.sort_values("n_regions")
    ax.plot(g.n_regions, g.auc, color=MUTED, lw=1, alpha=0.45, zorder=2)
mean = atl.groupby("n_regions").auc.mean().sort_index()
ax.plot(mean.index, mean.values, color=BLUE, lw=2.4, zorder=4,
        marker="o", ms=5, mfc=SURFACE, mew=1.8, mec=BLUE)
base = sweep[sweep.atlas == "n_peaks only"].auc.mean()
ax.axhline(base, color=ORANGE, lw=1.6, ls="--", zorder=3)
ax.axhline(0.5, color=MUTED, lw=1, zorder=1)
ax.set_xscale("log")
ticks = sorted(atl.n_regions.unique())
ax.set_xticks(ticks)
ax.set_xticklabels([str(t) for t in ticks], fontsize=8.5)
# 128 and 148 sit almost on top of each other on a log axis; drop 148's tick
for tick, label in zip(ticks, ax.get_xticklabels()):
    if tick == 148:
        label.set_visible(False)
ax.set_xlabel("atlas regions (log scale)"); ax.set_ylabel("cross-validated ROC AUC")
ax.set_title("Cognitive-domain decoding does not improve with finer atlases")
best = mean.idxmax()
ax.annotate(f"mean across 10 domains\npeaks at {best} regions ({mean.max():.3f})",
            xy=(best, mean.max()), xytext=(46, 0.565),
            color=BLUE, fontsize=8.5, va="center",
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=1,
                            connectionstyle="angle3,angleA=0,angleB=70"))
ax.text(atl.n_regions.max(), base - 0.012, "focus-count baseline",
        color=ORANGE, fontsize=8.5, ha="right", va="top")
for d in ["Action", "Language", "Executive cognitive control"]:
    g = atl[atl.domain == d].sort_values("n_regions")
    ax.text(g.n_regions.iloc[-1] * 1.04, g.auc.iloc[-1], d, color=INK2,
            fontsize=8, va="center")
ax.set_xlim(35, 1900); ax.set_ylim(0.45, 0.78)
ax.tick_params(axis="x", which="minor", length=0)
finish(fig, f"{OUT}/01_atlas_granularity.png")

# --- 2. per-domain decodability ------------------------------------------
full = pd.read_parquet(f"{W}/results_domain_auc_full.parquet").sort_values("auc")
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.set_axisbelow(True); ax.xaxis.grid(True)
y = np.arange(len(full))
ax.barh(y, full.auc - 0.5, left=0.5, color=BLUE, height=0.62)
ax.set_yticks(y); ax.set_yticklabels(full.index)
ax.axvline(0.5, color=MUTED, lw=1.2)
for i, v in enumerate(full.auc):
    ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=8.5, color=INK2)
ax.set_xlim(0.5, full.auc.max() + 0.045)
ax.set_xlabel("cross-validated ROC AUC (0.5 = chance)")
ax.set_title("How well do reported coordinates identify a cognitive domain?")
finish(fig, f"{OUT}/02_domain_decodability.png")

# --- 3. domain similarity -------------------------------------------------
C = pd.read_parquet(f"{W}/results_domain_similarity.parquet")
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
d = 1 - C.values; np.fill_diagonal(d, 0)
order = leaves_list(linkage(squareform(d, checks=False), method="average"))
Cs = C.iloc[order, order]
vals = np.array(Cs.values, dtype=float, copy=True)
np.fill_diagonal(vals, np.nan)  # 1.0 by construction; it only flattens the scale
fig, ax = plt.subplots(figsize=(6.6, 5.6))
im = ax.imshow(vals, cmap=DIVERGING, norm=TwoSlopeNorm(0, -0.5, 0.5))
im.cmap.set_bad("#e8e7e3")
ax.set_xticks(range(len(Cs))); ax.set_yticks(range(len(Cs)))
ax.set_xticklabels(Cs.columns, rotation=40, ha="right", fontsize=8)
ax.set_yticklabels(Cs.index, fontsize=8)
for i in range(len(Cs)):
    for j in range(len(Cs)):
        if i == j:
            continue
        v = vals[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="#ffffff" if abs(v) > 0.33 else INK)
ax.set_title("Domains that share a coordinate signature\n(r between one-vs-rest weight maps)")
fig.colorbar(im, ax=ax, shrink=0.72, label="correlation")
ax.set_xticks(np.arange(-.5, len(Cs), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(Cs), 1), minor=True)
ax.grid(which="minor", color=SURFACE, lw=2); ax.tick_params(which="minor", length=0)
finish(fig, f"{OUT}/03_domain_similarity.png")

# --- 4. clinical confound ladder -----------------------------------------
lad = pd.read_parquet(f"{W}/results_clinical_ladder.parquet")
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.set_axisbelow(True); ax.yaxis.grid(True)
x = np.arange(len(lad))
ax.bar(x, lad.auc - 0.5, bottom=0.5, color=BLUE, width=0.6)
ax.set_xticks(x)
ax.set_xticklabels([s.split(". ", 1)[1] for s in lad.step], fontsize=8.5,
                   rotation=12, ha="right")
ax.axhline(0.5, color=MUTED, lw=1.2)
for i, (v, n) in enumerate(zip(lad.auc, lad.n)):
    ax.text(i, v + 0.004, f"{v:.3f}\nn={n:,}", ha="center", fontsize=8, color=INK2)
DESIGN_ONLY = 0.605  # domain labels + focus count, no coordinates (21_clinical.py)
ax.axhline(DESIGN_ONLY, color=ORANGE, lw=1.8, ls="--", zorder=5)
ax.text(len(lad) - 0.5, DESIGN_ONLY + 0.002,
        "study design alone (no coordinates): 0.605", color=ORANGE, fontsize=8.5,
        ha="right", va="bottom")
ax.set_ylim(0.5, lad.auc.max() + 0.055)
ax.set_ylabel("ROC AUC, patients vs healthy")
ax.set_title("Most of the apparent 'clinical signature' is study-design confound")
finish(fig, f"{OUT}/04_clinical_ladder.png")

# --- 5. cross-diagnosis transfer -----------------------------------------
T = pd.read_parquet(f"{W}/results_transfer.parquet")
PSYCH = ["Schizophrenia", "Bipolar disorder", "Major depression", "Anxiety disorders",
         "PTSD", "ADHD", "Substance use", "Autism spectrum"]
NEURO = ["Stroke / vascular", "Parkinson's disease", "Epilepsy",
         "Tinnitus / sensory", "Chronic pain"]
blocks = [d for d in PSYCH + NEURO if d in T.index]
T = T.loc[blocks, blocks]
split = sum(d in PSYCH for d in blocks)
fig, ax = plt.subplots(figsize=(7.4, 6.2))
im = ax.imshow(T.values.astype(float), cmap=DIVERGING,
               norm=TwoSlopeNorm(0.5, 0.42, 0.68))
ax.set_xticks(range(len(T.columns))); ax.set_yticks(range(len(T.index)))
ax.set_xticklabels(T.columns, rotation=40, ha="right", fontsize=8)
ax.set_yticklabels(T.index, fontsize=8)
for i in range(T.shape[0]):
    for j in range(T.shape[1]):
        v = T.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="#ffffff" if abs(v - 0.5) > 0.12 else INK)
for pos in (split - 0.5,):
    ax.axhline(pos, color=INK, lw=1.6); ax.axvline(pos, color=INK, lw=1.6)
ax.text(split / 2 - 0.5, -0.75, "psychiatric", ha="center", fontsize=9,
        color=INK, weight="semibold")
ax.text((split + len(blocks)) / 2 - 0.5, -0.75, "neurological / sensory",
        ha="center", fontsize=9, color=INK, weight="semibold")
ax.set_xlabel("tested on (held-out studies)"); ax.set_ylabel("trained on")
ax.set_title("Does a disorder's coordinate signature transfer to another disorder?",
             pad=34)
fig.colorbar(im, ax=ax, shrink=0.72, label="ROC AUC (0.5 = chance)")
ax.set_xticks(np.arange(-.5, len(T.columns), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(T.index), 1), minor=True)
ax.grid(which="minor", color=SURFACE, lw=2); ax.tick_params(which="minor", length=0)
finish(fig, f"{OUT}/05_transfer_matrix.png")

# --- 6. who is studied ----------------------------------------------------
df = cohort.load_cohort()
task = cohort.clinical_task_mask(df)
sub = df[task & (df.group == "patients")]
g = sub.groupby("dx").agg(n=("dx", "size"), pct_female=("female_frac", "mean"))
g = g[g.n >= 300].sort_values("pct_female")
g["pct_female"] *= 100
fig, ax = plt.subplots(figsize=(7.2, 5.4))
ax.set_axisbelow(True); ax.xaxis.grid(True)
y = np.arange(len(g))
colors = [BLUE if v < 50 else ORANGE for v in g.pct_female]
ax.barh(y, g.pct_female - 50, left=50, color=colors, height=0.66)
ax.axvline(50, color=MUTED, lw=1.4)
ax.set_yticks(y); ax.set_yticklabels([f"{i}  (n={n:,})" for i, n in zip(g.index, g.n)],
                                     fontsize=8)
for i, v in enumerate(g.pct_female):
    ax.text(v + (1.2 if v >= 50 else -1.2), i, f"{v:.0f}%", va="center", fontsize=8,
            ha="left" if v >= 50 else "right", color=INK2)
ax.set_xlim(0, 100); ax.set_xlabel("% female in the reported patient group")
ax.set_title("Who is in the clinical imaging literature")
ax.text(50.8, -0.75, "parity", color=MUTED, fontsize=8, va="center")
finish(fig, f"{OUT}/06_sex_representation.png")
print("figures done")
