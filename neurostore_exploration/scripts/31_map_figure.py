"""A montage of the domain weight maps 24_maps.py wrote.

Rendered from the saved images so the display can be tuned without refitting.
Each map is drawn to its own PNG and the montage is composed from those: giving
``plot_stat_map`` a short shared axes makes it scale its own title to fill the
row, which covers the slices.
"""
import warnings; warnings.filterwarnings("ignore")
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))

from cohort import WORK  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from nilearn import plotting  # noqa: E402
from nilearn.image import load_img, math_img  # noqa: E402

import nsnorm  # noqa: E402

OUT = str(_ROOT.parent / "figures")
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
AUC = {"Action": 0.753, "Language": 0.716, "Emotion": 0.688, "Motivation": 0.677,
       "Social function": 0.656, "Perception": 0.655,
       "Reasoning and decision making": 0.616, "Learning and memory": 0.613,
       "Attention": 0.602, "Executive cognitive control": 0.588}
ORDER = sorted(nsnorm.DOMAINS, key=lambda d: -AUC[d])
CUTS = [-16, -2, 12, 28, 46]


def render(domain, path):
    """Draw one domain's positive weights to ``path``."""
    source = f"{WORK}/maps/domain_{domain.replace(' ', '_')}.nii.gz"
    # Positive weights only: the regions that argue *for* this domain. The
    # negative half means "belongs to some other domain" and is not
    # interpretable one domain at a time.
    img = math_img("np.clip(img, 0, None)", img=load_img(source))
    positive = np.asarray(img.dataobj)
    positive = positive[positive > 0]
    # A fixed threshold would flood one map and empty another, so each map is
    # cut, and its colour saturated, at its own percentiles.
    # Its own figure, so the panel's background matches the montage page
    # rather than landing as a white block on it.
    figure = plt.figure(figsize=(8.0, 1.7), facecolor=SURFACE)
    plotting.plot_stat_map(
        img, figure=figure, display_mode="z", cut_coords=CUTS,
        threshold=float(np.percentile(positive, 98.5)),
        vmax=float(np.percentile(positive, 99.95)),
        colorbar=False, annotate=False, black_bg=False, cmap="hot_r",
    )
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


with tempfile.TemporaryDirectory() as tmp:
    panels = []
    for domain in ORDER:
        path = f"{tmp}/{domain.replace(' ', '_')}.png"
        render(domain, path)
        panels.append(mpimg.imread(path))
        print(f"rendered {domain}", flush=True)

    fig, axes = plt.subplots(len(ORDER), 1, figsize=(8.6, 1.16 * len(ORDER)),
                             facecolor=SURFACE)
    for ax, domain, panel in zip(axes, ORDER, panels):
        ax.imshow(panel)
        ax.axis("off")
        ax.text(0.0, 0.5, domain, transform=ax.transAxes, ha="right", va="center",
                fontsize=9.5, color=INK)
        ax.text(1.0, 0.5, f"AUC {AUC[domain]:.3f}", transform=ax.transAxes,
                ha="left", va="center", fontsize=9, color=MUTED)

    fig.suptitle("What each cognitive domain's model reads off the coordinates",
                 fontsize=12.5, fontweight="bold", color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.005, "positive one-vs-rest weights, healthy task-fMRI "
                          f"(n=50,665), DiFuMo-256, z = {CUTS[0]} to {CUTS[-1]}; "
                          "each map cut at its own 98.5th percentile",
             fontsize=7.5, color=MUTED)
    fig.subplots_adjust(left=0.25, right=0.88, top=0.96, bottom=0.03, hspace=0.0)
    fig.savefig(f"{OUT}/07_domain_maps.png", dpi=150, facecolor=SURFACE)
    print(f"wrote {OUT}/07_domain_maps.png")
