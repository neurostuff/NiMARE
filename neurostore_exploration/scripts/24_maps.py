"""Read the fitted models back to the brain with nimare.ml.coefficient_image.

The atlas features were precomputed with a validated fast equivalent of
MaskerTransformer, so the pipeline handed to coefficient_image carries the same
fitted MaskerTransformer (whose fit depends only on the feature width) and the
model fitted on those features.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))
import numpy as np, pandas as pd
import cohort, nsnorm
from atlases import ATLASES
from nimare.extract import fetch_neurostore
from nimare.meta.kernel import MKDAKernel
from nimare.ml import MAKernel, MaskerTransformer, coefficient_image
from nilearn import plotting
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

W, SEED, ATLAS = cohort.W, 13, "DiFuMo-256"
os.makedirs(f"{W}/maps", exist_ok=True)

ss = fetch_neurostore(version="nightly")
bunch = ss.to_bunch()
rows = np.load(f"{W}/feat/rows.npy")

kernel = MAKernel(MKDAKernel(r=10), source_masker=bunch.masker)
peaks = bunch.data[:, bunch.voxel_columns]
probe = kernel.fit_transform(peaks[:2])
atlas_step = MaskerTransformer(ATLASES[ATLAS](), source_masker=bunch.masker).fit(probe)
print(f"atlas step fitted: {ATLAS}", flush=True)

df = cohort.load_cohort()

def fit_and_project(mask, y, tag, title):
    idx = np.flatnonzero(mask)
    X = cohort.load_features(ATLAS, idx)
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=3000, class_weight="balanced",
                               random_state=SEED).fit(scaler.transform(X), y[idx])
    # A standardized model's weight per raw feature is w / scale_. Undoing the
    # scaler with its own inverse_transform would add the means back, which is
    # not what a coefficient means, so hand the back-transformed weights over
    # and leave the scaler out of the pipeline that is walked.
    pipe = Pipeline([("kernel", kernel), ("atlas", atlas_step), ("clf", model)])
    image = coefficient_image(pipe, bunch, coef=model.coef_.ravel() / scaler.scale_)
    path = f"{W}/maps/{tag}.nii.gz"
    image.to_filename(path)
    plotting.plot_stat_map(
        image, display_mode="z", cut_coords=[-16, -4, 8, 24, 44], title=title,
        output_file=f"{W}/maps/{tag}.png", colorbar=True,
    )
    print(f"  {tag:<34} n={len(idx):>6,}  -> {path}", flush=True)
    return image

print("\nDomain weight maps (healthy task fMRI, one-vs-rest):", flush=True)
hmask = cohort.healthy_task_mask(df).values
for d in nsnorm.DOMAINS:
    y = df["dom_" + d].values.astype(int)
    fit_and_project(hmask, y, "domain_" + d.replace(" ", "_"),
                    f"{d} vs other domains")

print("\nClinical map (task fMRI, patients vs healthy):", flush=True)
cmask = cohort.clinical_task_mask(df).values
fit_and_project(cmask, (df.group == "patients").values.astype(int),
                "clinical_patients_vs_healthy", "Patients vs healthy (task fMRI)")
print("\nmaps written")
