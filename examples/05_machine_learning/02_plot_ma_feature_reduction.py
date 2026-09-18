"""
.. _ma_feature_reduction:

============================================
Masked activation feature reduction workflow
============================================

Compare truncated SVD with DiFuMo atlas aggregation for reducing voxelwise
modeled activation (MA) features. Both approaches produce 64 features and are
evaluated on the same study-grouped holdout split.
"""

from pathlib import Path

from nilearn.datasets import fetch_atlas_difumo
from nilearn.maskers import NiftiMapsMasker
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline

from nimare.meta.kernel import MKDAKernel
from nimare.ml import MAFeatureExtractor
from nimare.nimads import Studyset
from nimare.utils import get_resource_path

RANDOM_SEED = 13
N_COMPONENTS = 64

###############################################################################
# Load the n-back/flanker Studyset
# -----------------------------------------------------------------------------
# As in the preceding example, the bundled Studyset provides analysis-level
# coordinates and n-back/flanker task labels.
studyset_dir = Path(get_resource_path()) / "nback_vs_flanker_studyset_2026-07"
studyset = Studyset(studyset_dir)

extractor = MAFeatureExtractor(
    kernel_transformer=MKDAKernel(r=10),
    target_field={"source": "metadata", "field": "comparison_task"},
)

###############################################################################
# Configure feature reduction workflows
# -----------------------------------------------------------------------------
# Atlas aggregation summarizes each voxelwise MA map with one feature for each
# of the 64 DiFuMo components.
difumo = fetch_atlas_difumo(dimension=N_COMPONENTS, resolution_mm=2)
atlas_masker = NiftiMapsMasker(
    maps_img=difumo.maps,
    standardize=False,
    resampling_target="data",
    reports=False,
)

reduction_workflows = {
    "Truncated SVD": (
        "truncated_svd",
        {
            "n_components": N_COMPONENTS,
            "random_state": RANDOM_SEED,
        },
    ),
    "DiFuMo atlas": (
        "atlas_aggregation",
        {
            "atlas_masker": atlas_masker,
            "batch_size": 10,
        },
    ),
}

###############################################################################
# Export the scikit-learn datasets and compare task classification
# -----------------------------------------------------------------------------
# The extractor caches the voxelwise MA features, so the second workflow does
# not recompute them. Each reducer is fit within the pipeline using only the
# training partition, which prevents information from leaking from the holdout
# data.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
workflow_scores = {}

for name, (method, params) in reduction_workflows.items():
    bunch = extractor.transform(
        studyset,
        map_reducer=method,
        map_reducer_params=params,
    )

    pipeline = make_pipeline(
        bunch.preprocessor,
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
    )
    workflow_scores[name] = cross_val_score(
        pipeline,
        bunch.data,
        bunch.target,
        cv=splitter,
        groups=bunch.groups,
    )[0]

print(f"Voxelwise feature shape: {bunch.data.shape}")
print(f"Reduced map features: {N_COMPONENTS}")
for name, score in workflow_scores.items():
    print(f"{name} grouped holdout accuracy: {score:.3f}")
