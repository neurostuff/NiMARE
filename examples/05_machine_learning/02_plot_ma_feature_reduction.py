"""
.. _ma_feature_reduction:

=============================================
Modeled activation feature reduction workflow
=============================================

Compare the reduction workflows :func:`~nimare.ml.make_map_reducer` provides:
truncated SVD, DiFuMo atlas aggregation, and variance thresholding. The first
two produce 64 features each, and all three are evaluated on the same
study-grouped splits of one Studyset.
"""

from pathlib import Path

from nilearn.datasets import fetch_atlas_difumo
from nilearn.maskers import NiftiMapsMasker
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline

from nimare.meta.kernel import MKDAKernel
from nimare.ml import MAFeatureExtractor, make_map_reducer
from nimare.nimads import Studyset
from nimare.utils import get_resource_path

RANDOM_SEED = 13
N_COMPONENTS = 64

###############################################################################
# Convert the Studyset once
# -----------------------------------------------------------------------------
# As in the preceding example, the bundled Studyset provides analysis-level
# coordinates and n-back/flanker task labels. The MA maps are generated once
# here and reused by every reduction workflow below.
studyset_dir = Path(get_resource_path()) / "nback_vs_flanker_studyset_2026-07"
studyset = Studyset(studyset_dir)

extractor = MAFeatureExtractor(
    kernel_transformer=MKDAKernel(r=10),
    target_field=("metadata", "comparison_task"),
)
data = extractor.transform(studyset)

print(f"Voxelwise feature shape: {data.features.shape}")

###############################################################################
# Configure the reduction workflows
# -----------------------------------------------------------------------------
# Atlas aggregation summarizes each voxelwise MA map with one feature for each
# of the 64 DiFuMo components. Variance thresholding keeps the matrix sparse
# and simply drops the voxels that barely vary across analyses.
difumo = fetch_atlas_difumo(dimension=N_COMPONENTS, resolution_mm=2)
atlas_masker = NiftiMapsMasker(
    maps_img=difumo.maps,
    standardize=False,
    resampling_target="data",
    reports=False,
)

reducers = {
    "Truncated SVD": make_map_reducer(
        "truncated_svd", n_components=N_COMPONENTS, random_state=RANDOM_SEED
    ),
    "DiFuMo atlas": make_map_reducer(
        # Bigger batches hold more rows in dense image form at once, and pay
        # nilearn's per-call least-squares setup fewer times.
        "atlas_aggregation",
        masker=data.masker,
        atlas_masker=atlas_masker,
        batch_size=64,
    ),
    "Variance threshold": make_map_reducer("variance_threshold", threshold=0.01),
}

###############################################################################
# Compare them under grouped cross-validation
# -----------------------------------------------------------------------------
# Each reducer goes inside the pipeline, so it is fitted on the training
# analyses of the split and only then applied to the held-out ones.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
bunch = data.to_sklearn()

for name, reducer in reducers.items():
    pipeline = make_pipeline(
        data.make_preprocessor(reducer),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
    )
    score = cross_val_score(
        pipeline,
        bunch.data,
        bunch.target,
        cv=splitter,
        groups=bunch.groups,
    )[0]
    print(f"{name} grouped holdout accuracy: {score:.3f}")

###############################################################################
# Reduce a held-out partition by hand
# -----------------------------------------------------------------------------
# Outside a pipeline, fit the reducer on the training dataset and hand the same
# fitted reducer to the test dataset. Passing an unfitted one raises, because
# fitting it there would use the held-out analyses.
train, test = data.split(test_size=0.2, random_state=RANDOM_SEED)
svd = make_map_reducer("truncated_svd", n_components=N_COMPONENTS, random_state=RANDOM_SEED)

train_reduced = train.fit_transform_maps(svd)
test_reduced = test.transform_maps(svd)

print(f"Reduced train features: {train_reduced.features.shape}")
print(f"Reduced test features:  {test_reduced.features.shape}")
print(f"First feature names: {train_reduced.feature_names[:3]}")
