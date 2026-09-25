"""
.. _ma_feature_reduction:

=============================================
Modeled activation feature reduction workflow
=============================================

Reduce voxelwise modeled activation (MA) features with any scikit-learn
transformer, or over the regions of any atlas nilearn can load. This example
compares truncated SVD, a sparse random projection, DiFuMo atlas aggregation
and variance thresholding on the same study-grouped split of one Studyset.
"""

from pathlib import Path

from nilearn.datasets import fetch_atlas_difumo
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import SparseRandomProjection

from nimare.meta.kernel import MKDAKernel
from nimare.ml import extract_features, make_map_reducer
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

features = extract_features(
    studyset,
    kernel_transformer=MKDAKernel(r=10),
    target_field=("metadata", "comparison_task"),
)

print(f"Voxelwise feature shape: {features.features.shape}")

###############################################################################
# Configure the reduction workflows
# -----------------------------------------------------------------------------
# :func:`~nimare.ml.make_map_reducer` takes a workflow name, any scikit-learn
# transformer or transformer class, or any atlas: what a
# ``nilearn.datasets.fetch_atlas_*`` function returns, an atlas image or file, a
# fetcher name such as ``atlas="harvard_oxford"``, or a nilearn masker you
# configured yourself. A 4D atlas is summarised with a ``NiftiMapsMasker`` and a
# 3D one with a ``NiftiLabelsMasker``, and region names come along for the ride.
#
# Reducers see the sparse voxel matrix, so they have to accept sparse input:
# truncated SVD, sparse random projection, variance thresholding and atlas
# aggregation all do, while dense PCA would ask to be given dense data.
difumo = fetch_atlas_difumo(dimension=N_COMPONENTS, resolution_mm=2)

reducers = {
    "Truncated SVD": make_map_reducer(
        "truncated_svd", n_components=N_COMPONENTS, random_state=RANDOM_SEED
    ),
    "Sparse random projection": make_map_reducer(
        SparseRandomProjection, n_components=N_COMPONENTS, random_state=RANDOM_SEED
    ),
    "DiFuMo atlas": make_map_reducer(
        difumo,
        masker=features.masker,
        # Bigger batches hold more rows in dense image form at once, and pay
        # nilearn's per-call least-squares setup fewer times.
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
bunch = features.to_sklearn()

for name, reducer in reducers.items():
    pipeline = make_pipeline(
        # An atlas can go straight in here too: the dataset knows the masker
        # that defines its voxel order, so `features.make_preprocessor(difumo)`
        # builds the same aggregator.
        features.make_preprocessor(reducer),
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
train, test = features.split(test_size=0.2, random_state=RANDOM_SEED)
svd = make_map_reducer("truncated_svd", n_components=N_COMPONENTS, random_state=RANDOM_SEED)

train_reduced = train.fit_transform_maps(svd)
test_reduced = test.transform_maps(svd)

print(f"Reduced train features: {train_reduced.features.shape}")
print(f"Reduced test features:  {test_reduced.features.shape}")
print(f"First feature names: {train_reduced.feature_names[:3]}")

###############################################################################
# Region names come from the atlas
# -----------------------------------------------------------------------------
# When the atlas carries names, the reduced dataset's features are named after
# the regions rather than by position.
atlas_reduced = train.fit_transform_maps(
    make_map_reducer(difumo, masker=features.masker, batch_size=64)
)

print(f"Atlas features: {atlas_reduced.map_features.shape[1]}")
print(f"First region names: {atlas_reduced.feature_names[:3]}")
