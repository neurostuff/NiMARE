"""
.. _ma_feature_dataset:

==========================================
Masked activation feature dataset workflow
==========================================

Convert a Studyset into a scikit-learn-compatible
dataset. This example uses modeled activation maps to classify n-back and
flanker task analyses from a parquet-backed Studyset.
"""

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from nimare.meta.kernel import MKDAKernel
from nimare.ml import MAFeatureExtractor
from nimare.nimads import Studyset
from nimare.utils import get_resource_path

RANDOM_SEED = 13

###############################################################################
# Load the n-back/flanker Studyset
# -----------------------------------------------------------------------------
# The bundled Studyset contains coordinate-based analyses of n-back and flanker
# tasks from NeuroStore. Its parquet tables are loaded from the accompanying
# ``studyset.json`` manifest.
studyset_dir = Path(get_resource_path()) / "nback_vs_flanker_studyset_2026-07"
studyset = Studyset(studyset_dir)

print(f"Studyset: {studyset.name}")
print(f"Analyses: {len(studyset.ids)}")
print(studyset.metadata["comparison_task"].value_counts().to_string())

###############################################################################
# Configure feature extraction
# -----------------------------------------------------------------------------
# The extractor applies an MKDA kernel to the coordinates from each analysis to
# generate voxelwise MA features. The ``comparison_task`` metadata field
# provides the ``"n-back"`` and ``"flanker"`` target labels.
extractor = MAFeatureExtractor(
    kernel_transformer=MKDAKernel(r=10),
    target_field={"source": "metadata", "field": "comparison_task"},
)

###############################################################################
# Export the scikit-learn dataset and classify task labels
# -----------------------------------------------------------------------------
# The voxelwise MA features are sparse and high-dimensional. Truncated SVD is
# fit within each cross-validation fold. GroupKFold keeps analyses from the
# same study together, preventing study-level leakage between partitions.
bunch = extractor.transform(
    studyset,
    map_reducer="truncated_svd",
    map_reducer_params={
        "n_components": 50,
        "random_state": RANDOM_SEED,
    },
)

pipeline = make_pipeline(
    bunch.preprocessor,
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    ),
)
scores = cross_val_score(
    pipeline,
    bunch.data,
    bunch.target,
    cv=GroupKFold(5),
    groups=bunch.groups,
)

print(f"Feature data shape: {bunch.data.shape}")
print(f"Labels: {sorted(set(bunch.target))}")
print(f"Cross-validation accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
