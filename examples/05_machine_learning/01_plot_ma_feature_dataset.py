"""
.. _ma_feature_dataset:

===========================================
Modeled activation feature dataset workflow
===========================================

Convert a Studyset into a scikit-learn-compatible dataset. This example uses
modeled activation maps to classify n-back and flanker task analyses from a
parquet-backed Studyset, and shows how to keep analyses from one study out of
two different partitions.
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
print(f"Analyses: {len(studyset.ids)} from {len(studyset.study_ids)} studies")
print(studyset.metadata["comparison_task"].value_counts().to_string())

###############################################################################
# Convert the Studyset into feature data
# -----------------------------------------------------------------------------
# The extractor applies an MKDA kernel to the coordinates of each analysis to
# generate voxelwise MA features, and reads the ``comparison_task`` metadata
# field as the ``"n-back"`` and ``"flanker"`` labels to predict. Fields are
# named by a bare field name, or by a ``(source, field)`` pair when the name
# appears in more than one Studyset table.
extractor = MAFeatureExtractor(
    kernel_transformer=MKDAKernel(r=10),
    target_field=("metadata", "comparison_task"),
)
data = extractor.transform(studyset)

print(data)
print(f"Voxel features: {data.map_features.shape[1]}")
print(f"Dropped for want of coordinates: {len(data.provenance['dropped_ids'])}")

###############################################################################
# Export it for scikit-learn
# -----------------------------------------------------------------------------
# ``to_sklearn`` returns the familiar bundle: sparse ``data``, aligned
# ``target``, and ``groups`` holding the study each analysis came from.
# ``extractor.to_sklearn(studyset)`` does the conversion and the export in one
# call when the container itself is not needed.
bunch = extractor.to_sklearn(studyset)

print(f"Feature data: {bunch.data.shape}, sparse={bunch.data.format}")
print(f"Labels: {sorted(set(bunch.target))}")

###############################################################################
# Split without leaking a study
# -----------------------------------------------------------------------------
# Analyses from one study are related, so a study belongs to exactly one
# partition. ``test_size`` is a fraction of *studies*, so the analysis counts
# only approximate it.
train, test = data.split(test_size=0.25, random_state=RANDOM_SEED)

print(f"Train: {len(train)} analyses from {len(set(train.study_ids))} studies")
print(f"Test:  {len(test)} analyses from {len(set(test.study_ids))} studies")
print(f"Shared studies: {set(train.study_ids) & set(test.study_ids)}")

###############################################################################
# Classify the task label
# -----------------------------------------------------------------------------
# The voxelwise MA features are sparse and high-dimensional, so truncated SVD
# reduces them before the classifier sees them. Putting the reducer in the
# pipeline is what keeps it fitted on training rows only;
# :meth:`~nimare.ml.MAFeatureDataset.make_preprocessor` builds the step that
# reduces the map columns and leaves any descriptor columns alone. GroupKFold
# reads the same study labels the split used.
pipeline = make_pipeline(
    data.make_preprocessor("truncated_svd", n_components=50, random_state=RANDOM_SEED),
    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
)
scores = cross_val_score(
    pipeline,
    bunch.data,
    bunch.target,
    cv=GroupKFold(5),
    groups=bunch.groups,
)

print(f"Cross-validation accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
