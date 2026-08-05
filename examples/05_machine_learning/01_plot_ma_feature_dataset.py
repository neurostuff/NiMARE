"""
.. _ma_feature_dataset:

==========================================
Masked activation feature dataset workflow
==========================================

Convert a Studyset with complete coordinates into scikit-learn-compatible
training and test datasets. This example uses modeled activation maps to
classify n-back and flanker task analyses from a parquet-backed Studyset.
"""

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from nimare.meta.kernel import MKDAKernel
from nimare.ml import MAFeatureExtractor
from nimare.nimads import Studyset
from nimare.utils import get_resource_path

RANDOM_SEED = 13

###############################################################################
# Load the n-back/flanker Studyset
# -----------------------------------------------------------------------------
# The bundled parquet Studyset contains coordinate analyses selected from
# NeuroStore for n-back and flanker tasks. The Studyset constructor reads the
# ``studyset.json`` manifest and keeps the table-backed views available without
# materializing nested Study and Analysis objects.
studyset_dir = Path(get_resource_path()) / "nback_vs_flanker_studyset_2026-07"
studyset = Studyset(studyset_dir)

print(f"Studyset: {studyset.name}")
print(f"Analyses: {len(studyset.ids)}")
print(studyset.metadata["comparison_task"].value_counts().to_string())

###############################################################################
# Configure feature extraction
# -----------------------------------------------------------------------------
# ``comparison_task`` is a metadata field with labels ``"n-back"`` and
# ``"flanker"``. The split is grouped by Studyset study ID, so analyses from
# one study cannot appear in both partitions.
extractor = MAFeatureExtractor(
    kernel_transformer=MKDAKernel(r=10),
    target_field={"source": "metadata", "field": "comparison_task"},
    test_size=0.25,
    random_state=RANDOM_SEED,
)

###############################################################################
# Export scikit-learn datasets and classify task labels
# -----------------------------------------------------------------------------
# The masked activation maps are sparse and high-dimensional, so we reduce only
# the map features with truncated SVD before fitting a simple linear classifier.
train_dataset, test_dataset = extractor.transform(
    studyset,
    map_reducer="truncated_svd",
    map_reducer_params={"n_components": 50},
)
train_bunch = train_dataset.to_sklearn()
test_bunch = test_dataset.to_sklearn()

classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=RANDOM_SEED,
)
classifier.fit(train_bunch.data, train_bunch.target)
predicted = classifier.predict(test_bunch.data)
labels = sorted(set(train_bunch.target))

print(f"Training data shape: {train_bunch.data.shape}")
print(f"Test data shape: {test_bunch.data.shape}")
print(f"Training labels: {labels}")
print(f"Test accuracy: {accuracy_score(test_bunch.target, predicted):.3f}")
print("Test balanced accuracy: " f"{balanced_accuracy_score(test_bunch.target, predicted):.3f}")
print("Confusion matrix:")
print(confusion_matrix(test_bunch.target, predicted, labels=labels))
