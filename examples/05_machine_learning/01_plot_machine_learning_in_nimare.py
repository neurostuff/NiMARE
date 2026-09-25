"""
.. _machine_learning_in_nimare:

============================
Machine learning in NiMARE
============================

Turn a Studyset into a scikit-learn dataset, split it without leaking a study,
and reduce the voxelwise features before fitting a model.

:class:`~nimare.ml.FeatureSet` holds modeled activation (MA) features, the
study each analysis came from, and an optional target, all in one row order.
NiMARE builds it; scikit-learn does the splitting, fitting and scoring.
"""

from pathlib import Path

from nilearn.datasets import fetch_atlas_difumo
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import SparseRandomProjection

from nimare.meta.kernel import MKDAKernel
from nimare.ml import AtlasAggregator, FeatureSet
from nimare.nimads import Studyset
from nimare.utils import get_resource_path

RANDOM_SEED = 13
N_COMPONENTS = 64

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
# Build the feature set
# -----------------------------------------------------------------------------
# :meth:`~nimare.ml.FeatureSet.from_studyset` applies an MKDA kernel to the
# coordinates of each analysis to generate voxelwise MA features, and reads the
# ``comparison_task`` metadata field as the ``"n-back"`` and ``"flanker"``
# labels to predict. Fields are named by a bare field name, or by a
# ``(source, field)`` pair when the name appears in more than one Studyset
# table.
#
# Generating the maps before splitting does not leak: an analysis's MA map is a
# function of that analysis's own foci, so it never sees the target or another
# analysis. Only steps that learn *across* rows have to be fitted inside a
# split, which is what the pipeline below is for.
features = FeatureSet.from_studyset(
    studyset,
    kernel_transformer=MKDAKernel(r=10),
    target_field=("metadata", "comparison_task"),
)

print(features)
print(f"Voxel features: {features.map_features.shape[1]}")
print(f"Dropped for want of coordinates: {len(features.provenance['dropped_ids'])}")

###############################################################################
# Export it for scikit-learn
# -----------------------------------------------------------------------------
# ``to_sklearn`` returns the familiar bundle: sparse ``data``, aligned
# ``target``, and ``groups`` holding the study each analysis came from. Pass
# ``return_X_y=True`` for a bare ``(X, y)`` pair instead.
bunch = features.to_sklearn()

print(f"Feature data: {bunch.data.shape}, sparse={bunch.data.format}")
print(f"Labels: {sorted(set(bunch.target))}")

###############################################################################
# Split without leaking a study
# -----------------------------------------------------------------------------
# Analyses from one study are related, so a study belongs to exactly one
# partition. ``test_size`` is a fraction of *studies*, so the analysis counts
# only approximate it.
train, test = features.split(test_size=0.25, random_state=RANDOM_SEED)

print(f"Train: {len(train)} analyses from {len(set(train.study_ids))} studies")
print(f"Test:  {len(test)} analyses from {len(set(test.study_ids))} studies")
print(f"Shared studies: {set(train.study_ids) & set(test.study_ids)}")

###############################################################################
# Classify the task label
# -----------------------------------------------------------------------------
# ``bunch.data`` is an ordinary sparse matrix, so an ordinary scikit-learn
# pipeline works on it. The voxelwise MA features are high-dimensional, so
# truncated SVD reduces them before the classifier sees them; putting the
# reducer in the pipeline is what keeps it fitted on training rows only.
# GroupKFold reads the same study labels the split used.
pipeline = make_pipeline(
    TruncatedSVD(n_components=50, random_state=RANDOM_SEED),
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

###############################################################################
# Add study information as extra features
# -----------------------------------------------------------------------------
# Numeric metadata and annotation fields can be appended to the feature matrix
# as extra columns. Nothing is filled in silently: by default a field that some
# analyses do not report stops the conversion and names them, because a column
# that is mostly imputed is a modelling decision rather than a detail. Pass
# ``missing_values="drop"`` to remove those analyses or ``"keep"`` to leave the
# gaps for an imputer in your pipeline.
try:
    FeatureSet.from_studyset(
        studyset,
        kernel_transformer=MKDAKernel(r=10),
        descriptor_fields=["sample_sizes"],
    )
except ValueError as exc:
    print(f"{str(exc)[:160]}...")

###############################################################################
# Once there are descriptor columns, the reducer has to be kept off them, which
# is what :class:`~sklearn.compose.ColumnTransformer` is for.
# :meth:`~nimare.ml.FeatureSet.make_preprocessor` builds one with the column
# boundary filled in and ``sparse_threshold=1.0``, so a wide sparse map block
# is never quietly densified. :attr:`~nimare.ml.FeatureSet.map_columns` and
# :attr:`~nimare.ml.FeatureSet.descriptor_columns` are public if you would
# rather write it out. With map features alone, as above, there is nothing to
# keep the reducer away from and the method hands the reducer straight back.
with_descriptors = FeatureSet.from_studyset(
    studyset,
    kernel_transformer=MKDAKernel(r=10),
    target_field=("metadata", "comparison_task"),
    descriptor_fields=["sample_sizes"],
    missing_values="keep",
)
preprocessor = with_descriptors.make_preprocessor(
    TruncatedSVD(n_components=50, random_state=RANDOM_SEED),
    descriptor_transformer=SimpleImputer(strategy="median"),
)

print(f"Descriptor columns: {with_descriptors.descriptor_columns}")
print(f"Preprocessor: {type(preprocessor).__name__}")
print(f"Map-only feature set: {type(features.make_preprocessor(TruncatedSVD(2))).__name__}")

###############################################################################
# Compare reduction workflows
# -----------------------------------------------------------------------------
# Any scikit-learn transformer will do. They see the sparse voxel matrix, so
# they have to accept sparse input: truncated SVD, sparse random projection,
# variance thresholding and atlas aggregation all do, while dense PCA would ask
# to be given dense data.
reducers = {
    "Truncated SVD": TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_SEED),
    "Sparse random projection": SparseRandomProjection(
        n_components=N_COMPONENTS, random_state=RANDOM_SEED
    ),
    "Variance threshold": VarianceThreshold(threshold=0.01),
}
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)

for name, reducer in reducers.items():
    reduced_pipeline = make_pipeline(
        reducer,
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
    )
    score = cross_val_score(
        reduced_pipeline,
        bunch.data,
        bunch.target,
        cv=splitter,
        groups=bunch.groups,
    )[0]
    print(f"{name} grouped holdout accuracy: {score:.3f}")

###############################################################################
# Reduce over the regions of an atlas
# -----------------------------------------------------------------------------
# :class:`~nimare.ml.AtlasAggregator` is the one reducer NiMARE adds, because
# it is the one that has to know which voxel each column is. An atlas is
# anything nilearn can load: what a ``fetch_atlas_*`` function returns, an
# atlas image or file, the name of a fetcher such as ``atlas="harvard_oxford"``,
# or a masker you configured yourself. A 4D atlas is summarised with a
# ``NiftiMapsMasker`` and a 3D one with a ``NiftiLabelsMasker``, and the
# atlas's own region names become the feature names.
#
# Outside a pipeline, fit the reducer on the training feature set and hand the
# same fitted reducer to the test one. Passing an unfitted reducer to
# :meth:`~nimare.ml.FeatureSet.transform_maps` raises, because fitting it there
# would use the held-out analyses.
difumo = fetch_atlas_difumo(dimension=N_COMPONENTS, resolution_mm=2)
atlas_reducer = AtlasAggregator(
    difumo,
    masker=features.masker,
    # Bigger batches hold more rows in dense image form at once, and pay
    # nilearn's per-call least-squares setup fewer times.
    batch_size=64,
)

train_reduced = train.fit_transform_maps(atlas_reducer)
test_reduced = test.transform_maps(atlas_reducer)

print(f"Reduced train features: {train_reduced.features.shape}")
print(f"Reduced test features:  {test_reduced.features.shape}")
print(f"First region names: {train_reduced.feature_names[:3]}")

model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
model.fit(train_reduced.features, train_reduced.target)

print(f"DiFuMo holdout accuracy: {model.score(test_reduced.features, test_reduced.target):.3f}")
