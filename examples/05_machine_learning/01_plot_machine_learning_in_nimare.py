"""
.. _machine_learning_in_nimare:

============================
Machine learning in NiMARE
============================

Turn a Studyset into a scikit-learn dataset, split it without leaking a study,
and reduce the voxelwise features before fitting a model.

:meth:`~nimare.studyset.Studyset.to_bunch` returns a
:class:`~sklearn.utils.Bunch` of modeled activation (MA) features, the study
each analysis came from, and an optional target, all in one row order. NiMARE
builds it; scikit-learn does the splitting, fitting and scoring, on ordinary
arrays it already knows how to handle.
"""

from pathlib import Path

import numpy as np
from nilearn.datasets import fetch_atlas_difumo
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import SparseRandomProjection

from nimare.extract import fetch_neurostore
from nimare.meta.kernel import MKDAKernel
from nimare.ml import AtlasAggregator, describe_fields, make_preprocessor
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
# :meth:`~nimare.studyset.Studyset.to_bunch` applies an MKDA kernel to the
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
bunch = studyset.to_bunch(
    MKDAKernel(r=10),
    target_field=("metadata", "comparison_task"),
)

print(f"Feature data: {bunch.data.shape}, sparse={bunch.data.format}")
print(f"Labels: {sorted(set(bunch.target))}")
print(f"Dropped for want of coordinates: {len(bunch.provenance['dropped_ids'])}")

###############################################################################
# The bundle is the familiar scikit-learn one: sparse ``data``, aligned
# ``target``, and ``groups`` holding the study each analysis came from. It also
# carries ``ids``, ``feature_names``, ``map_columns``, ``descriptor_columns``,
# ``descriptor_names``, the ``masker`` the voxels came from, and ``provenance``.
print(f"Bundle: {', '.join(sorted(bunch))}")

###############################################################################
# Split without leaking a study
# -----------------------------------------------------------------------------
# Analyses from one study are related, so a study belongs to exactly one
# partition. ``groups`` is what makes that happen, and any scikit-learn group
# splitter takes it; ``test_size`` is a fraction of *studies*, so the analysis
# counts only approximate it.
train, test = next(
    GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED).split(
        bunch.data, bunch.target, bunch.groups
    )
)

print(f"Train: {len(train)} analyses from {len(set(bunch.groups[train]))} studies")
print(f"Test:  {len(test)} analyses from {len(set(bunch.groups[test]))} studies")
print(f"Shared studies: {set(bunch.groups[train]) & set(bunch.groups[test])}")

###############################################################################
# Classify the task label
# -----------------------------------------------------------------------------
# ``bunch.data`` is an ordinary sparse matrix, so an ordinary scikit-learn
# pipeline works on it. The voxelwise MA features are high-dimensional, so
# truncated SVD reduces them before the classifier sees them; putting the
# reducer in the pipeline is what keeps it fitted on training rows only.
# GroupKFold reads the same study labels the split used.
#
# Every column here is a voxel -- ``bunch.descriptor_names`` is empty -- so the
# reducer can see the whole matrix. The section after next adds descriptor
# columns, which a bare reducer would decompose along with the voxels.
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
    studyset.to_bunch(MKDAKernel(r=10), descriptor_fields=["sample_sizes"])
except ValueError as exc:
    print(f"{str(exc)[:160]}...")

###############################################################################
# Once there are descriptor columns, the reducer has to be kept off them, which
# is what :class:`~sklearn.compose.ColumnTransformer` is for.
# :func:`~nimare.ml.make_preprocessor` builds one with the column boundary read
# off the bundle and ``sparse_threshold=1.0``, so a wide sparse map block is
# never quietly densified. ``bunch.map_columns`` and
# ``bunch.descriptor_columns`` are right there if you would rather write it
# out. With map features alone, as above, there is nothing to keep the reducer
# away from and the function hands the reducer straight back.
#
# One transformer covers every descriptor. When they want different treatment,
# pass a mapping instead -- ``{"sample_sizes": SimpleImputer(), "year":
# StandardScaler()}`` -- and the descriptors it does not name are passed
# through, in the order they came in. Descriptor columns are handed over dense,
# which is what most transformers expect of a few numeric columns.
with_descriptors = studyset.to_bunch(
    MKDAKernel(r=10),
    target_field=("metadata", "comparison_task"),
    descriptor_fields=["sample_sizes"],
    missing_values="keep",
)
preprocessor = make_preprocessor(
    with_descriptors,
    TruncatedSVD(n_components=50, random_state=RANDOM_SEED),
    descriptor_transformer=SimpleImputer(strategy="median"),
)

print(f"Descriptors: {with_descriptors.descriptor_names}")
print(f"Descriptor columns: {with_descriptors.descriptor_columns}")
print(f"With descriptors: {type(preprocessor).__name__}")
print(f"Map features only: {type(make_preprocessor(bunch, TruncatedSVD(2))).__name__}")

###############################################################################
# Select annotation labels
# -----------------------------------------------------------------------------
# An annotation is thousands of mostly-empty columns -- the Neurosynth release
# annotates 115,747 analyses with 794 labels -- so naming them one at a time is
# not a workflow. A glob pattern takes them all, each under its own name, and
# reads them from the Studyset's sparse label block rather than densifying
# them. A label no analysis carries is a zero rather than a gap, so
# ``missing_values`` has nothing to report about a pattern selection.
neurosynth = Studyset(str(Path(get_resource_path()) / "neurosynth_laird_studyset.json"))
annotated = neurosynth.to_bunch(
    MKDAKernel(r=10),
    descriptor_fields=[("annotations", "Neurosynth_TFIDF__*")],
)

labels = annotated.data[:, annotated.descriptor_columns]
print(f"Bundle: {annotated.data.shape}, of which labels: {labels.shape[1]}")
print(f"Non-zero labels: {labels.nnz}")
print(f"Names kept whole: {annotated.descriptor_names[:2]}")
print(f"Still sparse: {sparse.issparse(annotated.data)}")

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
# The bundle carries the ``masker`` its voxels came from, which is the one
# thing an atlas reducer cannot work out for itself.
#
# Outside a pipeline, fit the reducer on the training rows and apply the same
# fitted reducer to the held-out ones -- calling ``fit_transform`` on the test
# rows would use the analyses you are holding out.
difumo = fetch_atlas_difumo(dimension=N_COMPONENTS, resolution_mm=2)
atlas_reducer = AtlasAggregator(difumo, masker=bunch.masker)

maps = bunch.data[:, bunch.map_columns]
train_reduced = atlas_reducer.fit_transform(maps[train])
test_reduced = atlas_reducer.transform(maps[test])

print(f"Reduced train features: {train_reduced.shape}")
print(f"Reduced test features:  {test_reduced.shape}")
# get_feature_names_out follows scikit-learn and returns an array of numpy
# strings; tolist() gives the plain ones.
print(f"First region names: {atlas_reducer.get_feature_names_out()[:3].tolist()}")

model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
model.fit(train_reduced, bunch.target[train])

print(f"DiFuMo holdout accuracy: {model.score(test_reduced, bunch.target[test]):.3f}")

###############################################################################
# Work with a release-scale Studyset
# -----------------------------------------------------------------------------
# :func:`~nimare.extract.fetch_neurostore` downloads a published NeuroStore
# release: 115,748 analyses from 32,444 studies, annotated with 924 labels by
# LLM extractors. Nothing below is specific to that release -- the same three
# steps work on any Studyset large enough that reading its field list is not
# an option.
studyset = fetch_neurostore(version="2026-09")

print(f"Analyses: {len(studyset.ids)} from {len(set(studyset.study_ids))} studies")

###############################################################################
# Ask what is usable, rather than reading 997 field names
# -----------------------------------------------------------------------------
# :func:`~nimare.ml.describe_fields` reports every field a selector may name,
# with the kind :meth:`~nimare.studyset.Studyset.to_bunch` will read it as and
# the fraction of analyses reporting it. Most of a release is a long tail that
# no analysis fills in, so picking a field becomes a query.
all_fields = describe_fields(studyset)
fields = all_fields[all_fields.coverage >= 0.5]
targets = fields[fields.n_unique.between(2, 12)]

print(f"Fields at >=50% coverage: {len(fields)} of {len(all_fields)}")
print(targets[["source", "field", "kind", "coverage", "n_unique"]].to_string(index=False))

###############################################################################
# Convert the part you are modelling
# -----------------------------------------------------------------------------
# An MA row is denser than a Studyset row -- about 4,700 voxels at a 10 mm
# radius -- so the whole release is roughly 6 GB of sparse data and does not
# convert on a 16 GB machine. :meth:`~nimare.nimads.Studyset.slice` takes
# analysis ids, so take the part being modelled first. ``missing_values="drop"``
# removes the analyses the extractor could not fill in.
subset = studyset.slice(analyses=list(studyset.ids)[:4000])
resting = subset.to_bunch(
    MKDAKernel(r=10),
    target_field=("annotations", "TaskExtractor.fMRITasks[0].RestingState"),
    missing_values="drop",
)

print(f"Kept {resting.data.shape[0]} of {len(subset.ids)} analyses")
print(f"Resting-state rows: {int(np.sum(np.asarray(resting.target) == 1.0))}")

###############################################################################
# An extractor's repeated fields are indexed, and a bracket is a glob character
# class, so ``*groups[0].*`` would ordinarily match nothing. A pattern that
# matches nothing is retried with its brackets taken literally, so it selects
# group zero as intended. That group mixes numeric and categorical labels, and
# only numbers go into a feature matrix, so the ``field`` column filtered to
# ``kind == "numeric"`` is the descriptor list that was meant.
demographics = fields[
    (fields.kind == "numeric") & fields.field.str.contains("groups[0].", regex=False)
]
with_demographics = subset.to_bunch(
    MKDAKernel(r=10),
    target_field=("annotations", "TaskExtractor.fMRITasks[0].RestingState"),
    descriptor_fields=[("annotations", name) for name in demographics.field],
    # A descriptor gap an imputer can fill; a target gap it cannot, so the two
    # roles get different policies.
    missing_values={"target": "drop", "descriptors": "keep"},
)

print(f"Bundle: {with_demographics.data.shape}")
print(f"Descriptors: {[name.split('.')[-1] for name in with_demographics.descriptor_names]}")

###############################################################################
# Classify resting-state against task
# -----------------------------------------------------------------------------
# From here it is the workflow above: a grouped split so no study spans both
# sides, and an imputer for the descriptor gaps the ``"descriptors": "keep"``
# policy left for the pipeline.
#
# Only one analysis in seven is resting-state, so accuracy would reward always
# answering "task"; ``roc_auc`` asks the question actually being put, which is
# whether the foci rank a resting-state analysis above a task one.
release_pipeline = make_pipeline(
    make_preprocessor(
        with_demographics,
        TruncatedSVD(n_components=50, random_state=RANDOM_SEED),
        descriptor_transformer=SimpleImputer(strategy="median"),
    ),
    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
)
release_bunch = with_demographics
release_scores = cross_val_score(
    release_pipeline,
    release_bunch.data,
    release_bunch.target,
    cv=GroupKFold(5),
    groups=release_bunch.groups,
    scoring="roc_auc",
)

n_resting = int(np.sum(release_bunch.target == 1.0))

print(f"Resting-state rows: {n_resting} of {len(release_bunch.target)}")
print(f"Resting-state ROC AUC: {release_scores.mean():.3f} +/- {release_scores.std():.3f}")
