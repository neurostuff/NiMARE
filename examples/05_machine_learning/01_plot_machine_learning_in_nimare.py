"""
.. _machine_learning_in_nimare:

============================
Machine learning in NiMARE
============================

Turn a Studyset into a scikit-learn dataset, split it without leaking a study,
and build the voxelwise features inside a pipeline.

:meth:`~nimare.studyset.Studyset.to_bunch` returns a
:class:`~sklearn.utils.Bunch` of the peaks each analysis reported, the study it
came from, and an optional target, all in one row order. NiMARE reads the
Studyset; scikit-learn does everything that transforms it -- the modeled
activation (MA) kernel included, as :class:`~nimare.ml.MAKernel`.
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
from sklearn.preprocessing import OneHotEncoder
from sklearn.random_projection import SparseRandomProjection

from nimare.extract import fetch_neurostore
from nimare.meta.kernel import MKDAKernel
from nimare.ml import (
    MAKernel,
    MaskerTransformer,
    describe_fields,
    make_nimare_column_transformer,
)
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
# :meth:`~nimare.studyset.Studyset.to_bunch` reads each analysis's foci into a
# row of peak counts over the image grid, and reads the ``comparison_task``
# metadata field as the ``"n-back"`` and ``"flanker"`` labels to predict.
# Fields are named by a bare field name, or by a ``(source, field)`` pair when
# the name appears in more than one Studyset table.
#
# No kernel is named here. Convolving peaks into MA maps is a modelling choice,
# and one choice among several -- peak counts per parcel are a feature set too
# -- so it belongs with the others, in the pipeline.
bunch = studyset.to_bunch(
    target_field=("metadata", "comparison_task"),
    # Hold out a quarter of the studies; see the next section.
    test_size=0.25,
    random_state=RANDOM_SEED,
)

print(f"Feature data: {bunch.data.shape}, sparse={bunch.data.format}")
print(f"Non-zeros: {bunch.data.nnz:,} peaks")
print(f"Labels: {sorted(set(bunch.target))}")
print(f"Dropped for want of coordinates: {len(bunch.provenance['dropped_ids'])}")

###############################################################################
# The bundle is the familiar scikit-learn one: sparse ``data``, aligned
# ``target``, and ``groups`` holding the study each analysis came from. It also
# carries ``ids``, ``feature_names``, ``voxel_columns``, ``descriptor_columns``,
# ``descriptor_names``, the ``masker`` whose grid the voxels span, and
# ``provenance``.
#
# The voxel columns span the whole image grid, not just the mask, because a
# coordinate outside the mask still reaches into it once a kernel spreads it.
print(f"Bundle: {', '.join(sorted(bunch))}")

###############################################################################
# Split without leaking a study
# -----------------------------------------------------------------------------
# Analyses from one study are related, so a study belongs to exactly one
# partition. Passing ``test_size`` above added ``train`` and ``test`` row
# positions to the bundle, grouped by study, which is what keeps a plain
# shuffle from putting the same study on both sides. ``test_size`` counts
# *studies*, so the analysis counts only approximate the fraction.
train, test = bunch.train, bunch.test

print(f"Train: {len(train)} analyses from {len(set(bunch.groups[train]))} studies")
print(f"Test:  {len(test)} analyses from {len(set(bunch.groups[test]))} studies")
print(f"Shared studies: {set(bunch.groups[train]) & set(bunch.groups[test])}")

###############################################################################
# The split is a :class:`~sklearn.model_selection.GroupShuffleSplit` over
# ``groups``. For several splits of one bundle, or for cross-validation, hand
# ``groups`` to a group splitter rather than converting again.

###############################################################################
# Classify the task label
# -----------------------------------------------------------------------------
# ``bunch.data`` is an ordinary sparse matrix, so an ordinary scikit-learn
# pipeline works on it. :class:`~nimare.ml.MAKernel` convolves the peaks into
# MA maps as the first step, truncated SVD reduces them before the classifier
# sees them, and both are fitted on training rows only because they are in the
# pipeline. GroupKFold reads the same study labels the split used.
#
# Every column here is a voxel -- ``bunch.descriptor_names`` is empty -- so the
# steps can see the whole matrix. The section after next adds descriptor
# columns, which a bare reducer would decompose along with the voxels.
pipeline = make_pipeline(
    MAKernel(MKDAKernel(r=10), source_masker=bunch.masker),
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
    studyset.to_bunch(descriptor_fields=["sample_sizes"])
except ValueError as exc:
    print(f"{str(exc)[:160]}...")

###############################################################################
# Once there are descriptor columns, the reducer has to be kept off them, which
# is what :class:`~sklearn.compose.ColumnTransformer` is for.
# :func:`~nimare.ml.make_nimare_column_transformer` is
# :func:`~sklearn.compose.make_column_transformer` with the bundle's blocks
# filled in: ``(transformer, columns)`` pairs as scikit-learn takes them, where
# ``columns`` may be ``"voxels"``, ``"descriptors"``, or a descriptor's own
# field name. It also binds the bundle's masker into an atlas reducer, keeps
# the column names so a coefficient can be read back, and defaults
# ``sparse_threshold`` to 1.0 -- scikit-learn's 0.3 would densify a wide voxel
# block.
#
# A transformer per descriptor is just another pair:
# ``(SimpleImputer(), "sample_sizes"), (StandardScaler(), "year")``. Anything
# this does not cover is written out with ``bunch.voxel_columns`` and
# ``bunch.descriptor_columns``, which are ordinary slices. With voxel features
# alone there is nothing to keep the reducer away from, so none of this is
# needed.
with_descriptors = studyset.to_bunch(
    target_field=("metadata", "comparison_task"),
    descriptor_fields=["sample_sizes"],
    missing_values="keep",
)
preprocessor = make_nimare_column_transformer(
    with_descriptors,
    (
        make_pipeline(
            MAKernel(MKDAKernel(r=10), source_masker=with_descriptors.masker),
            TruncatedSVD(n_components=50, random_state=RANDOM_SEED),
        ),
        "voxels",
    ),
    (SimpleImputer(strategy="median"), "descriptors"),
)

print(f"Descriptors: {with_descriptors.descriptor_names}")
print(f"Descriptor columns: {with_descriptors.descriptor_columns}")
print(f"With descriptors: {type(preprocessor).__name__}")
print(f"Steps: {[name for name, _, _ in preprocessor.transformers]}")

###############################################################################
# Use a categorical field as a feature
# -----------------------------------------------------------------------------
# A feature matrix is numeric, so a string cannot be a column of it. What goes
# in is the *position* of a category, and ``descriptor_categories`` says what
# the positions mean -- a representation rather than an encoding, so the choice
# of encoder stays in the pipeline with every other transformation.
#
# The bundle fills in the two things scikit-learn cannot work out from an array
# of numbers: ``categories=``, so a training split that happens to miss a
# category still yields the same number of columns, and the real labels in
# ``get_feature_names_out``. A code cannot be passed through or scaled, because
# it stands for a label rather than a quantity.
GROUP = "ParticipantDemographicsExtractor.groups[0].group_name"
with_group = studyset.to_bunch(
    target_field=("metadata", "comparison_task"),
    descriptor_fields=[("annotations", GROUP)],
    missing_values={"target": "drop", "descriptors": "keep"},
)
encoded = make_nimare_column_transformer(
    with_group,
    ("drop", "voxels"),
    (OneHotEncoder(handle_unknown="ignore"), GROUP),
).fit(with_group.data)

print(f"Categories: {with_group.descriptor_categories}")
print(f"Encoded as: {[n.split('.')[-1] for n in encoded.get_feature_names_out()]}")

try:
    make_nimare_column_transformer(with_group, ("drop", "voxels"), ("passthrough", GROUP))
except ValueError as exc:
    print(f"{str(exc)[:150]}...")

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
# Any scikit-learn transformer will do, downstream of the kernel. They see the
# sparse MA matrix, so they have to accept sparse input: truncated SVD, sparse
# random projection, variance thresholding and atlas aggregation all do.
# ``PCA`` accepts sparse input as well, but only through its ``arpack`` or
# ``covariance_eigh`` solvers, and it centres the data, which is why truncated
# SVD is the usual choice for a matrix this wide.
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
        MAKernel(MKDAKernel(r=10), source_masker=bunch.masker),
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
# Let nilearn transform the voxels
# -----------------------------------------------------------------------------
# A nilearn masker is already a scikit-learn transformer, but it takes images
# where a ColumnTransformer hands out columns of an array.
# :class:`~nimare.ml.MaskerTransformer` is that bridge, and the one transformer
# NiMARE adds, because it is the one that has to know which voxel each column
# is. The bundle's ``masker`` says that, which is why it travels with the data.
#
# What it applies is any nilearn masker, or anything nilearn loads as an atlas:
# what a ``fetch_atlas_*`` function returns, an atlas image or file, or the name
# of a fetcher such as ``"harvard_oxford"``. A 4D atlas is summarised with a
# ``NiftiMapsMasker`` and a 3D one with a ``NiftiLabelsMasker``, and the atlas's
# own region names become the feature names. A
# :class:`~nilearn.maskers.NiftiMasker` returns voxels instead, which is how
# nilearn's smoothing and standardizing reach these features::
#
#     (NiftiMasker(smoothing_fwhm=6), "voxels")
#
# It reads either column space, deciding from the width: the bundle's raw peak
# columns, which gives the number of reported coordinates per region, or the MA
# maps a kernel made, as here.
#
# Outside a pipeline, fit the transformer on the training rows and apply the
# same fitted one to the held-out rows -- calling ``fit_transform`` on the test
# rows would use the analyses you are holding out.
difumo = fetch_atlas_difumo(dimension=N_COMPONENTS, resolution_mm=2)
atlas_reducer = MaskerTransformer(difumo, source_masker=bunch.masker)

maps = MAKernel(MKDAKernel(r=10), source_masker=bunch.masker).fit_transform(
    bunch.data[:, bunch.voxel_columns]
)
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
# The whole release converts in under two seconds now that conversion reads
# peaks rather than making maps -- 115,748 analyses and 852,973 non-zeros. What
# is still expensive is the kernel: an MA row is about 4,700 voxels at a 10 mm
# radius, so fitting over the whole release would be. :meth:`~nimare.nimads.Studyset.slice`
# takes analysis ids, so take the part being modelled first.
# ``missing_values="drop"`` removes the analyses the extractor could not fill
# in.
subset = studyset.slice(analyses=list(studyset.ids)[:4000])
resting = subset.to_bunch(
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
    make_nimare_column_transformer(
        with_demographics,
        (
            make_pipeline(
                MAKernel(MKDAKernel(r=10), source_masker=with_demographics.masker),
                TruncatedSVD(n_components=50, random_state=RANDOM_SEED),
            ),
            "voxels",
        ),
        (SimpleImputer(strategy="median"), "descriptors"),
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
