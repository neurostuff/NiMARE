.. include:: links.rst

Machine learning with Studysets
===============================

:mod:`nimare.ml` turns a :class:`~nimare.nimads.Studyset` into the arrays a
scikit-learn workflow expects: a sparse analysis-by-voxel matrix of modeled
activation (MA) features, an optional target, and the study labels that keep
analyses from one study out of two different partitions.

.. code-block:: python

    from nimare.meta.kernel import MKDAKernel
    from nimare.ml import FeatureSet

    features = FeatureSet.from_studyset(
        studyset,
        kernel_transformer=MKDAKernel(r=10),
        target_field=("metadata", "comparison_task"),
    )
    bunch = features.to_sklearn()

Who does what
-------------

NiMARE owns *extraction*: reading the Studyset, generating MA maps through a
kernel transformer, matching every row to its analysis, and reporting what is
missing. scikit-learn owns *evaluation*: splitting, fitting, reducing and
scoring.

Generating the maps before splitting does not leak. An analysis's MA map is a
function of that analysis's own foci, so it never sees the target or another
analysis. Everything that learns *across* rows -- decomposition, feature
selection, imputation, scaling -- must be fitted on training rows only, which
is what a :class:`~sklearn.pipeline.Pipeline` is for.

Map rows are matched to analyses by analysis id. Kernel transformers return
one row per analysis that has coordinates, ordered by id, and
``return_type="sparse"`` drops the ids that name them, so pairing the two by
position is only correct while the Studyset happens to be in sorted order --
which :meth:`~nimare.studyset.Studyset.select_analyses` does not guarantee.

Why a named constructor
-----------------------

:meth:`~nimare.ml.FeatureSet.from_studyset` is a classmethod rather than
``__init__`` because the container is also built from blocks that already
exist, by :meth:`~nimare.ml.FeatureSet.split`,
:meth:`~nimare.ml.FeatureSet.select_analyses`,
:meth:`~nimare.ml.FeatureSet.copy` and the map-reduction methods. A
constructor that ran a kernel would push all of those onto a back door, and
would make a twenty-second kernel run look like a cheap wrap. ``__init__``
takes the blocks directly, which is also what to use for features generated
some other way.

Selecting fields
----------------

Descriptor and target fields are named by a bare field name, by a
``(source, field)`` tuple, or by a mapping. The sources are ``"metadata"``,
``"annotations"`` and ``"texts"``. A bare name is looked up in each in turn,
and an ambiguous one asks for the tuple form.

Metadata is read so that study-level fields are inherited by their analyses,
and list-valued fields such as ``sample_sizes`` are reduced the way the rest of
NiMARE reduces them.

Annotation labels
~~~~~~~~~~~~~~~~~

An annotation is usually thousands of mostly-empty columns: the bundled
Neurosynth studyset carries 3,228 labels, and the NeuroStore release annotates
115,747 analyses with 794. Naming them one at a time is not a workflow, so a
field that reads as a glob pattern selects every label matching it:

.. code-block:: python

    features = FeatureSet.from_studyset(
        studyset,
        kernel_transformer=MKDAKernel(r=10),
        descriptor_fields=[("annotations", "Neurosynth_TFIDF__*")],
    )

Such a selection is read from the Studyset's sparse
:class:`~nimare.studyset.blocks.LabelBlock` and stays sparse through
splitting and export. Labels keep their own names, double underscores and
brackets included.

An exact name always wins over pattern matching, so a label named
``ParticipantDemographicsExtractor.groups[0].BMI`` is selectable even though
its name reads as a glob.

The two forms mean different things about absence, deliberately:

* a **pattern** gives a label matrix, where a label no analysis carries is a
  zero rather than a gap, which is what a sparse annotation means;
* an **exact name** gives a value, where an absent value is missing and
  ``missing_values`` applies to it.

Missing and unusable data
-------------------------

Nothing is filled in silently.

* ``missing_coordinates`` decides what happens to analyses that report no
  coordinates: ``"drop"`` (the default) removes them before rows are built and
  records their ids in provenance, ``"include"`` keeps them as all-zero sparse
  map rows.
* ``missing_values`` decides what happens to a selected descriptor or target
  value that an analysis does not report: ``"raise"`` (the default) names the
  fields and the analyses, ``"drop"`` removes those analyses, ``"keep"`` leaves
  the gaps for an imputer in a pipeline. It speaks only for analyses that
  survive ``missing_coordinates``.
* Categorical and text fields are refused as descriptors, because the feature
  matrix is numeric and encoding them during extraction would fit the encoder
  on the analyses about to be held out. Their raw values are in the Studyset's
  own tables; encode them and select the numeric result.
* A target that says the same thing about every analysis that was kept is
  refused, since there is nothing to predict. The check runs after
  ``missing_coordinates``, because the minority class can leave with the
  analyses that had no coordinates.

Splitting without leaking a study
---------------------------------

:meth:`~nimare.ml.FeatureSet.split` is a grouped holdout over
:class:`~sklearn.model_selection.GroupShuffleSplit`, and ``test_size`` is a
fraction of *studies*, so analysis counts only approximate it. For
cross-validation, hand ``bunch.groups`` to any scikit-learn group splitter.

Reducing the voxel features
---------------------------

Map features are an ordinary sparse matrix, so ordinary scikit-learn
transformers reduce them, imported from scikit-learn and used as scikit-learn
documents them:

.. code-block:: python

    from sklearn.decomposition import TruncatedSVD

    pipeline = make_pipeline(TruncatedSVD(n_components=50), LogisticRegression())

Anything that reads sparse input works: truncated SVD, sparse random
projection, variance thresholding. Dense PCA will ask for dense data.

Keeping a reducer off the descriptor columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A transformer placed directly in a pipeline sees every column it is given. With
map features alone that is the whole matrix and nothing else is needed. Once
there are descriptor columns, a bare reducer would decompose them along with
the voxels, which is what :class:`~sklearn.compose.ColumnTransformer` exists to
prevent. :meth:`~nimare.ml.FeatureSet.make_preprocessor` builds one with the
column boundary filled in, the masker bound into an atlas reducer, and
``sparse_threshold=1.0`` -- scikit-learn's default of ``0.3`` would densify a
map block above 30% density, about 1.6 GB at 228,000 columns:

.. code-block:: python

    preprocessor = features.make_preprocessor(
        TruncatedSVD(n_components=50),
        descriptor_transformer={
            "sample_sizes": SimpleImputer(strategy="median"),
            "year": StandardScaler(),
        },
    )

Descriptors the mapping does not name are passed through, and the columns come
out in the order they went in. Transformers are handed the descriptor columns
dense, which is what most of them expect of a few numeric columns; the map
block stays sparse.

``features.map_columns``, ``features.descriptor_columns`` and
``features.descriptor_names`` are public, so the same thing can be written out
by hand. With no descriptor columns, ``make_preprocessor`` hands the reducer
straight back.

Atlas aggregation
~~~~~~~~~~~~~~~~~

:class:`~nimare.ml.AtlasAggregator` is the one reducer NiMARE adds, because it
is the one that has to know which voxel each column is. It takes any atlas
nilearn can load: what a ``fetch_atlas_*`` function returns, an atlas image or
file, the name of a fetcher with its arguments, or a masker configured by the
caller. A 4D atlas is summarised with a
:class:`~nilearn.maskers.NiftiMapsMasker` and a 3D one with a
:class:`~nilearn.maskers.NiftiLabelsMasker`, and the atlas's own region names
become the feature names.

How many regions an atlas yields is nilearn's answer, not NiMARE's, and it
differs by version: a region falling outside the mask is kept by nilearn 0.12
and dropped by 0.13. A feature matrix built this way is comparable across
environments only when the nilearn version is.

Outside a pipeline, fit the reducer on the training feature set with
:meth:`~nimare.ml.FeatureSet.fit_transform_maps` and hand the same fitted
reducer to :meth:`~nimare.ml.FeatureSet.transform_maps` for the held-out one.
Passing an unfitted reducer to the latter raises, because fitting it there
would use the held-out analyses.

Rows are converted back into images in batches, because nilearn's maskers
aggregate an image rather than a row. Most of the cost is per call rather than
per row, so ``batch_size`` trades memory for speed steeply at first and then
hardly at all: against a 2 mm whole-brain mask, a batch of 8 costs 436 ms per
row, 32 costs 178 ms, and 128 costs 141 ms for four times the dense working
set. The default of 32 sits at the knee, at roughly 60 MB.

Scale
-----

A Studyset of 1,000 studies converts and splits in well under the budget the
feature was designed to: roughly a second, and under a gigabyte of peak memory,
against a target of three minutes and five gigabytes. Unreduced voxelwise
features are sparse everywhere -- in the container, in the exported bundle, and
through :meth:`~nimare.ml.FeatureSet.make_preprocessor` -- and only an explicit
reducer produces a dense representation.

.. seealso::

    :ref:`machine_learning_in_nimare` walks through the whole workflow.
