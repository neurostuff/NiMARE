.. include:: links.rst

Machine learning with Studysets
===============================

:meth:`~nimare.studyset.Studyset.to_bunch` turns a
:class:`~nimare.nimads.Studyset` into the arrays a scikit-learn workflow
expects: a sparse analysis-by-voxel matrix of modeled activation (MA)
features, an optional target, and the study labels that keep analyses from one
study out of two different partitions.

.. code-block:: python

    from nimare.meta.kernel import MKDAKernel

    bunch = studyset.to_bunch(
        MKDAKernel(r=10),
        target_field=("metadata", "comparison_task"),
    )

:mod:`nimare.ml` holds what a Studyset cannot answer on its own:
:func:`~nimare.ml.describe_fields` reports which of its fields are worth
modelling, :func:`~nimare.ml.make_preprocessor` keeps a reducer off the
descriptor columns, and :class:`~nimare.ml.AtlasAggregator` reduces voxels over
an atlas.

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

What the bundle holds
---------------------

A :class:`~sklearn.utils.Bunch`, which is a dict whose keys are also
attributes, holding ``data`` (sparse while the map features are unreduced),
``target``, ``groups`` (the study each analysis came from), ``ids``,
``feature_names``, ``map_columns``, ``descriptor_columns``,
``descriptor_names``, the ``masker`` the voxels came from, and ``provenance``.

There is no container class and no estimator to configure. Everything after
conversion is scikit-learn working on ordinary arrays: a grouped split is
:class:`~sklearn.model_selection.GroupShuffleSplit` over ``groups``, a subset
is an index into ``data``, ``ids`` and ``target`` together, and a reduction is
a transformer applied to ``data[:, bunch.map_columns]``.

``to_bunch`` is the only ``to_*`` method on a Studyset that computes rather
than reformats: a release of 40,000 analyses takes about half a minute and a
few gigabytes, because it runs a kernel over every analysis. It is also the
only one that needs scikit-learn, which it imports when called rather than at
module load, so :mod:`nimare.studyset` does not depend on it.

Selecting fields
----------------

Descriptor and target fields are named by a bare field name, or by a
``(source, field)`` tuple. The sources are ``"metadata"``, ``"annotations"``
and ``"texts"``. A bare name is looked up in each in turn, and an ambiguous one
asks for the tuple form, so the tuple is for the case the bare name cannot
express rather than a second way to spell the same thing.

Metadata is read so that study-level fields are inherited by their analyses,
and list-valued fields such as ``sample_sizes`` are reduced the way the rest of
NiMARE reduces them.

Annotation labels
~~~~~~~~~~~~~~~~~

An annotation is usually thousands of mostly-empty columns: the bundled
Neurosynth studyset carries 3,228 labels, and the 2026-09 NeuroStore release
annotates 115,748 analyses with 924. Naming them one at a time is not a
workflow, so a field that reads as a glob pattern selects every label matching
it:

.. code-block:: python

    bunch = studyset.to_bunch(
        MKDAKernel(r=10),
        descriptor_fields=[("annotations", "Neurosynth_TFIDF__*")],
    )

Such a selection is read from the Studyset's sparse
:class:`~nimare.studyset.blocks.LabelBlock` and stays sparse through
splitting and export. Labels keep their own names, double underscores and
brackets included.

An exact name always wins over pattern matching, so a label named
``ParticipantDemographicsExtractor.groups[0].BMI`` is selectable even though
its name reads as a glob.

A pattern's brackets are a :mod:`fnmatch` character class, which is not what
they mean in an extractor's repeated fields: ``*groups[0].*`` asks for a name
containing ``groups0``, and matches none of them. A pattern that matches
nothing is therefore retried with its brackets taken literally, so
``*groups[0].*`` selects group zero and nothing else. A pattern that already
matches is left alone, so ``[mt]*`` still means a character class.

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
* A mapping sets that policy per role, because the two roles differ in what
  can be done about a gap. A descriptor gap an imputer in the pipeline can
  fill; a target gap it cannot, and a ``NaN`` target fails inside the estimator
  rather than at extraction:

  .. code-block:: python

      missing_values={"target": "drop", "descriptors": "keep"}

  A role the mapping does not name is ``"raise"``. This is the usual setting
  on a release, where the field being predicted and the fields describing the
  analyses are reported by different subsets of it.
* Categorical and text fields are refused as descriptors, because the feature
  matrix is numeric and encoding them during extraction would fit the encoder
  on the analyses about to be held out. Their raw values are in the Studyset's
  own tables; encode them and select the numeric result.
* A target that says the same thing about every analysis that was kept is
  refused, since there is nothing to predict. The check runs after
  ``missing_coordinates``, because the minority class can leave with the
  analyses that had no coordinates.

Finding the fields worth using
------------------------------

A release-scale Studyset offers more fields than anyone can read. The 2026-09
NeuroStore release has 76 metadata columns and 924 annotation labels, and 875
of the 997 are reported by fewer than one analysis in a hundred --
``control_sampeslize`` and ``young sampel size`` among them.
:func:`~nimare.ml.describe_fields` reports what each one holds, using the same
reader :meth:`~nimare.studyset.Studyset.to_bunch` uses, so a field it calls
numeric is numeric there:

.. code-block:: python

    from nimare.extract import fetch_neurostore
    from nimare.ml import describe_fields

    studyset = fetch_neurostore()
    fields = describe_fields(studyset, min_coverage=0.5)

    fields[fields.n_unique.between(2, 12)]      # classification targets
    fields[fields.kind == "numeric"]            # descriptors and regression targets

It returns a :class:`~pandas.DataFrame` of ``source``, ``field``, ``kind``,
``coverage``, ``n_unique`` and ``example``, ordered by coverage, so picking a
field is a query rather than a search. On the full release it takes about
thirteen seconds, and cuts 997 fields to the 29 reported by at least half the
analyses.

This is also how to answer a pattern that spans both kinds. ``*groups[0].*``
names six categorical labels as well as the numeric ones, and a categorical
label cannot go into a numeric feature matrix, so the selection is refused.
The ``field`` column, filtered to ``kind == "numeric"``, is the
``descriptor_fields`` list that was meant:

.. code-block:: python

    numeric = fields[fields.kind == "numeric"].field
    bunch = studyset.to_bunch(
        MKDAKernel(r=10),
        descriptor_fields=[("annotations", name) for name in numeric],
        missing_values="keep",
    )

Selecting rows and splitting
----------------------------

Subsetting a bundle is indexing, done to every aligned field together. Select
rows on the Studyset when you can -- :meth:`~nimare.studyset.Studyset.slice`
takes analysis ids and :meth:`~nimare.studyset.Studyset.select_analyses` takes
a mask or positions -- because conversion is the expensive step and it is
cheaper to run it once on the rows being modelled.

For a holdout, ``test_size`` adds ``train`` and ``test`` row positions to the
bundle:

.. code-block:: python

    bunch = studyset.to_bunch(MKDAKernel(r=10), test_size=0.25, random_state=13)

    X_train, y_train = bunch.data[bunch.train], bunch.target[bunch.train]

It is a :class:`~sklearn.model_selection.GroupShuffleSplit` over ``groups``, so
a study belongs to exactly one partition. That grouping is the point: on the
bundled Studyset a plain :func:`~sklearn.model_selection.train_test_split`
puts 112 of its 320 studies on both sides at once, and nothing about the
resulting score says so. ``test_size`` counts *studies*, so analysis counts
only approximate a fraction.

Without ``test_size`` the two keys are absent and the bundle is exactly as it
was. The split costs milliseconds where the conversion runs a kernel, so for
several splits of one bundle -- or for cross-validation, which needs no split
at all -- hand ``groups`` to a group splitter rather than converting again:

.. code-block:: python

    cross_val_score(pipeline, bunch.data, bunch.target,
                    cv=GroupKFold(5), groups=bunch.groups)

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
prevent. :func:`~nimare.ml.make_preprocessor` builds one with the
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

Outside a pipeline, fit the reducer on the training rows and apply the same
fitted reducer to the held-out ones. The bundle's ``masker`` is what an atlas
reducer needs and cannot work out for itself:

.. code-block:: python

    maps = bunch.data[:, bunch.map_columns]
    reducer = AtlasAggregator(atlas, masker=bunch.masker)

    train_reduced = reducer.fit_transform(maps[train])
    test_reduced = reducer.transform(maps[test])

Calling ``fit_transform`` on the held-out rows would fit the reduction on the
analyses being held out.

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
through :func:`~nimare.ml.make_preprocessor` -- and only an explicit
reducer produces a dense representation.

Conversion is linear in analyses, and an MA row is denser than a Studyset row:
against the 2026-09 NeuroStore release at a 10 mm MKDA radius, measured on a
2 mm whole-brain mask,

=================  ========  ===========  =============
Analyses           Time      Non-zeros    Peak memory
=================  ========  ===========  =============
500                  3.8 s     1,350,169        63 MB
2,000                3.2 s     6,045,930       105 MB
8,000               10.2 s    33,437,983       549 MB
20,000              27.7 s    91,017,001      1,483 MB
=================  ========  ===========  =============

Extrapolating the observed 4,700 non-zeros per row, all 115,748 analyses would
be roughly 6 GB of sparse data and about twice that at peak, so the whole
release does not convert on a 16 GB machine. Take the part being modelled
first -- :meth:`~nimare.nimads.Studyset.slice` selects analyses by id, and
``describe_fields`` says which ones carry the field of interest -- and pass
``memory=`` so a second pass over the same analyses reuses the maps it already
generated.

.. seealso::

    :ref:`machine_learning_in_nimare` walks through the whole workflow.
