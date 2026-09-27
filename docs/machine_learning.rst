.. include:: links.rst

Machine learning with Studysets
===============================

:meth:`~nimare.studyset.Studyset.to_bunch` turns a
:class:`~nimare.nimads.Studyset` into the arrays a scikit-learn workflow
expects: a sparse analysis-by-voxel matrix of peak counts, an optional target,
and the study labels that keep analyses from one study out of two different
partitions.

.. code-block:: python

    bunch = studyset.to_bunch(target_field=("metadata", "comparison_task"))

:mod:`nimare.ml` holds what a Studyset cannot answer on its own:
:func:`~nimare.ml.describe_fields` reports which of its fields are worth
modelling, :class:`~nimare.ml.MAKernel` turns the peaks into modeled activation
(MA) maps, :func:`~nimare.ml.make_nimare_column_transformer` routes each block
to its own transformer, and :class:`~nimare.ml.MaskerTransformer` lets a nilearn
masker act on the voxel columns.

Who does what
-------------

NiMARE owns *reading*: what each analysis reported, matched to the analysis it
came from, with what is missing reported rather than guessed. scikit-learn owns
everything that *transforms* it, the kernel included.

That line is why ``to_bunch`` takes no kernel. A kernel is not a property of a
Studyset, it is a modelling choice -- how wide a sphere or a Gaussian stands
for a reported coordinate -- and it is one choice among several, since peak
counts per parcel are a feature set too. Making it a transformer puts every
such choice in one place, where it is fitted, cloned, cached and tuned like any
other:

.. code-block:: python

    from nimare.meta.kernel import MKDAKernel
    from nimare.ml import MAKernel, make_nimare_column_transformer

    pipeline = make_pipeline(
        make_nimare_column_transformer(
            bunch,
            (MAKernel(MKDAKernel(r=10), source_masker=bunch.masker), "voxels"),
        ),
        TruncatedSVD(n_components=64),
        LogisticRegression(),
    )

Convolving before splitting would not have leaked -- an analysis's MA map is a
function of that analysis's own foci, so it never sees the target or another
analysis -- and that is the point: the move buys composition, not correctness.
Everything that learns *across* rows -- decomposition, feature selection,
imputation, scaling -- must still be fitted on training rows only, which is
what a :class:`~sklearn.pipeline.Pipeline` is for.

Why the peaks span the grid
---------------------------

The voxel columns cover the whole image grid of the bundle's ``masker``, not
just the voxels inside its mask. A coordinate outside the mask still reaches
into it once a kernel spreads it, so in the mask's own column space such a
focus has nowhere to be recorded and its contribution is lost. On the bundled
n-back/flanker studyset that is 266 foci in 112 of 906 analyses, which changes
99 rows -- silently, since every row still has a plausible map.

Spanning the grid costs almost nothing, because peaks are far sparser than the
maps they generate: 9,359 nonzeros against 3,889,276 for the same studyset at a
10 mm radius. Converting a release is no longer the memory wall it was; the
expansion happens per fold, inside the pipeline, on a training subset.

:class:`~nimare.ml.MAKernel` takes grid columns in and returns the masker's
voxels, which is the space :class:`~nimare.ml.MaskerTransformer` and nilearn
expect. ``MaskerTransformer`` reads either, deciding from the width, so an
atlas can summarise the raw peaks or the maps a kernel made.

What the bundle holds
---------------------

A :class:`~sklearn.utils.Bunch`, which is a dict whose keys are also
attributes, holding ``data`` (sparse), ``target``, ``groups`` (the study each
analysis came from), ``ids``, ``feature_names``, ``voxel_columns``,
``descriptor_columns``, ``descriptor_names``, the ``masker`` whose grid the
voxels span, and ``provenance``.

There is no container class and no estimator to configure. Everything after
conversion is scikit-learn working on ordinary arrays: a grouped split is
:class:`~sklearn.model_selection.GroupShuffleSplit` over ``groups``, a subset
is an index into ``data``, ``ids`` and ``target`` together, and a reduction is
a transformer applied to ``data[:, bunch.voxel_columns]``.

``feature_names`` names a column when asked rather than up front, because a
grid of 902,629 columns would otherwise cost more in strings than the sparse
matrix they describe. It indexes, slices and reports membership like a list.

``to_bunch`` is the only ``to_*`` method on a Studyset that needs
scikit-learn, which it imports when called rather than at module load, so
:mod:`nimare.studyset` does not depend on it.

Turning peaks into MA maps
--------------------------

:class:`~nimare.ml.MAKernel` wraps any NiMARE kernel transformer as a
scikit-learn one. It takes the bundle's grid columns and returns the masker's
voxels, so it goes first among the voxel steps:

.. code-block:: python

    make_nimare_column_transformer(
        bunch,
        (
            make_pipeline(
                MAKernel(MKDAKernel(r=10), source_masker=bunch.masker),
                TruncatedSVD(n_components=64),
            ),
            "voxels",
        ),
        (SimpleImputer(strategy="median"), "descriptors"),
    )

Because it is an ordinary estimator, the kernel's own parameters are nested
parameters of the pipeline, and tune like any other:

.. code-block:: python

    GridSearchCV(pipeline, {"makernel__kernel__r": [6, 10, 14]}, cv=...)

Whether that is worth searching is a separate question. On the bundled
n-back/flanker studyset, MKDA radii from 6 mm to 15 mm score between 0.587 and
0.597 against a fold standard deviation of 0.03, while the widest kernel costs
eleven times the non-zeros and eight times the fitting time of the narrowest.
Bandwidth is cheap to search and, on these data, has nothing to find.

Skipping the kernel entirely is also a feature set: an atlas applied straight
to the peak columns gives the number of reported coordinates per region, which
:class:`~nimare.ml.MaskerTransformer` will do because it reads either column
space.

One kernel cannot move into the pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An :class:`~nimare.meta.kernel.ALEKernel` given neither ``fwhm`` nor
``sample_size`` derives a separate width for every analysis from that
analysis's sample size. A transformer is handed a slice of rows and is not told
which rows they are, so there is no way to line a per-analysis sample size up
with the row it belongs to, and ``MAKernel`` refuses it rather than guessing.

scikit-learn's metadata routing exists for exactly this, but it is not
dependable here: routing into a :class:`~sklearn.compose.ColumnTransformer`
raises on scikit-learn 1.4.0, works on 1.4.2, and between 1.5 and 1.7 delivers
the metadata to the training fold and silently drops it on the held-out one --
a model scored against maps built with a different kernel width than it was
fitted with. It works again from 1.8, which requires Python 3.11.

Pass a width that holds across analyses instead -- ``ALEKernel(fwhm=10)`` or
``ALEKernel(sample_size=20)``.

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

Categorical fields
~~~~~~~~~~~~~~~~~~

A feature matrix is numeric, and it is sparse and 900,000 columns wide, so a
string cannot be a column of it: a pandas frame would hold one, but each of its
columns is a separate array, and a single row-slice of one that wide costs
about 34 seconds against 0.4 milliseconds for the matrix. Cross-validation
slices rows constantly, so that is not a trade worth making.

What enters the matrix is therefore the *position* of a category, and the
bundle carries what the positions mean::

    bunch.descriptor_categories
    # {'group_name': ['healthy', 'patients']}

That is a representation rather than an encoding, so the choice of encoding
stays where every other transformation now is -- in the pipeline:

.. code-block:: python

    make_nimare_column_transformer(
        bunch,
        (MAKernel(MKDAKernel(r=10), source_masker=bunch.masker), "voxels"),
        (OneHotEncoder(handle_unknown="ignore"), "group_name"),
        (SimpleImputer(strategy="median"), "count"),
    )

A code plus its category list is bit-identical to one-hot encoding the strings
themselves, and ``OneHotEncoder``, ``OrdinalEncoder`` and ``TargetEncoder`` are
all equally available. Two things the bundle fills in, because scikit-learn
cannot work them out from an array of numbers: ``categories=``, so that a fold
whose training split happens to miss a category still yields the same number of
columns, and the real labels in ``get_feature_names_out``, so a coefficient
reads back as ``group_name_healthy`` rather than ``group_name_0.0``.

None of scikit-learn's encoders accept sparse input --
``OneHotEncoder``, ``OrdinalEncoder`` and ``TargetEncoder`` all raise ``Sparse
data was passed, but dense data is required`` -- which the per-block
densification already handles.

**A code cannot be passed through.** It stands for a label, not a quantity, so
handing it to a model as it stands says that the third category is three times
the first. A spec covering a coded column must cover only coded columns, and
must not be ``"passthrough"``; ``("drop", "group_name")`` is still the way to
say it is not wanted. Free text stays refused outright, since it has no reading
as a column at all.

Annotation labels
~~~~~~~~~~~~~~~~~

An annotation is usually thousands of mostly-empty columns: the bundled
Neurosynth studyset carries 3,228 labels, and the 2026-09 NeuroStore release
annotates 115,748 analyses with 924. Naming them one at a time is not a
workflow, so a field that reads as a glob pattern selects every label matching
it:

.. code-block:: python

    bunch = studyset.to_bunch(
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
  records their ids in provenance, ``"include"`` keeps them as all-zero peak
  rows.
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

    bunch = studyset.to_bunch(test_size=0.25, random_state=13)

    X_train, y_train = bunch.data[bunch.train], bunch.target[bunch.train]

It is a :class:`~sklearn.model_selection.GroupShuffleSplit` over ``groups``, so
a study belongs to exactly one partition. That grouping is the point: on the
bundled Studyset a plain :func:`~sklearn.model_selection.train_test_split`
puts 112 of its 320 studies on both sides at once, and nothing about the
resulting score says so. ``test_size`` counts *studies*, so analysis counts
only approximate a fraction.

Without ``test_size`` the two keys are absent and the bundle is exactly as it
was. For several splits of one bundle -- or for cross-validation, which needs
no split at all -- hand ``groups`` to a group splitter rather than converting
again:

.. code-block:: python

    cross_val_score(pipeline, bunch.data, bunch.target,
                    cv=GroupKFold(5), groups=bunch.groups)

Reducing the voxel features
---------------------------

Once :class:`~nimare.ml.MAKernel` has made the MA maps, they are an ordinary
sparse matrix, so ordinary scikit-learn transformers reduce them, imported from
scikit-learn and used as scikit-learn documents them:

.. code-block:: python

    from sklearn.decomposition import TruncatedSVD

    pipeline = make_pipeline(
        MAKernel(MKDAKernel(r=10), source_masker=bunch.masker),
        TruncatedSVD(n_components=50),
        LogisticRegression(),
    )

Anything that reads sparse input works: truncated SVD, sparse random
projection, variance thresholding. :class:`~sklearn.decomposition.PCA` also
accepts sparse input, but only with its ``arpack`` or ``covariance_eigh``
solvers -- the default ``auto`` picks a dense one and refuses -- and it centres
the data, so :class:`~sklearn.decomposition.TruncatedSVD` remains the better
fit for a matrix this wide.

Keeping a reducer off the descriptor columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A transformer placed directly in a pipeline sees every column it is given. With
voxel features alone that is the whole matrix and nothing else is needed. Once
there are descriptor columns, a bare reducer would decompose them along with
the voxels, which is what :class:`~sklearn.compose.ColumnTransformer` exists to
prevent.

:func:`~nimare.ml.make_nimare_column_transformer` is
:func:`~sklearn.compose.make_column_transformer` with the bundle filled in. It
takes the same ``(transformer, columns)`` pairs, names the steps the same way,
and passes ``remainder``, ``sparse_threshold``, ``n_jobs``, ``verbose`` and
``verbose_feature_names_out`` straight through:

.. code-block:: python

    preprocessor = make_nimare_column_transformer(
        bunch,
        (TruncatedSVD(n_components=50), "voxels"),
        (SimpleImputer(strategy="median"), "sample_sizes"),
        (StandardScaler(), "year"),
    )

``columns`` may be a name, as it may be for a ColumnTransformer reading a
frame: ``"voxels"`` and ``"descriptors"`` name the two blocks, and a descriptor
may be named by its own field name. It may equally be a slice, indices, a mask
or a callable -- whatever scikit-learn accepts.

What the bundle adds is the four things scikit-learn cannot work out from an
array of numbers:

* the column spans of the two blocks;
* the masker, bound into an atlas reducer, so ``(difumo, "voxels")`` works;
* the column names, so a fitted coefficient can be read back to its field;
* ``sparse_threshold=1.0``, because scikit-learn's ``0.3`` would densify an
  unreduced voxel block -- about 6.5 GB at 902,629 grid columns.

**It is not needed until there are descriptors.** ``descriptor_fields`` is
None by default, and then ``descriptor_columns`` is empty and the whole matrix
is voxels, so a plain ``make_pipeline(TruncatedSVD(50), LogisticRegression())``
is already correct.

There is no way to mark a column so that a transformer skips it: scikit-learn
hands every transformer the whole of ``X``, and a ColumnTransformer names what
each group gets -- indeed its default ``remainder="drop"`` *discards* the
columns nobody claimed. What stands in for marking is the column spec, which is
what :func:`~sklearn.compose.make_column_selector` builds for a frame and what
``bunch.voxel_columns`` already is for this array: an ordinary :class:`slice`.
``remainder="drop"`` is right for a frame of many columns and wrong here,
where the two blocks are the whole of the matrix: naming only the descriptors
would discard every voxel and leave a model fitted on one column of sample
sizes, without a word. So leaving *either* block unclaimed raises. Say
``("drop", "voxels")`` or ``remainder="passthrough"`` when that is what is
meant -- the refusal is about silence, not about the outcome.

So the same thing can be written out by hand, and should be for anything this
function does not cover:

.. code-block:: python

    ColumnTransformer(
        [
            ("voxels", TruncatedSVD(n_components=50), bunch.voxel_columns),
            ("descriptors", SimpleImputer(), bunch.descriptor_columns),
        ],
        sparse_threshold=1.0,
    )

Sparse descriptors and scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Descriptor columns sit inside the same matrix as the voxels, so they arrive
sparse. scikit-learn handles sparse input widely; what it refuses is
*centring* it, because subtracting a mean from every entry fills in the
implicit zeros. :class:`~sklearn.preprocessing.StandardScaler` says so and
names the fix, ``with_mean=False``, and
:class:`~sklearn.preprocessing.MaxAbsScaler` is the sparse-safe scaler.

:func:`~nimare.ml.make_nimare_column_transformer` decides per transformer, by
fitting a clone on a sparse probe: the answer is a property of the arguments
rather than the class, since ``StandardScaler()`` refuses sparse input and
``StandardScaler(with_mean=False)`` does not, and no estimator tag separates
them. A transformer that passes keeps the block sparse; one that fails is
handed dense columns, which always works.

What the probe asks is whether sparsity is what the failure is *about*, not
whether the probe failed. A probe is small, so plenty of transformers refuse it
for their own reasons: ``TruncatedSVD(n_components=50)`` cannot fit it at any
sparsity, and :class:`~nimare.ml.MAKernel` reads the width to know which space
its columns are in. Counting those as needing dense input densified the voxel
block before the canonical sparse reducer, which at 902,629 columns is 16.9 GB.
So a transformer that fails the sparse probe is tried again
on the same probe made dense, and only a failure that densifying *fixes* is
about sparsity.

A pipeline is probed whole rather than by its first step, because a step that
takes sparse input may also pass it on: ``SimpleImputer`` hands sparse columns
to whatever follows, so ``make_pipeline(SimpleImputer(), StandardScaler())``
needs dense input even though its first step does not.

An encoder is probed as you wrote it, before the bundle fills in
``categories=``: an encoder told which categories to expect refuses the probe's
own values, and that is not a statement about sparsity.

That matters most for a pattern selection. Scaling the Neurosynth release's
3,228 labels over 115,748 analyses would be 2.8 GB dense and is 2% filled, so
``MaxAbsScaler`` stays sparse while ``StandardScaler()`` -- which was asked to
centre -- does not.

Reading a model back
~~~~~~~~~~~~~~~~~~~~

``get_feature_names_out`` works through the preprocessor, so a fitted
coefficient can be read back to the thing it weighs:

.. code-block:: python

    pipeline.fit(bunch.data, bunch.target)
    names = pipeline[:-1].get_feature_names_out()
    dict(zip(names, pipeline[-1].coef_[0]))
    # {'pipeline__truncatedsvd0': 0.032, ..., 'simpleimputer__sample_sizes': -0.009}

A :class:`~sklearn.compose.ColumnTransformer` selects these columns by
position, because the feature matrix is an array rather than a frame, so a
descriptor would otherwise come out as ``x902629``.
:func:`~nimare.ml.make_nimare_column_transformer` restores the real names, including for
descriptors left at ``"passthrough"``.

Letting nilearn transform the voxels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A nilearn masker is already a scikit-learn transformer -- it subclasses
:class:`~sklearn.base.BaseEstimator` and
:class:`~sklearn.base.TransformerMixin`, it clones, it carries ``get_params``.
The one thing that keeps it out of a pipeline over these features is that it
takes *images*, where a :class:`~sklearn.compose.ColumnTransformer` hands out
columns of an array.

:class:`~nimare.ml.MaskerTransformer` is that bridge, one of the two
transformers NiMARE adds, because it has to know which voxel each
column is. Rows are unmasked back into images in the source mask's space,
handed to the nilearn masker, and returned as an array. The bundle's
``masker`` is what says where the columns came from, which is why it travels
with the data:

.. code-block:: python

    MaskerTransformer(fetch_atlas_difumo(dimension=64), source_masker=bunch.masker)
    MaskerTransformer(NiftiMasker(smoothing_fwhm=6), source_masker=bunch.masker)

What it applies is any nilearn masker, or anything nilearn loads as an atlas:
what a ``fetch_atlas_*`` function returns, an atlas image or file, or the name
of a fetcher with its arguments. A 4D atlas is summarised with a
:class:`~nilearn.maskers.NiftiMapsMasker` and a 3D one with a
:class:`~nilearn.maskers.NiftiLabelsMasker`, and the atlas's own region names
become the feature names. A :class:`~nilearn.maskers.NiftiMasker` returns
voxels rather than regions, which is how nilearn's smoothing, standardizing
and detrending reach these features.

How many regions an atlas yields is nilearn's answer, not NiMARE's, and it
differs by version: a region falling outside the mask is kept by nilearn 0.12
and dropped by 0.13. A feature matrix built this way is comparable across
environments only when the nilearn version is.

Outside a pipeline, fit it on the training rows and apply the same fitted one
to the held-out rows:

.. code-block:: python

    voxels = bunch.data[:, bunch.voxel_columns]
    reducer = MaskerTransformer(atlas, source_masker=bunch.masker)

    train_reduced = reducer.fit_transform(voxels[train])
    test_reduced = reducer.transform(voxels[test])

Calling ``fit_transform`` on the held-out rows would fit the reduction on the
analyses being held out.

Rows are converted back into images in batches, because nilearn's maskers
aggregate an image rather than a row. Most of the cost is per call rather than
per row, and ``batch_size`` has an optimum rather than a trade: against a 2 mm
whole-brain mask, a batch of 8 costs 70.6 ms per row, 32 costs 32.7 ms, and 128
costs 50.8 ms for four times the dense working set. The default of 32 is that
optimum, at roughly 60 MB.

Peak columns cost about the same as MA columns here despite spanning four times
the grid -- 35.5 ms per row against 32.7 -- because what dominates is nilearn's
per-call setup rather than the width of the array.

Scale
-----

Conversion no longer generates maps, so it is a read rather than a
computation. Against the 2026-09 NeuroStore release on a 2 mm whole-brain mask,
whose grid is 902,629 columns:

=================  ========  ===========
Analyses           Time      Non-zeros
=================  ========  ===========
500                  0.8 s         3,008
2,000                0.2 s        13,825
8,000                0.2 s        79,344
20,000               0.3 s       218,485
115,748              1.5 s       852,973
=================  ========  ===========

The whole release converts in under two seconds, and peak memory did not rise
measurably above the cost of holding the Studyset itself. Under the previous
design, which ran a kernel over every analysis at conversion, 20,000 analyses
took 27.7 s and 91 million non-zeros, and the full release was estimated at
roughly 6 GB of sparse data -- too much for a 16 GB machine, so it had to be
sliced first.

What that moved rather than removed is the expansion: a 10 mm MKDA kernel still
produces about 4,700 non-zeros per row. It now happens inside the pipeline,
per fold, over a training subset, and a :class:`~sklearn.pipeline.Pipeline`
built with ``memory=`` caches it across folds and across candidates in a
search.

Unreduced voxel features stay sparse everywhere -- in the bundle, through
:func:`~nimare.ml.make_nimare_column_transformer`, and through
:class:`~nimare.ml.MAKernel` -- and only an explicit reducer produces a dense
representation.

.. seealso::

    :ref:`machine_learning_in_nimare` walks through the whole workflow.
