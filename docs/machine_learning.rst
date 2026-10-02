.. include:: links.rst

Machine learning with Studysets
===============================

:meth:`~nimare.studyset.Studyset.to_bunch` transforms a
:class:`~nimare.nimads.Studyset` into scikit-learn compatible arrays: an ``X``
feature matrix containing the voxel data and any additional information about
the analysis, and the ``y`` target.

.. code-block:: python

    bunch = studyset.to_bunch(target_field=("metadata", "comparison_task"))

:mod:`nimare.ml` is the convenience wrapper to help do scikit-learn style
analyses with meta-analytic data. :func:`~nimare.ml.describe_fields` reports
which of a Studyset's fields are worth modelling, :class:`~nimare.ml.MAKernel`
applies a kernel to reported peaks creating modeled activation (MA) maps,
:func:`~nimare.ml.make_nimare_column_transformer` routes each block to its own
transformer, :class:`~nimare.ml.MaskerTransformer` lets a nilearn masker act on
the voxel columns, and :func:`~nimare.ml.coefficient_image` puts a fitted
model's weights back on the brain.

Questions you can ask with it:

1. Does the pattern of reported coordinates predict something about a study,
   such as the task it used or the population it recruited?
2. Which regions carry that prediction?
3. Does study information, such as sample size or an annotation label, add to
   what the coordinates already say?

What is a bunch?
-----------------

A :class:`~sklearn.utils.Bunch` holds ``data`` (sparse), ``target``, ``groups``
(the study each analysis came from), ``ids``, ``feature_names``,
``voxel_columns``, ``descriptor_columns``, ``descriptor_names``, the ``masker``
whose grid the voxels span, and ``provenance``.

There are some niceties and conveniences for using the bunch with NiMARE
architecture, and if you're just starting out we recommend you do. If you're a
seasoned scikit-learn practitioner and have enough context to understand the
meta-analytic data, you can take the bunch and use it as you would any other
scikit-learn dataset.

The voxel columns span the whole image grid of the bunch's masker. A
coordinate just outside the mask still spreads into it once a kernel is
applied, so keeping the full grid keeps that contribution. Peaks are sparse, so
this stays cheap.

Turning peaks into Modeled Activation maps
------------------------------------------

``to_bunch`` gives you the peaks each analysis reported.
:class:`~nimare.ml.MAKernel` turns those peaks into MA maps, and it is a
transformer, so it goes in your pipeline alongside everything else:

.. code-block:: python

    from nimare.meta.kernel import MKDAKernel
    from nimare.ml import MAKernel

    pipeline = make_pipeline(
        MAKernel(MKDAKernel(r=10), source_masker=bunch.masker),
        TruncatedSVD(n_components=50),
        LogisticRegression(),
    )

Keeping the kernel in the pipeline means it is fitted on training rows only,
and that its radius can be tuned like any other hyperparameter. Any NiMARE
kernel works: :class:`~nimare.meta.kernel.MKDAKernel`,
:class:`~nimare.meta.kernel.KDAKernel`,
:class:`~nimare.meta.kernel.ALEKernel`.

You can also convolve once, outside the pipeline, and cross-validate the maps
that come out:

.. code-block:: python

    maps = MAKernel(MKDAKernel(r=10), source_masker=bunch.masker).fit_transform(
        bunch.data[:, bunch.voxel_columns]
    )
    cross_val_score(rest_of_pipeline, maps, bunch.target, groups=bunch.groups, cv=cv)

.. note::

    Give :class:`~nimare.meta.kernel.ALEKernel` a width that holds across
    analyses, such as ``ALEKernel(fwhm=10)`` or ``ALEKernel(sample_size=20)``.
    A transformer receives a slice of rows without being told which rows they
    are, so a per-analysis sample size has nothing to line up with.

Choosing what to model
----------------------

Name a target field to predict, and descriptor fields to predict it from
alongside the voxels. Both take a bare field name, or a ``(source, field)``
tuple when the same name appears in more than one place. The sources are
``"metadata"``, ``"annotations"`` and ``"texts"``.

.. code-block:: python

    bunch = studyset.to_bunch(
        target_field=("metadata", "comparison_task"),
        descriptor_fields=["sample_sizes"],
    )

Study-level metadata is inherited by that study's analyses, and list-valued
fields such as ``sample_sizes`` are reduced the way the rest of NiMARE reduces
them.

Categorical fields
~~~~~~~~~~~~~~~~~~

A feature matrix holds numbers, so a categorical descriptor enters it as a
category code. The bunch tells you what the codes mean::

    bunch.descriptor_categories
    # {'group_name': ['healthy', 'patients']}

Pick your encoder in the pipeline, as you would for any other dataset:

.. code-block:: python

    make_nimare_column_transformer(
        bunch,
        (MAKernel(MKDAKernel(r=10), source_masker=bunch.masker), "voxels"),
        (OneHotEncoder(handle_unknown="ignore"), "group_name"),
        (SimpleImputer(strategy="median"), "sample_sizes"),
    )

:func:`~nimare.ml.make_nimare_column_transformer` hands the encoder the
category list, so a training fold that happens to miss a category still
produces the same columns, and ``get_feature_names_out`` gives you
``group_name_healthy`` instead of ``group_name_0.0``. A code stands for a
label, so give it an encoder; use ``("drop", "group_name")`` to leave it out.

Annotation labels
~~~~~~~~~~~~~~~~~

Annotations usually run to thousands of labels, so a field name that reads as a
glob pattern selects every label matching it:

.. code-block:: python

    bunch = studyset.to_bunch(
        descriptor_fields=[("annotations", "Neurosynth_TFIDF__*")],
    )

Pattern selections stay sparse, and labels keep their own names. An exact name
always wins over pattern matching, so a label whose name happens to contain
``*`` or ``[`` is still selectable by name.

The two forms say different things about absence. A pattern gives you a label
matrix, where a label no analysis carries is a zero. An exact name gives you a
value, where an absent value is missing and ``missing_values`` applies to it.

Handling missing data
---------------------

You have explicit control on how to handle missing data.

``missing_coordinates`` says what to do with analyses that report no
coordinates. ``"drop"`` (the default) removes them and records their ids in
provenance; ``"include"`` keeps them as all-zero peak rows.

``missing_values`` says what to do when an analysis does not report a selected
descriptor or target. ``"raise"`` (the default) names the fields and analyses
so you can decide; ``"drop"`` removes those analyses; ``"keep"`` leaves the
gaps for an imputer in your pipeline.

Targets and descriptors differ in what you can do about a gap, so you can set
the policy per role:

.. code-block:: python

    missing_values={"target": "drop", "descriptors": "keep"}

An imputer in your pipeline can fill a missing descriptor. A missing target is
what you are trying to predict, so those analyses are usually better dropped.
This pairing is the common one on a release, where the field you want to
predict and the fields describing each analysis are reported by different
subsets of it.

Two other checks run at conversion. A target needs more than one value among
the analyses that were kept, and descriptors need to be numeric or categorical.
For free text, read it from the Studyset's own tables, turn it into numbers,
and select the numeric result.

Finding the fields worth using
------------------------------

A release-scale Studyset offers more fields than anyone can read, and most of
them are reported by very few analyses. :func:`~nimare.ml.describe_fields`
tells you what each one holds, using the same reader ``to_bunch`` uses, so a
field it calls numeric is numeric there:

.. code-block:: python

    from nimare.extract import fetch_neurostore
    from nimare.ml import describe_fields

    studyset = fetch_neurostore()
    fields = describe_fields(studyset, min_coverage=0.5)

    fields[fields.n_unique.between(2, 12)]      # classification targets
    fields[fields.kind == "numeric"]            # descriptors and regression targets

It returns a :class:`~pandas.DataFrame` of ``source``, ``field``, ``kind``,
``coverage``, ``n_unique`` and ``example``, ordered by coverage, so picking a
field is a query. The ``field`` column, filtered how you like, is the
``descriptor_fields`` list to pass back to ``to_bunch``.

Splitting without leaking a study
---------------------------------

Analyses from one study are related, so keep each study on one side of a split.
Pass ``test_size`` to get ``train`` and ``test`` row positions on the bunch:

.. code-block:: python

    bunch = studyset.to_bunch(test_size=0.25, random_state=13)

    X_train, y_train = bunch.data[bunch.train], bunch.target[bunch.train]

It is a :class:`~sklearn.model_selection.GroupShuffleSplit` over ``groups``, so
each study lands in exactly one partition. ``test_size`` counts *studies*, so
analysis counts approximate the fraction you asked for.

For cross-validation, hand ``groups`` to a group splitter and convert once:

.. code-block:: python

    cross_val_score(pipeline, bunch.data, bunch.target,
                    cv=GroupKFold(5), groups=bunch.groups)

Select rows on the Studyset where you can.
:meth:`~nimare.studyset.Studyset.slice` takes analysis ids and
:meth:`~nimare.studyset.Studyset.select_analyses` takes a mask or positions.
Conversion is the expensive step, so run it on the rows you are modelling.

Reducing the voxel features
---------------------------

MA maps are an ordinary sparse matrix, so ordinary scikit-learn transformers
reduce them. Truncated SVD, sparse random projection and variance thresholding
all read sparse input:

.. code-block:: python

    pipeline = make_pipeline(
        MAKernel(MKDAKernel(r=10), source_masker=bunch.masker),
        TruncatedSVD(n_components=50),
        LogisticRegression(),
    )

.. note::

    :class:`~sklearn.decomposition.PCA` reads sparse input through its
    ``arpack`` or ``covariance_eigh`` solvers, and it centres the data.
    :class:`~sklearn.decomposition.TruncatedSVD` is the better fit for a matrix
    this wide.

Keeping a transformer off the descriptor columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With voxel features alone, a plain pipeline is all you need. Once you add
descriptor columns, give each block its own transformer so it works on the
voxels only. :func:`~nimare.ml.make_nimare_column_transformer` is
:func:`~sklearn.compose.make_column_transformer` with the bunch filled in: the
same ``(transformer, columns)`` pairs, the same step names, and ``remainder``,
``sparse_threshold``, ``n_jobs``, ``verbose`` and ``verbose_feature_names_out``
passed straight through.

.. code-block:: python

    preprocessor = make_nimare_column_transformer(
        bunch,
        (TruncatedSVD(n_components=50), "voxels"),
        (SimpleImputer(strategy="median"), "sample_sizes"),
        (StandardScaler(), "year"),
    )

``columns`` may be ``"voxels"``, ``"descriptors"``, a descriptor's own field
name, or anything scikit-learn accepts: a slice, indices, a mask, a callable.
What the bunch fills in is the span of each block, the masker an atlas needs,
the column names so a coefficient reads back to its field, and a
``sparse_threshold`` that keeps a wide voxel block sparse.

Claim both blocks. Naming only the descriptors would quietly leave you with a
model fitted on one column of sample sizes, so say what you mean with
``("drop", "voxels")`` or ``remainder="passthrough"``.

Descriptor columns arrive sparse, alongside the voxels. Sparse-safe scalers
such as :class:`~sklearn.preprocessing.MaxAbsScaler`, or
``StandardScaler(with_mean=False)``, keep them that way;
``make_nimare_column_transformer`` hands dense columns to a transformer that
wants them.

Letting nilearn transform the voxels
------------------------------------

:class:`~nimare.ml.MaskerTransformer` lets any nilearn masker act on the voxel
columns. Give it an atlas and you get one feature per region, with the atlas's
own names:

.. code-block:: python

    from nilearn.datasets import fetch_atlas_difumo
    from nimare.ml import MaskerTransformer

    atlas = MaskerTransformer(fetch_atlas_difumo(dimension=64), source_masker=bunch.masker)

It takes what nilearn loads as an atlas: a ``fetch_atlas_*`` result, an image
or file, or a fetcher name such as ``"harvard_oxford"``. A 4D atlas is
summarised with a :class:`~nilearn.maskers.NiftiMapsMasker` and a 3D one with a
:class:`~nilearn.maskers.NiftiLabelsMasker`. A
:class:`~nilearn.maskers.NiftiMasker` gives you voxels back, which is how
nilearn's smoothing and standardizing reach these features::

    (NiftiMasker(smoothing_fwhm=6), "voxels")

It reads either column space and works out which from the width, so an atlas
can summarise the raw peaks — coordinates reported per region — or the MA maps
a kernel made.

Reading a model back to the brain
---------------------------------

:func:`~nimare.ml.coefficient_image` walks a fitted pipeline backwards, undoing
each reduction until the weights are one per voxel, and unmasks them:

.. code-block:: python

    image = coefficient_image(pipeline, bunch)
    plotting.plot_stat_map(image)

A weight is not a point in feature space, so a step is not undone with its
``inverse_transform``: a model scoring ``w @ (A x + c)`` scores
``(A.T @ w) @ x`` plus a constant, and the weight moves back by the transpose of
the step's linear part, its offset going to the intercept. Through a
:class:`~sklearn.preprocessing.StandardScaler` that is ``w / scale``, where the
inverse would give ``w * scale + mean``. The steps read back are the ones whose
linear part is known — the scikit-learn scalers, PCA, truncated SVD and feature
selectors such as variance thresholding — and an atlas reduction is read back
through the atlas; any other step is refused rather than guessed past. With a
:class:`~sklearn.compose.ColumnTransformer` it follows the branch covering the
voxels, so descriptor weights stay where they belong. The walk stops at
:class:`~nimare.ml.MAKernel`, whose input is peaks.

Through an atlas, each voxel gets its own share of a region's weight — the
region's weight divided by its size for an averaging labels atlas, corrected for
the overlap of the maps for a probabilistic one — so that the image still scores
a map as the model does. ``atlas="region"`` paints each region's weight over its
voxels instead, which shows the regions but is not a per-voxel weight:

.. code-block:: python

    per_voxel = coefficient_image(pipeline, bunch)
    painted = coefficient_image(pipeline, bunch, atlas="region")

Pass ``coef`` to project weights the model does not carry itself, such as a
permutation importance.

Weights say what a model uses, which includes voxels that only cancel noise
elsewhere, so a weight map is not an activation map. ``kind="pattern"`` returns
the activation pattern of :footcite:t:`haufe2014interpretation` instead: how
each voxel covaries with the model's scores. It is computed from the voxel maps
and the scores rather than by undoing each step, so it works through any
pipeline whose final model is linear in its features, including steps whose
weights cannot be read back:

.. code-block:: python

    pattern = coefficient_image(pipeline, bunch, kind="pattern")

Pass ``X`` when the pipeline was fitted on something other than ``bunch.data``,
such as the voxel block alone.

Working at release scale
------------------------

:func:`~nimare.extract.fetch_neurostore` downloads a published NeuroStore
release. Conversion reads peaks, so a whole release converts in a couple of
seconds and stays sparse from the bunch through the column transformer.

Applying a kernel is the expensive step, and it happens in your pipeline, per
fold, on a training subset. Slice the Studyset to the analyses you are
modelling first, then convert:

.. code-block:: python

    studyset = fetch_neurostore(version="2026-09")
    subset = studyset.slice(analyses=analysis_ids_you_want)
    bunch = subset.to_bunch(target_field=("annotations", "TaskExtractor.fMRITasks[0].RestingState"))

If you are searching over the steps that follow the kernel, ``cache=True`` on
:class:`~nimare.ml.MAKernel` reuses maps it has already made, and
:func:`~nimare.ml.clear_map_cache` releases them.

.. seealso::

    :ref:`machine_learning_in_nimare` walks through the whole workflow.

References
----------
.. footbibliography::
