# Public API Contract: `nimare.ml`

This contract defines the additive public surface for converting NiMARE
Studysets into scikit-learn-compatible feature datasets.

**Revised 2026-09-25** against `interface-design.md`, which records the options
weighed for each decision below. Terminology: each dataset row is an analysis
row.

## Module

`nimare.ml`

The module is exported from `nimare/__init__.py` and documented in
`docs/api.rst`. Its public names are `FeatureSet`, which builds itself from a
Studyset, and `AtlasAggregator`. Every other reduction is an ordinary
scikit-learn transformer: the module must not re-export scikit-learn under
NiMARE names.

## Division of Responsibility

NiMARE owns extraction: reading the Studyset, generating modeled activation
(MA) maps, aligning every row to its analysis, and reporting what is missing.
scikit-learn owns evaluation: splitting, fitting, reduction and scoring.

Kernel transformation is row-independent, so building the whole matrix before
splitting leaks nothing. Everything that learns *across* rows must be fit on
training rows only, which is what a `Pipeline` is for. The module must not
reimplement that machinery.

## Utility Preference Order

For every task, implementation must look for reusable functionality in this
order before adding local helpers:

1. Existing NiMARE utilities, classes, fixtures, and documentation conventions.
2. nilearn utilities for mask/image-aware operations.
3. scikit-learn utilities for dataset containers, splitters, preprocessing,
   decomposition, and pipelines.
4. New local helpers only when none of the above provides the needed behavior.

## `FeatureSet.from_studyset`

`FeatureSet.from_studyset(studyset, kernel_transformer, **options)` is the
public entry point. It converts one Studyset and returns one `FeatureSet`.
Conversion is a single call, not a configure-then-call pair: the settings are
arguments, and what comes back is the thing the researcher works with.

It is a named constructor rather than `__init__` because the container is also
built from blocks that already exist -- by `split`, `select_analyses`, `copy`
and the map-reduction methods -- and those must not go through a kernel. A
constructor that ran a kernel would push every internal path onto a back door
and put the container's validation there with it.

The conversion logic lives in an internal `_FeatureExtractor` class, so that the
stages -- field selection, target handling, row retention, map generation,
provenance -- stay separate methods over shared configuration. That class is not
exported, not documented, and not part of the public surface. No public object
in this module takes a Studyset and exposes `fit` or `fit_transform`.

### Signature

Required, positionally:

- `studyset`: the Studyset to convert. One analysis becomes one row.
- `kernel_transformer`: existing NiMARE kernel transformer instance or class.
  No implicit scientific default is selected; public examples must pass an
  explicit kernel transformer.

Keyword-only:

- `descriptor_fields`: field selectors appended to the feature matrix as extra
  numeric columns.
- `target_field`: field selector for `y`.
- `target_transformer`: callable or stateless transformer applied to the raw
  target values, for free-text and multi-label fields that need a label
  extractor.
- `missing_coordinates`: `"drop"` (default) or `"include"`.
- `missing_values`: `"raise"` (default), `"drop"` or `"keep"`, for missing
  descriptor and target values.
- `memory`, `memory_level`: joblib cache location for MA map generation,
  applied to a copy of the kernel transformer when it does not define its own,
  following the NiMARE `CacheMixin` convention. Kernel transformers cache their
  maps at memory level 2, so `memory_level` defaults to 2 here; a lower level
  is a request not to cache. Repeated conversions of the same Studyset then
  reuse the maps, across processes as well as within one.

The option vocabulary is validated before any work is done.

### Required behavior

- Use Studyset-native access for IDs, coordinates, masker, metadata,
  annotations, and texts.
- Generate MA features through `KernelTransformer.transform(..., return_type="sparse")`.
- **Align map rows to analyses by id.** Kernel transformers return maps ordered
  by analysis id and discard the ids that name them, so positional pairing is
  only correct while the Studyset happens to be in sorted order. A row count
  that cannot be reconciled must raise.
- Reject duplicate analysis identifiers, which would otherwise collapse into
  one map row unnoticed.
- Determine study groups from the Studyset's per-analysis study ids.
- When `missing_coordinates="include"`, analyses with no coordinates are
  all-zero sparse map rows. When `"drop"`, they are removed before row
  construction and recorded in provenance.
- Append numeric descriptor fields directly. Reject categorical and text
  descriptor fields, naming the field, its kind, and the two supported routes:
  encode it and select the numeric result, its raw values being in the Studyset
  tables the message names.
- Export scalar numeric and scalar categorical targets as a one-dimensional
  `y`. Reject raw free-text and multi-label targets unless `target_transformer`
  is supplied, and reject a target that has one value for every analysis.
- Report missing descriptor and target values under `missing_values="raise"`,
  naming the fields and the affected analysis ids; record them in provenance
  under `"drop"` and `"keep"`. Only analyses that survive `missing_coordinates`
  can be missing anything: the rest are not in the output to be missing from.
- Refuse a target that is constant over the analyses that were **kept**, since
  a minority class can leave with the analyses that had no coordinates.
- Inherit study-level metadata even when a sibling analysis declares the same
  field.
- Never mutate the caller's kernel transformer.

## `FeatureSet`

The aligned container, and the module's one class. Row `i` is analysis `ids[i]` from study `study_ids[i]`,
and that order is preserved by every method.

### Required attributes

- `ids`: full Studyset analysis identifiers, `<study_id>-<analysis_id>`.
- `study_ids`: one study-group label per row.
- `map_features`: the analysis-by-voxel block. Sparse while unreduced.
- `descriptor_features`: the numeric descriptor block, sparse when it holds
  annotation labels, or `None`.
- `descriptor_names`: the descriptor columns, in order, under their real names.
- `features`: map features and descriptor features side by side, derived from
  the two blocks on first access so they cannot disagree.
- `feature_names`: names for `features` in column order, built on first access
  because naming every voxel eagerly costs more memory than the matrix.
- `target`: optional row-aligned prediction target.
- `masker`: the masker defining voxel order for unreduced map features.
- `provenance`: conversion settings and source Studyset details, including
  `missing_coordinates`, `dropped_ids`, `missing_value_ids`, the kernel
  transformer and its parameters, and any map reductions applied.
- `map_columns`, `descriptor_columns`: the column slices of `features`.

### Required methods

- `to_sklearn(return_X_y=False)`: return a `sklearn.utils.Bunch` with `data`,
  `target`, `groups`, `feature_names`, `ids`, `provenance`, `map_columns` and
  `descriptor_columns`; or `(data, target)`, following the `sklearn.datasets`
  convention.
- Every derivation -- `split`, `select_analyses`, `copy`, the map-reduction
  methods -- MUST return the caller's own type and give its result an
  independent `provenance` whose `n_rows` describes that result.
- `split(test_size=0.25, random_state=None)`: grouped holdout by study through
  `GroupShuffleSplit`, returning `(train, test)`. `test_size` is a fraction of
  *studies*. Validate the study count first and raise before returning anything
  partial.
- `make_preprocessor(map_reducer="truncated_svd", descriptor_transformer="passthrough", **reducer_params)`:
  return an unfitted `ColumnTransformer` that reduces the map columns and
  handles the descriptor columns separately, with `sparse_threshold=1.0` so
  unreduced voxelwise features are never densified on the way through.
  `map_reducer` may be a workflow name, an already-built transformer, or
  `None`/`"passthrough"`.
- `fit_transform_maps(reducer)`: fit the reducer on this dataset's map features
  and return a reduced dataset.
- `transform_maps(reducer)`: apply an already fitted reducer, raising
  `NotFittedError` otherwise, because fitting it there would use held-out data.
- `select_analyses(rows)`: restrict to a boolean mask or an array of positions.
- `copy()`: return an independent dataset copy.

### Errors

- Raise `ValueError` when feature, target, descriptor, group, or row dimensions
  do not align.
- Raise `ValueError` when a split cannot be created from the available number of
  study groups, or when `test_size` cannot serve one.
- Raise `ValueError` when a reducer returns a row count that does not match the
  input dataset.

## Field Selectors

A selector names one descriptor or target field, or a set of annotation
labels, using the vocabulary NiMARE estimators already use in
`_required_inputs`:

- a bare field name (`"sample_sizes"`), looked up in metadata, annotations and
  texts in turn; an ambiguous name raises and asks for the explicit form;
- a `(source, field)` tuple (`("annotations", "motor_label")`);
- a mapping with `source` and `field`.

A tuple is one selector; a list holds several. Sources are `"metadata"`,
`"annotations"` and `"texts"`, with `"annotations_df"` and `"text"` accepted as
aliases.

### Annotation labels

A field that reads as a glob pattern -- `("annotations", "Neurosynth_TFIDF__*")`
-- selects every annotation label matching it, in the Studyset's order, each
under its own name. An exact name MUST win over pattern matching, so that a
label called `ParticipantDemographicsExtractor.groups[0].BMI` is selectable at
all; 878 of the labels in the bundled NeuroStore studyset are named that way.
A pattern MUST refuse the non-numeric labels it matches, the way an exactly
named field is refused, rather than reading them as zeros. This is how an annotation is used at all: the Neurosynth
release annotates 115,747 analyses with 794 labels, and naming them one at a
time is not a workflow.

Such a selection MUST be read through `label_block_for`, which is also what
decides what the labels are called, so that selection and extraction cannot
disagree about a Studyset carrying several annotations. It MUST stay sparse through extraction, splitting and export, because a dense read of that
annotation is 92 million cells holding 3 million values. Label names MUST
survive whole, double underscores included, since `Neurosynth_TFIDF__pain` is
the name of the thing; where a name has to be softened for a scikit-learn step
name, the softening is internal and `descriptor_names` keeps the real one.

A label no analysis carries is a zero, not a gap, so `missing_values` has
nothing to report about a pattern selection. An exactly named field keeps the
value semantics it has today, where an absent value is missing.

A pattern matching no label MUST raise and show what the Studyset does
annotate with. Patterns are for annotations; a pattern against another source
MUST say so.

Provenance records the selectors as they were given, plus
`n_descriptor_features`, rather than thousands of expanded names.

Numeric metadata is read through `nimare.studyset.requirements.PerAnalysis`, so
study-level fields are inherited by their analyses and list-valued fields such
as `sample_sizes` are reduced the way the rest of NiMARE reduces them.

Default field behavior:

- Descriptor fields must resolve to numeric values; there is no implicit
  encoding.
- Target fields may resolve to scalar numeric or scalar categorical values.
- Raw title, abstract, description, and other free-text fields, and multi-label
  fields, require an explicit `target_transformer` to become a target, and
  cannot become descriptor features.

## Reduction

Map features are an ordinary sparse matrix, so ordinary scikit-learn
transformers reduce them: `TruncatedSVD`, `VarianceThreshold`,
`SparseRandomProjection` and anything else that accepts sparse input, used as
scikit-learn documents them and imported from scikit-learn. A reducer that
cannot read sparse input (`PCA`, for one) fails when it is fitted, with
scikit-learn's own message. There is no NiMARE vocabulary for any of this, and
no factory that re-names it.

`AtlasAggregator` is the one reducer the module adds, because it is the one
that has to know which voxel each column is. It accepts any atlas nilearn can
load:

- a `Bunch` from a `nilearn.datasets.fetch_atlas_*` function, read for its
  `maps` and, when present, its `labels`;
- a 3D (deterministic) or 4D (probabilistic) atlas image, or a path to one;
- the name of a nilearn fetcher, such as `"harvard_oxford"`, with any arguments
  it needs in `atlas_kwargs`;
- a `NiftiLabelsMasker` or `NiftiMapsMasker` the caller configured, which is
  cloned rather than modified.

A 4D atlas is summarised with a `NiftiMapsMasker` and a 3D one with a
`NiftiLabelsMasker`, both with `resampling_target="data"`. A masker that
extracts voxels rather than regions, an image that is neither 3D nor 4D, and a
string that names neither a file nor a fetcher must each raise and say what was
expected. Region definitions, resampling and the aggregation strategy stay
nilearn's business; `get_feature_names_out()` reports region names from the
atlas when it carries them, and from the masker otherwise.

How many regions an atlas yields is nilearn's answer, not NiMARE's, and it
differs by version: a region falling outside the mask is kept by nilearn 0.12
and dropped by 0.13, and for a maps atlas 0.13 drops it from the output without
dropping it from `maps_img_` or `n_elements_`. `get_feature_names_out()` must
therefore report exactly as many names as the masker returns columns, counting
them rather than trusting either attribute, and must fall back to positional
names when the atlas's own labels cannot be matched to the surviving regions.
A feature matrix is comparable across environments only when the nilearn
version is, which the docstring says.

### Applying a reducer to the map columns only

`FeatureSet.make_preprocessor(map_reducer, descriptor_transformer="passthrough",
**reducer_params)` is a `ColumnTransformer` with the column boundary filled in,
the masker bound into an atlas reducer, and `sparse_threshold=1.0` so that a
map block denser than scikit-learn's default threshold is not quietly
densified. It accepts a transformer, a transformer class built from
`**reducer_params`, or an atlas.

`descriptor_transformer` takes one transformer for the whole descriptor block,
or a mapping from descriptor name to transformer when the descriptors need
different treatment. Descriptors the mapping does not name are passed through,
the column order is the one they came in with, and a name that is not a
descriptor raises and lists the ones that are. Descriptor transformers are
handed their columns dense -- the block is stored dense and is a few numeric
columns, and `StandardScaler` will not centre sparse data -- while the map
block stays sparse. Asking for descriptor handling on a feature set that has no
descriptor columns must raise rather than be ignored.

**When there are no descriptor columns it must return the reducer itself.**
There is nothing to keep the reducer away from, and a pipeline step that wraps
one transformer in a `ColumnTransformer` over every column is ceremony. The
documented workflow for map-only feature sets is to put a scikit-learn
transformer straight into the pipeline.

`map_columns`, `descriptor_columns` and `descriptor_names` are public so that
the two-block recipe can be written by hand, and the docstring shows it written
out.

`fit_transform_maps` requires a built transformer: passing an atlas must raise
and name `AtlasAggregator(atlas, masker=features.masker)`, because the fitted
aggregator is what `transform_maps` needs for the held-out rows.

## Documentation Contract

Required public example, one page covering the whole workflow:

- `examples/05_machine_learning/01_plot_machine_learning_in_nimare.py`

The gallery only executes files matching `NN_plot_`, so the numeric prefix
stays even with a single example.

Required docs:

- API autosummary entries in `docs/api.rst`.
- Numpydoc docstrings for all public classes and functions.
- The 1,000-study conversion-and-split budget of <=3 minutes and <=5 GB peak
  memory, checked by a `performance_smoke` test.
