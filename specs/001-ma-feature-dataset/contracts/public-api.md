# Public API Contract: `nimare.ml`

This contract defines the additive public surface for converting NiMARE
Studysets into scikit-learn-compatible feature datasets.

**Revised 2026-09-25** against `interface-design.md`, which records the options
weighed for each decision below. Terminology: each dataset row is an analysis
row.

## Module

`nimare.ml`

The module is exported from `nimare/__init__.py` and documented in
`docs/api.rst`. Its public names are `MAFeatureExtractor`, `MAFeatureDataset`,
`AtlasAggregator` and `make_map_reducer`.

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

## `MAFeatureExtractor`

Converts a Studyset into feature data. It is a NiMARE conversion helper, not a
trainable scikit-learn estimator, and must not expose `fit` or `fit_transform`.

### Construction

Required parameter:

- `kernel_transformer`: existing NiMARE kernel transformer instance or class.
  No implicit scientific default is selected; public examples must pass an
  explicit kernel transformer.

Optional parameters:

- `descriptor_fields`: field selectors appended to the feature matrix as extra
  numeric columns.
- `target_field`: field selector for `y`.
- `target_transformer`: callable or stateless transformer applied to the raw
  target values, for free-text and multi-label fields that need a label
  extractor.
- `missing_coordinates`: `"drop"` (default) or `"include"`.
- `missing_values`: `"raise"` (default), `"drop"` or `"keep"`, for missing
  descriptor and target values.
- `cache_maps`: keep the most recently generated map matrix in memory, so that
  comparing reducers over one Studyset generates the maps once. Default `True`.
- `memory`, `memory_level`: joblib cache location for MA map generation,
  applied to the kernel transformer when it does not define its own, following
  the NiMARE `CacheMixin` convention.

The constructor must not raise for option values it can check later;
`transform` validates the option vocabulary before doing any work.

### Required methods

- `transform(studyset)`: validate the Studyset, extract map features,
  descriptors and target, and return one `MAFeatureDataset`. Splitting is not
  part of extraction.
- `to_sklearn(studyset, return_X_y=False)`: convert and export in one call,
  returning the `Bunch` (or `(data, target)`) that
  `MAFeatureDataset.to_sklearn` returns.

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
  encode it and select the numeric result, or encode it inside a pipeline from
  `MAFeatureDataset.descriptors`.
- Export scalar numeric and scalar categorical targets as a one-dimensional
  `y`. Reject raw free-text and multi-label targets unless `target_transformer`
  is supplied, and reject a target that has one value for every analysis.
- Report missing descriptor and target values under `missing_values="raise"`,
  naming the fields and the affected analysis ids; record them in provenance
  under `"drop"` and `"keep"`.
- Never mutate the caller's kernel transformer.

## `MAFeatureDataset`

The aligned container. Row `i` is analysis `ids[i]` from study `study_ids[i]`,
and that order is preserved by every method.

### Required attributes

- `ids`: full Studyset analysis identifiers, `<study_id>-<analysis_id>`.
- `study_ids`: one study-group label per row.
- `map_features`: the analysis-by-voxel block. Sparse while unreduced.
- `descriptor_features`: the numeric descriptor block, or `None`.
- `descriptors`: the selected descriptor values as read from the Studyset, as a
  `DataFrame` indexed by `ids`, or `None`.
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
  `target`, `groups`, `feature_names`, `ids`, `descriptors`, `provenance`,
  `map_columns` and `descriptor_columns`; or `(data, target)`, following the
  `sklearn.datasets` convention.
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

A selector names one descriptor or target field, using the vocabulary NiMARE
estimators already use in `_required_inputs`:

- a bare field name (`"sample_sizes"`), looked up in metadata, annotations and
  texts in turn; an ambiguous name raises and asks for the explicit form;
- a `(source, field)` tuple (`("annotations", "motor_label")`);
- a mapping with `source` and `field`.

A tuple is one selector; a list holds several. Sources are `"metadata"`,
`"annotations"` and `"texts"`, with `"annotations_df"` and `"text"` accepted as
aliases.

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

## Reduction Helpers

`make_map_reducer(reducer, masker=None, **kwargs)` returns an unfitted
scikit-learn transformer. It is deliberately permissive about what names a
reduction, because the module should not stand between a researcher and either
of the libraries underneath it:

- a named workflow -- `"variance_threshold"` (`VarianceThreshold`),
  `"truncated_svd"` (`TruncatedSVD`), or `"atlas_aggregation"`, which takes its
  atlas through the `atlas` keyword;
- any scikit-learn transformer, used as given;
- any scikit-learn transformer class, built from `**kwargs`;
- any atlas nilearn can load, wrapped in an `AtlasAggregator`.

A name that is not a workflow raises `ValueError` and names the workflows and
the alternatives to a name. An object that is neither a transformer nor an
atlas raises `TypeError`. Parameters passed alongside an already-built
transformer raise, rather than being silently ignored.

`MAFeatureDataset.make_preprocessor` takes the same forms and supplies the
dataset's masker, so an atlas needs nothing else from the caller.

Unreduced map features are sparse, so a reducer that cannot read sparse input
(`PCA`, for one) fails when it is fitted, with scikit-learn's own message.
That is documented rather than guarded: which reducers are appropriate is the
researcher's call.

`AtlasAggregator` is public and accepts any atlas nilearn can load:

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

## Documentation Contract

Required public examples:

- `examples/05_machine_learning/01_plot_ma_feature_dataset.py`
- `examples/05_machine_learning/02_plot_ma_feature_reduction.py`

Required docs:

- API autosummary entries in `docs/api.rst`.
- Numpydoc docstrings for all public classes and functions.
- The 1,000-study conversion-and-split budget of <=3 minutes and <=5 GB peak
  memory, checked by a `performance_smoke` test.
