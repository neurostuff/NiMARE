# Data Model: Modeled Activation Feature Dataset

This data model is organized around the classes and public functions the
module provides. **Revised 2026-09-25** alongside `contracts/public-api.md`;
`interface-design.md` records why each shape was chosen. Some conceptual pieces are not standalone classes;
they are attributes or derived views on `FeatureSet`.

Terminology: in this feature, each dataset row represents one analysis. Any
references to dataset rows, indices, or grouping refer to analysis rows.

## Class: `nimare.nimads.Studyset` Existing Input

Existing input class. The new feature must use this public surface rather than
inventing a separate Studyset schema.

**Implementation-Backed Attributes and Properties**

- `id`: Studyset identifier.
- `name`: human-readable Studyset name.
- `studies`: materialized list of `nimare.nimads.Study` objects.
- `annotations`: materialized list of `nimare.nimads.Annotation` objects.
- `ids`: computed one-dimensional array of full analysis identifiers exposed by
  `Studyset.ids`. These IDs use the Studyset's full `<study_id>-<analysis_id>`
  convention and are the row-alignment source for conversion.
- `study_ids`: computed one-dimensional array of unique study identifiers
  exposed by `Studyset.study_ids`.
- `coordinates`: projected pandas table of coordinate rows, with an `id`
  column containing full analysis identifiers.
- `images`: projected pandas table of image references.
- `metadata`: projected pandas table of metadata fields, with an `id` column
  containing full analysis identifiers.
- `annotations_df`: flattened analysis-level annotation table, with an `id`
  column containing full analysis identifiers.
- `texts`: projected pandas table of title, abstract, description, or other
  text fields, with an `id` column containing full analysis identifiers.
- `space`: execution-space label for projected Studyset tables.
- `masker`: masker from the Studyset execution profile, defining masked voxel
  space when map generation requires a mask.
- `basepath`: base path used for resolving relative image paths.

**Used By**

- `extract_features` reads IDs, grouping, coordinates,
  projected tables, and masker from this object.

**Validation Rules**

- Must provide non-empty `ids` for analysis alignment.
- Must provide unique full analysis identifiers through `ids`.
- Must provide unique study identifiers through `study_ids`.
- May contain analyses with no coordinate rows in `coordinates`.
- Must expose a masker when map generation or reducer workflows require masked
  voxel ordering.

## Function: `extract_features` New Public Entry Point

Public function in `nimare.ml`. It converts one Studyset into one `FeatureSet`
in a single call. Splitting is not part of extraction: it is an evaluation
choice, and lives on the container. The conversion logic lives in an internal
`_FeatureExtractor` class, which keeps the stages as separate methods over
shared configuration but is not part of the public surface; users meet one
function and one container.

**Arguments**

- `studyset`: the Studyset to convert, positionally first.
- `kernel_transformer`: existing NiMARE kernel transformer instance or class.
- `descriptor_fields`: optional selectors for metadata, annotations, or texts.
- `target_field`: optional selector for one prediction target.
- `target_transformer`: optional callable or stateless transformer applied to
  the raw target values, for fields with no scalar reading.
- `missing_coordinates`: `"drop"` (default) or `"include"`.
- `missing_values`: `"raise"` (default), `"drop"` or `"keep"`.
- `memory`, `memory_level`: joblib cache location for MA map generation,
  applied to a copy of the kernel transformer when it does not define its own.
  Kernel transformers cache at memory level 2, so that is the default here.

**Behavior**

- Reads one `nimare.nimads.Studyset` (a `Dataset` is normalised at the
  boundary, as elsewhere in NiMARE).
- Validates the option vocabulary before doing any work.
- Rejects an empty Studyset and duplicate analysis identifiers.
- Reads per-analysis study ids for grouping, and the coordinate block to decide
  which analyses can have a map at all.
- Generates sparse MA features through the configured kernel transformer, and
  aligns them to analyses **by id**, with an all-zero row for each retained
  analysis that has no coordinates.
- Extracts descriptors and target from the Studyset tables, rejecting
  non-numeric descriptors and unusable targets.
- Applies `missing_coordinates` and `missing_values` to decide which analyses
  are retained, and records both decisions in provenance.
- Returns one `FeatureSet`; `FeatureSet.to_sklearn()` exports it.

**Validation Rules**

- `missing_coordinates="include"` keeps coordinate-less analyses as all-zero
  sparse map rows; `"drop"` removes them before row construction and records
  their ids in `provenance["dropped_ids"]`.
- Map rows that cannot be reconciled with the analyses that have coordinates
  must raise.
- Retained rows keep alignment among map features, descriptors, target, `ids`
  and `study_ids`.
- Cached map features are keyed by the kernel transformer and the coordinates
  it is given, so a changed configuration is a new entry rather than a stale
  hit.
- The caller's kernel transformer is never mutated.

## Class: `FeatureSet` New Container

Public class in `nimare.ml`. The authoritative NiMARE container for
machine-learning-ready map features, provenance, grouping, optional descriptors,
and an optional target.

### Public Attributes

- `ids`: one full Studyset analysis identifier per retained row.
- `study_ids`: one study-group label per retained row.
- `map_features`: the analysis-by-voxel block; sparse while unreduced.
- `descriptor_features`: the numeric descriptor block, or `None`.
- `descriptors`: the selected descriptor values as read from the Studyset, as a
  `DataFrame` indexed by `ids`, or `None`.
- `features`: the two blocks side by side, derived on first access.
- `feature_names`: names for `features`, built on first access.
- `target`: optional one-dimensional row-aligned prediction target.
- `masker`: the masker defining voxel order for unreduced map features.
- `provenance`: conversion settings and source Studyset details.
- `map_columns`, `descriptor_columns`, `shape`: column slices and dimensions.

The blocks are the source of truth and `features` is derived from them, so the
combined matrix cannot disagree with its parts. `features` and `feature_names`
are built lazily and cached, because a whole-brain mask has more voxel names
than the sparse matrix has bytes.

### Conceptual Components

#### Analysis Rows

- Represented by `ids`, `study_ids`, and row positions in `map_features`.
- Every retained row has exactly one full analysis ID and one study ID.
- Row order is identical across every array and is preserved by `split`,
  `select_analyses`, `copy` and the map-reduction methods.

#### Map Feature Matrix

- Represented by `map_features`, `feature_names` and `masker`.
- Unreduced voxelwise data stays sparse; a reducer may return dense output.
- Coordinate-less retained analyses are all-zero sparse rows.

#### Descriptor Feature Set

- Numeric descriptor fields are appended directly.
- Text and categorical descriptor fields are rejected; their raw values stay on
  `descriptors` so a pipeline can encode them per fold.

#### Prediction Target

- Scalar numeric targets preserve numeric values; scalar categorical targets may
  remain strings.
- Free-text and multi-label targets require `target_transformer`.
- Constant targets are diagnosed.

#### Study Groups

- Represented by `study_ids` and exported as sklearn `groups`.
- No study group may appear in more than one split partition.

### Methods and Derived Outputs

#### `to_sklearn(return_X_y=False)`

Returns a `sklearn.utils.Bunch` with `data`, `target`, `groups`,
`feature_names`, `ids`, `descriptors`, `provenance`, `map_columns` and
`descriptor_columns`, or `(data, target)`.

#### `split(test_size=0.25, random_state=None)`

Returns train/test `FeatureSet` slices through `GroupShuffleSplit` over
`study_ids`. `test_size` is a fraction of studies. Fails clearly, and before
returning anything, when the study count cannot serve the request.

#### `make_preprocessor(map_reducer, descriptor_transformer, **reducer_params)`

Returns an unfitted `ColumnTransformer` that reduces the map columns and
handles the descriptor columns separately, with `sparse_threshold=1.0`. This is
the piece that goes into a `Pipeline`, where scikit-learn fits it on training
rows only.

#### `fit_transform_maps(reducer)` / `transform_maps(reducer)`

Reduce the map block outside a pipeline. The first fits the reducer and returns
a reduced dataset; the second requires an already fitted reducer and raises
`NotFittedError` otherwise, so leakage is an error rather than an option.
Descriptors, target, `ids`, `study_ids` and provenance are preserved, and the
reduction is recorded in `provenance["map_reductions"]`.

#### `select_analyses(rows)` / `copy()`

Row selection by boolean mask or positions, and an independent copy.

## Function: `make_map_reducer(reducer, masker=None, **kwargs)`

Public function in `nimare.ml` returning an unfitted scikit-learn transformer
for whatever names or describes a reduction:

- a named workflow: `variance_threshold` (`VarianceThreshold`; sparse in,
  sparse out), `truncated_svd` (`TruncatedSVD`), or `atlas_aggregation`, whose
  atlas arrives through the `atlas` keyword;
- any scikit-learn transformer, used as given, or transformer class, built from
  `**kwargs`;
- any atlas nilearn can load, wrapped in an `AtlasAggregator`.

**Validation Rules**

- A string that is not a workflow raises `ValueError`; an object that is neither
  a transformer nor an atlas raises `TypeError`.
- Parameters alongside a built transformer raise rather than being ignored.
- Atlas aggregation requires the source masker, which
  `FeatureSet.make_preprocessor` supplies from the dataset.
- Unreduced map features are sparse; a reducer that cannot read sparse input
  says so when it is fitted.

## Class: `AtlasAggregator`

Public scikit-learn transformer in `nimare.ml`. Takes any atlas nilearn can
load -- a fetched atlas `Bunch`, a 3D or 4D atlas image, a path, the name of a
`nilearn.datasets.fetch_atlas_*` function with its `atlas_kwargs`, or a masker
the caller configured -- resolves it to a `NiftiMapsMasker` (4D) or
`NiftiLabelsMasker` (3D), fits it in the source mask's space, and summarises
batches of rows back through nilearn, so region definitions, resampling and
aggregation strategy remain nilearn's. Reports region names through
`get_feature_names_out()`, preferring the atlas's own labels.

## Hierarchy Summary

```text
nimare.nimads.Studyset  (existing input class)
`-- extract_features
  |-- transform(studyset)  -> FeatureSet
  `-- to_sklearn(studyset) -> sklearn Bunch (or (X, y))

FeatureSet
|-- ids, study_ids, masker, provenance
|-- map_features + descriptor_features -> features, feature_names
|-- descriptors (raw values for pipeline-side encoding)
|-- target
|-- to_sklearn(return_X_y=False)
|-- split(test_size, random_state) -> (train, test)
|-- make_preprocessor(...) -> ColumnTransformer for a Pipeline
|-- fit_transform_maps(reducer) / transform_maps(reducer)
`-- select_analyses(rows) / copy()

make_map_reducer(reducer, masker=None, **kwargs) -> sklearn transformer
|-- a named workflow, or any sklearn transformer or transformer class
`-- any nilearn atlas -> AtlasAggregator
```

## State Transitions

**Pipeline workflow (recommended):**

1. `Studyset` -> `extract_features(studyset, kernel_transformer, ...)`
2. `dataset.make_preprocessor(...)` inside a `Pipeline`
3. `cross_val_score(pipeline, bunch.data, bunch.target, cv=GroupKFold(...),
   groups=bunch.groups)`

**Holdout workflow:**

1. `Studyset` -> `extract_features(studyset, kernel_transformer, ...)`
2. `train, test = dataset.split(test_size=0.25, random_state=13)`
3. `train_reduced = train.fit_transform_maps(reducer)`;
   `test_reduced = test.transform_maps(reducer)`
4. `train_reduced.to_sklearn()` / `test_reduced.to_sklearn()`
