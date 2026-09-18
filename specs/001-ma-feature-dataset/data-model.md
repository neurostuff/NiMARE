# Data Model: Masked Activation Feature Dataset

This data model is organized around the classes and public functions that will
exist after implementation. Some conceptual pieces are not standalone classes;
they are attributes or derived views on `MAFeatureDataset`.

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

- `MAFeatureExtractor.transform(studyset)` reads IDs, grouping, coordinates,
  projected tables, and masker from this object.

**Validation Rules**

- Must provide non-empty `ids` for analysis alignment.
- Must provide unique full analysis identifiers through `ids`.
- Must provide unique study identifiers through `study_ids`.
- May contain analyses with no coordinate rows in `coordinates`.
- Must expose a masker when map generation or reducer workflows require masked
  voxel ordering.

## Class: `MAFeatureExtractor` New Conversion Helper

New public class in `nimare.ml`. It stores conversion configuration and orchestrates
the full pipeline from Studyset to train/test outputs. It is not a
scikit-learn estimator and must not expose `fit` or `fit_transform`.

**Constructor Configuration**

- `kernel_transformer`: existing NiMARE kernel transformer instance or class.
- `descriptor_fields`: optional selectors for metadata, annotations, or texts.
- `descriptor_transformers`: optional transformers/vectorizers for non-numeric
  descriptor fields.
- `target_field`: optional selector for one prediction target.
- `target_transformer`: optional transformer or label extractor for unsupported
  target shapes.
- `missing_coordinates`: conversion option, either `include` or `drop`;
  default is `drop`.
- `test_size`: optional fraction or count for train/test split; default `None`
  means no split (return full dataset as train, `None` as test).
- `random_state`: random seed for reproducible splits.
- `cache_maps`: whether MA map generation should be cached across repeated
  calls that reuse the same Studyset and extraction settings; default `True`.
- `memory`, `memory_level`: optional caching controls passed through where
  compatible with NiMARE conventions.

**Behavior**

- Reads from one `nimare.nimads.Studyset`.
- Applies `missing_coordinates` before constructing map-feature rows.
- Generates sparse modeled activation map features through the configured
  kernel transformer, with extractor-level caching so repeated reducer choices
  do not recompute MA maps.
- Extracts optional descriptor and target data from Studyset projected tables.
- If `test_size` is set (not `None`), performs grouped train/test split by study ID.
- Fits descriptor/target transformers on training data when a split is requested,
  then applies them to train and test separately.
- `transform(studyset)` performs cached extraction and optional splitting, and
  returns a tuple `(train_dataset, test_dataset)`.
  - If `test_size` is `None` or `0.0`, returns `(full_dataset, None)`.
- `to_sklearn(studyset, map_reducer=None, map_reducer_params=None)` runs the
  full public pipeline as a convenience wrapper over `transform(...)`,
  returning a tuple of sklearn-compatible Bunch objects:
  `(train_bunch, test_bunch)`.
  - If `map_reducer` is provided, it is fit on training map features only and
    then applied to train and test.
  - If `test_size` is `None` or `0.0`, returns `(full_bunch, None)`.

**Validation Rules**

- `missing_coordinates="include"` keeps coordinate-less analyses in
  `ids` as all-zero sparse map rows.
- `missing_coordinates="drop"` removes coordinate-less analyses before row
  construction and records dropped analysis IDs in `MAFeatureDataset.provenance`.
- Invalid map rows unrelated to missing coordinates must fail clearly.
- Dropping coordinate-less analyses must preserve row alignment among
  `features`, `ids`, `study_ids`, descriptors, and target values.
- Descriptor/target transformers must fit on training data only when a split is
  requested.
- Map reducers must fit on training map features only when a split is requested.
- Cached map features must be invalidated whenever the Studyset content,
  kernel-transformer configuration, mask space/order, or missing-coordinate
  policy changes.

## Class: `MAFeatureDataset` New Container

New public class in `nimare.ml`. This is the authoritative NiMARE container for
machine-learning-ready map features, provenance, grouping, optional descriptors,
and optional target values.

### Public Attributes

Users interact with these attributes directly.

- `ids`: one full Studyset analysis identifier per retained row.
- `study_ids`: one study-group label per retained row.
- `features`: analysis-by-feature matrix combining map features (sparse voxelwise)
  and optional descriptor features (if extracted). Reduced map features may be
  dense only after an explicit map reducer returns a lower-dimensional matrix.
- `feature_names`: names for features in `features` column order, covering both
  voxel features and descriptor features when present.
- `target`: optional one-dimensional row-aligned prediction target.
- `provenance`: conversion settings and source Studyset details, including
  `missing_coordinates` and any `dropped_ids`.

### Private Attributes

Used internally for book-keeping and method implementation; not part of the
public API.

- `_map_features`: sparse analysis-by-voxel matrix underlying `features`.
- `_descriptor_features`: optional row-aligned descriptor feature matrix
  underlying `features`; `None` if no descriptors were extracted.
- `_masker`: masker defining voxel order for unreduced map features and
  atlas/label aggregation.

### Conceptual Components Within `MAFeatureDataset`

These are not separate public classes in the MVP; they are attributes,
metadata, or derived views of `MAFeatureDataset`.

#### Analysis Rows

Purpose: define the analysis axis shared by all row-aligned data.

- Represented by `ids`, `study_ids`, and row positions in `_map_features`.
- Every retained row has exactly one full analysis ID and one study ID.
- Row order must be identical across `features`, `target`, and split outputs.

#### Map Feature Matrix

Purpose: store modeled activation map features.

- Represented by `_map_features`, `feature_names`, and `_masker`.
- Row count must equal `len(ids)`.
- Unreduced voxelwise data must remain sparse.
- Coordinate-less retained analyses must be all-zero sparse rows when
  `missing_coordinates="include"`.
- The masker defines voxel ordering for unreduced features and atlas/label
  aggregation.

#### Descriptor Feature Set

Purpose: optional non-map predictors extracted from Studyset metadata,
annotations, or texts.

- Represented internally by `_descriptor_features` and exposed via `features`
  and `feature_names`.
- Numeric descriptor fields may be appended directly.
- Text and categorical descriptor fields must be rejected unless an explicit
  transformer or vectorizer converts them into numeric features.
- Descriptor transformer fitting must happen on training rows only before
  held-out transformation.

#### Prediction Target

Purpose: optional supervised outcome exported as `y`.

- Represented by `target`.
- Target length must equal `len(ids)`.
- Scalar numeric targets preserve numeric values.
- Scalar categorical targets may remain strings or encoded values.
- Raw free-text and multi-label targets must be rejected unless an explicit
  target transformer or label extractor is supplied.
- Study-level targets may repeat across analyses, but grouped splitting must
  keep repeated study-level targets in one partition.

#### Study Groups

Purpose: prevent study leakage during splitting.

- Represented by `study_ids` and exported sklearn `groups`.
- Every retained row must have exactly one study group.
- No study group may appear in more than one split partition.
- Missing study groups are outside the MVP input contract and must fail before
  any split is returned.

### Methods and Derived Outputs

#### `to_sklearn()`

Returns a `sklearn.utils.Bunch`-compatible object with whatever was extracted
during conversion.

- `data`: same as `features` (map features + descriptor features if extracted,
  otherwise map features only).
- `target`: same as `target` if target was extracted, otherwise `None`.
- `groups`: same values as `study_ids`.
- `feature_names`: same as `feature_names`.
- Unreduced voxelwise `data` must remain sparse.

**Design Note**: Feature names are exported separately from `data` to preserve
sparsity. Dense pairing of names with data (e.g., via pandas DataFrame) would
multiply memory costs by orders of magnitude for neuroimaging datasets with
thousands to millions of voxels. Users who need automatic name tracking for
sklearn pipelines should either: (1) construct a DataFrame explicitly at the
point of use when dataset size permits densification, or (2) use sklearn
transformer `get_feature_names_out()` methods to track names through explicit
transformations.

#### `split(test_size=0.25, random_state=None, cv=None)`

Returns train/test `MAFeatureDataset` slices.

- Uses `study_ids` as sklearn groups.
- Slices `features`, `target`, `ids`, `study_ids`, and provenance
  consistently.
- Must fail clearly when too few study groups are available.

#### `apply_map_reducer(reducer, fit=False)`

Returns an `MAFeatureDataset` copy with transformed map features.

- Applies only to `_map_features`.
- Preserves descriptor features, `target`, `ids`, `study_ids`, and provenance.
- Validates that transformed row count and row order match the input dataset.
- Updates `features` and `feature_names` to reflect the transformed map
  dimensions.

#### `copy()`

Returns an independent copy of the dataset container.

## Function: `make_map_reducer(method, **kwargs)`

New public function in `nimare.ml`. It returns a scikit-learn-compatible
transformer or pipeline, not a NiMARE data entity.

**Reducer Types**

- `variance_threshold`: sparse-compatible map-feature filtering.
- `truncated_svd`: sparse-compatible low-rank map-feature reduction.
- `atlas_aggregation`: atlas/label aggregation using a masker-compatible labels
  image or atlas.

**Validation Rules**

- Reducers must fit only on training map features before held-out
  transformation.
- Reducers must not densify unreduced voxelwise inputs as an intermediary.
- Reduced outputs may be dense when the reducer explicitly returns a
  lower-dimensional component or parcel matrix.
- Atlas/label aggregation must use the dataset masker to align atlas labels to
  voxel columns.

## Hierarchy Summary

```text
nimare.nimads.Studyset  (existing input class)
|-- MAFeatureExtractor.to_sklearn(studyset, map_reducer=None, ...)
|   |-- Convenience wrapper over extraction/splitting/reduction
|   `-- Returns: (train_bunch, test_bunch)
`-- MAFeatureExtractor.transform(studyset)  (advanced orchestration method)
  |-- Extraction stage: kernel features, descriptors, target
  |-- Split stage (optional): train/test by study ID
  `-- Returns: (train_MAFeatureDataset, test_MAFeatureDataset|None)
    |-- Dataset API (per dataset):
    |   |-- ids, study_ids
    |   |-- features, feature_names
    |   |-- target
    |   |-- provenance
    |   |-- to_sklearn()
    |   |-- split() [for manual splits]
    |   |-- apply_map_reducer(reducer) [for manual reduction]
    |   `-- copy()
    `-- Private internals:
      |-- _map_features, _descriptor_features
      `-- _masker

make_map_reducer(method, **kwargs)  (new reducer factory function)
`-- sklearn-compatible reducer consumed by MAFeatureExtractor or 
    MAFeatureDataset.apply_map_reducer()
```

## State Transitions

**Simple pipeline (automatic):**
1. `Studyset` + configured `MAFeatureExtractor` → `to_sklearn(studyset, ...)`
2. Internal cached extraction + optional split + optional reduction
3. Returns `(train_bunch, test_bunch)`

**Advanced pipeline (manual control):**
1. `Studyset` + `MAFeatureExtractor` → `transform(studyset)`
2. Returns `(train_dataset, test_dataset|None)`
3. User iterates reducers via `make_map_reducer(...)` +
  `apply_map_reducer(...)` (reusing cached maps by rerunning `to_sklearn` on
  the same extractor and studyset)
4. User exports via dataset-level `to_sklearn()`
