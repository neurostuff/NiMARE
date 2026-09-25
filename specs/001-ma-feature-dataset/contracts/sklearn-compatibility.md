# Compatibility Contract: Scikit-Learn Dataset Use

This contract defines the shape and behavior expected by downstream
scikit-learn workflows. **Revised 2026-09-25** alongside
`contracts/public-api.md`.

## Dataset Export

`MAFeatureDataset.to_sklearn()` returns a `sklearn.utils.Bunch` with:

- `data`: two-dimensional analysis-by-feature matrix accepted by scikit-learn
  estimators; same as `MAFeatureDataset.features`.
- `target`: one-dimensional target array or `None`.
- `groups`: one-dimensional study group array aligned to `data` rows; same as
  `MAFeatureDataset.study_ids`.
- `feature_names`: feature names aligned to `data` columns.
- `ids`, `descriptors`, `provenance`, `map_columns`, `descriptor_columns`: the
  NiMARE-side context a researcher needs to trace a row or build a pipeline.

`to_sklearn(return_X_y=True)` returns `(data, target)` instead, following the
`sklearn.datasets` convention. `MAFeatureExtractor.to_sklearn(studyset, ...)`
converts and exports in one call.

Exported unreduced voxelwise feature data must remain a sparse numeric matrix.
Dense feature data may be exported only after an explicit reducer produces a
reduced dense representation.

When coordinate-less analyses are retained, their rows must be represented as
all-zero sparse map-feature rows. When coordinate-less analyses are dropped,
all exported arrays and metadata must be aligned to the retained rows only.

**Feature Name Pairing**: Feature names are exported as a separate attribute
rather than paired with dense data (e.g., via DataFrame). This preserves
sparsity essential for neuroimaging datasets, and the names are built only when
they are asked for, because naming 200,000 voxels costs more memory than the
matrix being named. Users working with the exported bundle are responsible for
maintaining name alignment when transforming data, using standard sklearn
patterns like `get_feature_names_out()` or explicit DataFrame construction when
densification is acceptable.

## Study Groups

Study groups must be one-dimensional labels aligned to exported rows.

- The Studyset's per-analysis study ids are the study group source.
- MVP inputs are assumed to provide unique study IDs and unique analysis IDs;
  duplicate analysis ids must raise rather than collapse rows.
- Missing groups must raise a clear error before export or split.

## Row Alignment

Every exported array describes the same analyses, in the same order, and that
order survives export, splitting, selection and reduction.

Map rows must be matched to analyses **by analysis id**. Kernel transformers
return one row per analysis with coordinates, ordered by id, and
`return_type="sparse"` discards the ids; pairing by position is only correct
while the Studyset is in sorted order, which `Studyset.select_analyses` does
not guarantee.

## Splitting

Grouped splitting uses study group labels. For any split:

- `set(groups[train])` and `set(groups[test])` must be disjoint.
- `data`, `target`, `groups`, `ids` and `descriptors` must be sliced with the
  same analysis-row indices.
- The same dataset and `random_state` must produce the same split.
- `test_size` is a fraction of studies, not of analyses.
- Too few study groups, or a `test_size` that would empty a partition, must
  raise before returning any split.

`MAFeatureDataset.split` covers the holdout case; `groups` goes straight to
`GroupKFold`, `StratifiedGroupKFold` or `GroupShuffleSplit` for
cross-validation.

## Descriptor Features

Descriptor features must be aligned to `ids`.

- Numeric descriptors are appended directly.
- Categorical and text descriptors are rejected, because encoding them at
  extraction time would fit the encoder on every row, including held-out rows.
  The raw values remain available on `MAFeatureDataset.descriptors` for
  encoding inside a pipeline.
- Missing descriptor values must be reported explicitly unless
  `missing_values` says to drop or keep them, and either choice must be
  recorded in provenance.
- A field selected as the prediction target must not be silently reused as a
  descriptor feature.

## Prediction Target

Target values must be aligned to exported rows.

- Scalar categorical targets may be strings or encoded values.
- Numeric targets must preserve numeric values.
- Raw free-text and multi-label targets must be rejected unless the caller
  supplies an explicit target transformer or label extractor.
- Missing or constant targets must be diagnosed.
- A target transform that learns from the distribution of `y` belongs in
  `TransformedTargetRegressor`, not in extraction.

## Reduction Workflows

Reduction workflows must be compatible with scikit-learn estimator workflows.

- Reducers expose `fit`, `transform` and `fit_transform`, and support `clone`.
- Reducers must not densify unreduced voxelwise inputs as an intermediary.
  `make_preprocessor` sets `sparse_threshold=1.0` for the same reason.
- Reduced output must preserve row order and row count.
- Fitting on training rows only is enforced by the pipeline, or by
  `fit_transform_maps` on train and `transform_maps` on test; the latter raises
  `NotFittedError` rather than silently fitting on held-out data.
- Required initial workflows are variance thresholding, truncated SVD, and
  atlas/label aggregation when a compatible masker is supplied.

## Acceptance Checks

The following checks must pass in tests:

- A scikit-learn estimator can call `fit(data, target)` on exported data when a
  target is present.
- A grouped split can be created without study leakage, reproducibly.
- A pipeline fits a reducer on training data and transforms held-out data, both
  under `cross_val_score` and under `GridSearchCV`.
- Exported `groups` can be passed directly to scikit-learn group splitters.
- Non-numeric descriptors fail with a message naming the field and the routes
  that work.
- Raw free-text and multi-label targets fail without explicit target handling.
- Map rows stay with their own analysis when the Studyset is not in sorted
  order.
- Unreduced voxelwise data survives `make_preprocessor` still sparse.
- A representative 1,000-study conversion and grouped split meets the <=3
  minute and <=5 GB peak memory budget under the `performance_smoke` check.
