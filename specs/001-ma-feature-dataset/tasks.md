# Tasks: Masked Activation Feature Dataset

**Input**: Design documents from `specs/001-ma-feature-dataset/`
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Use thin tests early to protect the garden-path behavior while the
implementation shape is still settling. Add focused edge-condition tests after
the first end-to-end sklearn-compatible pipeline works.

**Organization**: Tasks are ordered by implementation path first: shared setup,
garden-path implementation with perfect inputs, basic edge conditions, then the
remaining user-story hardening and documentation.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel because it touches different files and has no dependency on incomplete tasks in the same phase.
- **[Story]**: User story label from `specs/001-ma-feature-dataset/spec.md` when the task maps to a specific story.
- Every checklist task includes exact repository-relative file paths.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the minimum files and API hooks needed to start the garden-path implementation.

- [ ] T001 Create additive `nimare.ml` module scaffold with `__all__`, imports, and placeholder public objects for `MAFeatureDataset`, `MAFeatureExtractor`, and `make_map_reducer` in `nimare/ml.py`
- [ ] T002 [P] Create the machine-learning test module with shared imports and sparse/sklearn assertion helpers in `nimare/tests/test_ml.py`
- [ ] T003 [P] Create the Sphinx-Gallery dataset workflow example scaffold with public imports in `examples/05_machine_learning/01_plot_ma_feature_dataset.py`
- [ ] T004 [P] Create the Sphinx-Gallery reduction workflow example scaffold with public imports in `examples/05_machine_learning/02_plot_ma_feature_reduction.py`
- [ ] T005 Export the additive `ml` module from `nimare/__init__.py` without changing existing released imports in `nimare/__init__.py`
- [ ] T006 Add the `nimare.ml` autosummary target placeholder in `docs/api.rst`

---

## Phase 2: Garden-Path Foundation (Perfect Inputs Only)

**Purpose**: Build only the fixture and helper surface needed for a perfect-input workflow.

- [ ] T007 Add one perfect Studyset fixture builder with valid coordinates, a masker, numeric metadata, numeric annotations, text fields, unique study IDs, and unique full analysis IDs in `nimare/tests/test_ml.py`
- [ ] T008 Add shared assertions for sklearn Bunch shape, sparse `data`, aligned `target`, aligned `groups`, and estimator-fit compatibility in `nimare/tests/test_ml.py`
- [ ] T009 Add internal helper placeholders for Studyset table access, sparse feature stacking, grouped splitting, and reducer application in `nimare/ml.py`

**Checkpoint**: Foundation supports one clean fixture and a single end-to-end happy-path test.

---

## Phase 3: Garden-Path End-to-End Pipeline (Cross-Story MVP)

**Goal**: Produce usable sklearn datasets from a perfect Studyset, including numeric descriptors, scalar target, grouped train/test split, feature reduction, and estimator fitting.

**Independent Test**: A single test converts a perfect Studyset with `MAFeatureExtractor.to_sklearn(studyset, map_reducer="truncated_svd")`, receives train/test Bunches, and fits a scikit-learn estimator on `train_bunch.data` and `train_bunch.target`.

### Thin Tests and Example

- [ ] T010 [US1] Add one failing garden-path test for `MAFeatureExtractor.to_sklearn(studyset, map_reducer="truncated_svd")` returning train/test sklearn Bunches with sparse-or-reduced numeric `data`, aligned `target`, aligned `groups`, and `feature_names` in `nimare/tests/test_ml.py`
- [ ] T011 [US2] Extend the garden-path test to assert no study ID appears in both train and test `groups` in `nimare/tests/test_ml.py`
- [ ] T012 [US3] Extend the garden-path test to include one numeric descriptor field in exported `data` and `feature_names` in `nimare/tests/test_ml.py`
- [ ] T013 [US4] Extend the garden-path test to fit a simple scikit-learn estimator using `train_bunch.data` and `train_bunch.target` in `nimare/tests/test_ml.py`
- [ ] T014 [US5] Extend the garden-path test to assert the reducer lowers map-feature dimensionality before descriptor columns are appended in `nimare/tests/test_ml.py`
- [ ] T015 [P] [US1] Draft the happy-path dataset example using perfect inputs, numeric descriptors, scalar target, `test_size`, and `extractor.to_sklearn(studyset, ...)` in `examples/05_machine_learning/01_plot_ma_feature_dataset.py`

### Implementation

- [ ] T016 [US1] Implement `MAFeatureDataset.__init__` with public `ids`, `study_ids`, `features`, `feature_names`, `target`, and `provenance` attributes plus private `_map_features`, `_descriptor_features`, and `_masker` in `nimare/ml.py`
- [ ] T017 [US1] Implement minimal `MAFeatureDataset.to_sklearn()` returning a `sklearn.utils.Bunch` with `data`, `target`, `groups`, and `feature_names` in `nimare/ml.py`
- [ ] T018 [US1] Implement minimal `MAFeatureDataset.copy()` and internal row slicing for perfect aligned inputs in `nimare/ml.py`
- [ ] T019 [US1] Implement `MAFeatureExtractor.__init__` with `kernel_transformer`, `descriptor_fields`, `target_field`, `missing_coordinates`, `test_size`, `random_state`, `cache_maps`, `memory`, and `memory_level` storage in `nimare/ml.py`
- [ ] T020 [US1] Implement Studyset-native access for `ids`, `study_ids`, `coordinates`, `metadata`, `annotations_df`, `texts`, `masker`, `space`, and `basepath` for perfect Studysets in `nimare/ml.py`
- [ ] T021 [US1] Implement sparse MA map extraction through `KernelTransformer.transform(..., return_type="sparse")` and row alignment to Studyset `ids` for perfect Studysets in `nimare/ml.py`
- [ ] T022 [US3] Implement numeric descriptor extraction from Studyset metadata and annotations with retained-row alignment in `nimare/ml.py`
- [ ] T023 [US4] Implement scalar numeric and scalar categorical target extraction with retained-row alignment in `nimare/ml.py`
- [ ] T024 [US2] Implement grouped train/test splitting with `GroupShuffleSplit`, `test_size`, `random_state`, and `study_ids` as groups in `nimare/ml.py`
- [ ] T025 [US5] Implement `make_map_reducer("truncated_svd", **kwargs)` using scikit-learn `TruncatedSVD` in `nimare/ml.py`
- [ ] T026 [US5] Implement `MAFeatureDataset.apply_map_reducer(reducer, fit=False)` for the truncated-SVD happy path while preserving `ids`, `study_ids`, descriptors, `target`, and `provenance` in `nimare/ml.py`
- [ ] T027 [US5] Implement `MAFeatureExtractor.to_sklearn(studyset, map_reducer=None, map_reducer_params=None)` as the happy-path pipeline that extracts, splits, fits reducer on train rows, transforms test rows, and exports Bunches in `nimare/ml.py`
- [ ] T028 [US1] Implement extractor-level map cache reuse for repeated calls with unchanged perfect Studyset and extraction settings in `nimare/ml.py`
- [ ] T029 [US1] Run and fix the garden-path test with `python -m pytest nimare/tests/test_ml.py -k "garden_path"` for `nimare/tests/test_ml.py`

**Checkpoint**: A clean Studyset can be converted into train/test sklearn Bunches that fit a sklearn estimator.

---

## Phase 4: Basic Edge Conditions

**Goal**: Add only the edge handling already called out as MVP behavior, after the happy path works.

**Independent Test**: Focused tests cover missing-coordinate policy, sparse unreduced data, unsupported non-numeric fields, too-few-group splitting, target shape rejection, and reducer scope.

### Focused Edge Tests

- [ ] T030 [US1] Add focused tests that `MAFeatureExtractor` exposes no public `fit` or `fit_transform` attributes in `nimare/tests/test_ml.py`
- [ ] T031 [US1] Add focused tests that `missing_coordinates="drop"` is default and records dropped IDs in `provenance` in `nimare/tests/test_ml.py`
- [ ] T032 [US1] Add focused tests that `missing_coordinates="include"` keeps coordinate-less analyses as all-zero sparse map rows in `nimare/tests/test_ml.py`
- [ ] T033 [US1] Add focused tests that unreduced voxelwise `features` and Bunch `data` remain sparse before reduction in `nimare/tests/test_ml.py`
- [ ] T034 [US2] Add focused tests that too-few-study split requests raise a clear `ValueError` before returning partial outputs in `nimare/tests/test_ml.py`
- [ ] T035 [US3] Add focused tests that categorical/text descriptors raise without an explicit transformer or vectorizer in `nimare/tests/test_ml.py`
- [ ] T036 [US4] Add focused tests that raw free-text and multi-label targets raise without an explicit target transformer or label extractor in `nimare/tests/test_ml.py`
- [ ] T037 [US5] Add focused tests that reducers operate on internal `_map_features` only and do not reduce descriptor columns in `nimare/tests/test_ml.py`

### Edge Implementations

- [ ] T038 [US1] Implement `missing_coordinates` default `drop`, explicit `include`, invalid option errors, all-zero sparse rows, and `dropped_ids` provenance in `nimare/ml.py`
- [ ] T039 [US1] Tighten sparse validation so unreduced voxelwise public `features` and Bunch `data` cannot be emitted dense in `nimare/ml.py`
- [ ] T040 [US1] Add basic `ValueError` messages for missing masker, missing Studyset IDs, invalid map rows, and incompatible sparse map shape in `nimare/ml.py`
- [ ] T041 [US2] Add basic split diagnostics for missing groups, too few groups, and invalid `test_size` in `nimare/ml.py`
- [ ] T042 [US3] Implement non-numeric descriptor rejection with messages identifying the selector source and field in `nimare/ml.py`
- [ ] T043 [US4] Implement raw free-text and multi-label target rejection with messages identifying the selector source and field in `nimare/ml.py`
- [ ] T044 [US5] Enforce reducer scope by applying map reducers only to `_map_features` and rebuilding public `features` afterward in `nimare/ml.py`
- [ ] T045 [US1] Run and fix edge-condition tests with `python -m pytest nimare/tests/test_ml.py -k "missing_coordinates or sparse or non_numeric or target or too_few or reducer_scope"` for `nimare/tests/test_ml.py`

**Checkpoint**: Basic MVP failure modes are explicit without locking down every possible diagnostic yet.

---

## Phase 5: User Story 1 Completion - Convert Studysets Into Feature Data (Priority: P1)

**Goal**: Finish conversion details that were intentionally skipped during the garden path.

**Independent Test**: Map-only conversion and Bunch export work without descriptors, target, split, or reducer.

- [ ] T046 [US1] Add focused tests for map-only `transform(studyset)` returning `(full_dataset, None)` when `test_size=None` in `nimare/tests/test_ml.py`
- [ ] T047 [US1] Add focused tests that map-only `to_sklearn(studyset)` returns `(full_bunch, None)` with `target is None` in `nimare/tests/test_ml.py`
- [ ] T048 [US1] Add focused tests that MVP inputs use Studyset-provided unique `ids` and `study_ids` without duplicate-ID reconciliation or fallback parsing in `nimare/tests/test_ml.py`
- [ ] T049 [US1] Complete map-only code paths for `transform(studyset)` and `to_sklearn(studyset)` without descriptors, target, split, or reducer in `nimare/ml.py`
- [ ] T050 [US1] Complete provenance for source Studyset details, map generation settings, `missing_coordinates`, and `dropped_ids` in `nimare/ml.py`
- [ ] T051 [US1] Add Numpydoc docs for `MAFeatureDataset`, `MAFeatureExtractor`, `transform`, `to_sklearn`, and `copy` in `nimare/ml.py`
- [ ] T052 [US1] Finalize the conversion/provenance section of the dataset example in `examples/05_machine_learning/01_plot_ma_feature_dataset.py`

---

## Phase 6: User Story 2 Completion - Split Data Without Study Leakage (Priority: P1)

**Goal**: Finish manual dataset-level splitting and grouped CV support after extractor-level split works.

**Independent Test**: `MAFeatureDataset.split(...)` returns aligned train/test datasets and repeated split settings are reproducible.

- [ ] T053 [US2] Add focused tests for `MAFeatureDataset.split(test_size, random_state)` reproducibility and no study leakage in `nimare/tests/test_ml.py`
- [ ] T054 [US2] Add focused tests that manual split slices `features`, `ids`, `study_ids`, `target`, and `provenance` consistently in `nimare/tests/test_ml.py`
- [ ] T055 [US2] Implement public `MAFeatureDataset.split(test_size=0.25, random_state=None, cv=None)` using sklearn group splitters in `nimare/ml.py`
- [ ] T056 [US2] Implement grouped CV behavior for the `cv` parameter or raise a documented MVP `ValueError` if deferred in `nimare/ml.py`
- [ ] T057 [US2] Extend the dataset example with manual `feature_dataset.split(...)` and `groups == study_ids` checks in `examples/05_machine_learning/01_plot_ma_feature_dataset.py`

---

## Phase 7: User Story 3 Completion - Combine Activation Maps With Study Information (Priority: P2)

**Goal**: Finish descriptor transformers and descriptor diagnostics after numeric descriptors work.

**Independent Test**: Numeric descriptors append directly; explicit descriptor transformers/vectorizers work; missing descriptor values are reported.

- [ ] T058 [US3] Add focused tests for descriptor selector normalization across `metadata`, `annotations`, and `texts` sources in `nimare/tests/test_ml.py`
- [ ] T059 [US3] Add focused tests that explicit descriptor vectorizers produce numeric columns and feature names in `nimare/tests/test_ml.py`
- [ ] T060 [US3] Add focused tests that descriptor transformers fit only on training rows and transform held-out rows when `test_size` is set in `nimare/tests/test_ml.py`
- [ ] T061 [US3] Add focused tests for missing descriptor diagnostics listing affected `ids` and field selectors in `nimare/tests/test_ml.py`
- [ ] T062 [US3] Implement descriptor selector normalization for `metadata`, `annotations`, and `texts` in `nimare/ml.py`
- [ ] T063 [US3] Implement explicit descriptor transformer/vectorizer support with train-only fitting when split in `nimare/ml.py`
- [ ] T064 [US3] Implement missing descriptor diagnostics in `provenance` without silent row dropping or filling in `nimare/ml.py`
- [ ] T065 [US3] Implement guard that prevents a selected target field from being silently reused as a descriptor in `nimare/ml.py`
- [ ] T066 [US3] Extend the dataset example with numeric descriptors and explicit text vectorization in `examples/05_machine_learning/01_plot_ma_feature_dataset.py`

---

## Phase 8: User Story 4 Completion - Predict Selected Study Information (Priority: P2)

**Goal**: Finish target transformers and target diagnostics after scalar targets work.

**Independent Test**: Scalar numeric/categorical targets export as `y`; explicit target transformers work; missing and constant targets are diagnosed.

- [ ] T067 [US4] Add focused tests for scalar categorical targets, including string labels accepted by a downstream sklearn classifier in `nimare/tests/test_ml.py`
- [ ] T068 [US4] Add focused tests that study-level targets may repeat across analyses and grouped splits keep repeated targets in one partition in `nimare/tests/test_ml.py`
- [ ] T069 [US4] Add focused tests for missing target reports and constant target diagnostics in `nimare/tests/test_ml.py`
- [ ] T070 [US4] Add focused tests that explicit target transformers or label extractors produce a one-dimensional target aligned to retained `ids` in `nimare/tests/test_ml.py`
- [ ] T071 [US4] Implement scalar categorical target support and target dtype preservation in `nimare/ml.py`
- [ ] T072 [US4] Implement missing target reports and constant target diagnostics in `provenance` in `nimare/ml.py`
- [ ] T073 [US4] Implement explicit target transformer or label extractor support with retained-row alignment in `nimare/ml.py`
- [ ] T074 [US4] Preserve target alignment through `to_sklearn`, `split`, row slicing, and reduced dataset copies in `nimare/ml.py`
- [ ] T075 [US4] Extend the dataset example with target extraction, exported `y`, and a minimal sklearn estimator fit on train data in `examples/05_machine_learning/01_plot_ma_feature_dataset.py`

---

## Phase 9: User Story 5 Completion - Reduce Voxelwise Feature Dimensionality (Priority: P3)

**Goal**: Finish all required reducers after truncated SVD works.

**Independent Test**: Variance thresholding, truncated SVD, and atlas aggregation reduce map features while preserving rows, descriptors, targets, and groups.

- [ ] T076 [US5] Add focused tests for `make_map_reducer("variance_threshold", **kwargs)` on sparse map features in `nimare/tests/test_ml.py`
- [ ] T077 [US5] Add focused tests for atlas/label aggregation using the stored dataset `_masker` to align labels to voxel columns in `nimare/tests/test_ml.py`
- [ ] T078 [US5] Add focused tests that reduced datasets preserve `ids`, `study_ids`, `target`, descriptor features, and `provenance` in `nimare/tests/test_ml.py`
- [ ] T079 [US5] Add focused tests that reduced `feature_names` distinguish variance-threshold, SVD component, and atlas parcel outputs in `nimare/tests/test_ml.py`
- [ ] T080 [US5] Implement `make_map_reducer("variance_threshold", **kwargs)` using sparse-compatible variance filtering in `nimare/ml.py`
- [ ] T081 [US5] Implement `make_map_reducer("atlas_aggregation", atlas=..., masker=..., **kwargs)` contract and validation in `nimare/ml.py`
- [ ] T082 [US5] Implement atlas-to-mask alignment using the stored dataset `_masker`, voxel ordering, and a sparse voxel-by-parcel aggregation matrix in `nimare/ml.py`
- [ ] T083 [US5] Implement reduced feature naming for variance-threshold, SVD component, and atlas parcel outputs in `nimare/ml.py`
- [ ] T084 [US5] Fill the reduction workflow example with grouped split, train-fit truncated SVD, held-out transform, and atlas aggregation in `examples/05_machine_learning/02_plot_ma_feature_reduction.py`

---

## Phase 10: Performance, Documentation, and Compatibility

**Purpose**: Add slow checks and finish documentation after the public implementation has stabilized.

- [ ] T085 Add a `performance_smoke` test for 1,000-study sparse conversion and grouped split within <=3 minutes and <=5 GB peak memory in `nimare/tests/test_ml.py`
- [ ] T086 [P] Finalize Numpydoc parameter docs, return docs, examples, warnings, and errors for public objects in `nimare/ml.py`
- [ ] T087 [P] Update API documentation text for `nimare.ml`, `MAFeatureDataset`, `MAFeatureExtractor`, and `make_map_reducer` in `docs/api.rst`
- [ ] T088 [P] Verify both Sphinx-Gallery examples use only public APIs and match `specs/001-ma-feature-dataset/quickstart.md` in `examples/05_machine_learning/01_plot_ma_feature_dataset.py` and `examples/05_machine_learning/02_plot_ma_feature_reduction.py`
- [ ] T089 Run the targeted feature test suite with `python -m pytest nimare/tests/test_ml.py` for `nimare/tests/test_ml.py`
- [ ] T090 Run the performance smoke check with `python -m pytest -m performance_smoke nimare/tests/test_ml.py` for `nimare/tests/test_ml.py`
- [ ] T091 Run the documentation build with `make -C docs html` for `docs/api.rst` and `examples/05_machine_learning/`
- [ ] T092 Run linting with `make lint` for `nimare/ml.py`, `nimare/tests/test_ml.py`, `docs/api.rst`, and `examples/05_machine_learning/`
- [ ] T093 Review public API compatibility against baseline `0.16.0` for `nimare/__init__.py`, `nimare/ml.py`, and `docs/api.rst`
- [ ] T094 Review generated artifacts for consistency with `specs/001-ma-feature-dataset/spec.md`, `specs/001-ma-feature-dataset/contracts/public-api.md`, and `specs/001-ma-feature-dataset/contracts/sklearn-compatibility.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies; can start immediately.
- **Garden-Path Foundation (Phase 2)**: Depends on Setup completion.
- **Garden-Path End-to-End Pipeline (Phase 3)**: Depends on Phase 2 and intentionally implements thin slices of US1-US5 before comprehensive edge handling.
- **Basic Edge Conditions (Phase 4)**: Depends on a working garden path.
- **User Story Completion (Phases 5-9)**: Depends on Phase 4; can proceed by priority or by available ownership.
- **Performance, Documentation, and Compatibility (Phase 10)**: Depends on selected user-story completion.

### User Story Dependencies

- **US1 Convert Studysets into feature data**: Starts in Phase 3; completed in Phase 5.
- **US2 Split data without study leakage**: Starts in Phase 3; completed in Phase 6.
- **US3 Combine activation maps with study information**: Starts in Phase 3 for numeric descriptors; completed in Phase 7.
- **US4 Predict selected study information**: Starts in Phase 3 for scalar targets; completed in Phase 8.
- **US5 Reduce voxelwise feature dimensionality**: Starts in Phase 3 for truncated SVD; completed in Phase 9.

### Implementation Rule

- Keep early tests thin and workflow-oriented.
- Do not add broad edge-condition tests before the garden path passes.
- Add focused edge tests immediately before implementing each edge behavior.
- Prefer changing task details as design decisions evolve over preserving speculative early tests.

---

## Parallel Opportunities

- T002, T003, T004, and T006 can run in parallel after T001 is understood.
- T015 can run in parallel with T010-T014 because it touches the example file.
- T057, T066, T075, and T084 can run in parallel with their corresponding focused test tasks if example edits are coordinated.
- T086, T087, and T088 can run in parallel near the end because they touch different documentation surfaces.

## Garden-Path Parallel Example

```text
Task: "Add garden-path sklearn Bunch and estimator-fit test in nimare/tests/test_ml.py"
Task: "Draft happy-path dataset example in examples/05_machine_learning/01_plot_ma_feature_dataset.py"
```

## Edge-Condition Parallel Example

```text
Task: "Add focused missing-coordinate tests in nimare/tests/test_ml.py"
Task: "Implement missing-coordinate handling in nimare/ml.py after the tests are written"
```

---

## Implementation Strategy

### Garden Path First

1. Complete Phase 1 and Phase 2.
2. Complete Phase 3 until the perfect-input pipeline works.
3. Validate with `python -m pytest nimare/tests/test_ml.py -k "garden_path"`.
4. Review implementation shape before adding broad edge coverage.

### Basic Edges Second

1. Complete Phase 4 focused edge tests and implementation.
2. Keep diagnostics useful but avoid comprehensive behavior matrices until the public shape stabilizes.

### User Stories Third

1. Finish US1 map-only conversion.
2. Finish US2 manual splitting.
3. Finish US3 descriptor transformers and diagnostics.
4. Finish US4 target transformers and diagnostics.
5. Finish US5 remaining reducers.
6. Run Phase 10 verification.

## Notes

- Preserve released public behavior from tag `0.16.0`.
- Prefer existing NiMARE utilities first, then nilearn, then scikit-learn before adding helpers.
- Keep unreduced voxelwise masked activation matrices sparse; dense output is allowed only after explicit reduction.
- Treat Studyset as the only supported input type for this MVP.
