# Tasks: Masked Activation Feature Dataset

**Input**: Design documents from `specs/001-ma-feature-dataset/`
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Required by the NiMARE constitution and this feature plan. Test tasks
must be completed before the implementation tasks in each user story.

**Organization**: Tasks are grouped by user story so each story can be
implemented and tested independently after shared setup and foundational work.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel because it touches different files and has no dependency on incomplete tasks in the same phase.
- **[Story]**: User story label from `specs/001-ma-feature-dataset/spec.md`.
- Every checklist task includes exact repository-relative file paths.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create shared files and public API placeholders needed before TDD begins.

- [ ] T001 Create additive `nimare.ml` module scaffold with `__all__`, imports, and `NotImplementedError` placeholders for `MAFeatureDataset`, `MAFeatureExtractor`, and `make_map_reducer` in `nimare/ml.py`
- [ ] T002 [P] Create the machine-learning test module with imports, pytest markers, sparse-matrix assertions, and placeholder fixture names in `nimare/tests/test_ml.py`
- [ ] T003 [P] Create the Sphinx-Gallery dataset workflow example scaffold with public imports and placeholder narrative in `examples/05_machine_learning/01_plot_ma_feature_dataset.py`
- [ ] T004 [P] Create the Sphinx-Gallery reduction workflow example scaffold with public imports and placeholder narrative in `examples/05_machine_learning/02_plot_ma_feature_reduction.py`
- [ ] T005 Add the `nimare.ml` autosummary target placeholder and module description in `docs/api.rst`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish fixtures, Studyset identity assumptions, sparse assertions, and API wiring used by every user story.

**CRITICAL**: No user story implementation should begin until this phase is complete.

- [ ] T006 Add shared Studyset fixture builders with valid coordinates, a masker, metadata, annotations, texts, unique study IDs, and unique analysis IDs in `nimare/tests/test_ml.py`
- [ ] T007 Add shared Studyset fixture builders for coordinate-less analyses, missing descriptor fields, missing targets, constant targets, and too-few-study-group cases in `nimare/tests/test_ml.py`
- [ ] T008 Add shared test helpers that assert row alignment across `features`, `ids`, `study_ids`, `target`, and `provenance` in `nimare/tests/test_ml.py`
- [ ] T009 Add shared test helpers that assert unreduced voxelwise matrices remain scipy sparse and are never silently densified in `nimare/tests/test_ml.py`
- [ ] T010 Add shared test helpers that inspect `MAFeatureDataset.to_sklearn()` Bunch exports for `data`, `target`, `groups`, and `feature_names` in `nimare/tests/test_ml.py`
- [ ] T011 Export the additive `ml` module from `nimare/__init__.py` without changing existing released imports in `nimare/__init__.py`
- [ ] T012 Confirm the compatibility baseline remains `0.16.0` and record any discrepancy in `specs/001-ma-feature-dataset/plan.md`
- [ ] T013 Add module-level Numpydoc stubs documenting the analysis-row, voxel-column layout and unique-ID MVP input contract in `nimare/ml.py`
- [ ] T014 Add internal helper placeholders for Studyset field selection, sparse hstack, grouped slicing, and reducer validation in `nimare/ml.py`

**Checkpoint**: Foundation ready; user story tests and implementation can now begin.

---

## Phase 3: User Story 1 - Convert Studysets Into Feature Data (Priority: P1) MVP

**Goal**: Convert a NiMARE Studyset into an `MAFeatureDataset` with one row per retained analysis, sparse voxelwise map features, private masker storage, public IDs/groups/features/provenance, and sklearn-ready map-only export.

**Independent Test**: Given a Studyset with multiple studies and analyses, conversion produces one row per retained analysis, aligned sparse map features, unique full analysis IDs, Studyset-provided study IDs, `missing_coordinates` provenance, and sklearn Bunch export.

### Tests and Examples for User Story 1

> Write these tests first and confirm they fail before implementation.

- [ ] T015 [US1] Add failing tests that `MAFeatureExtractor.transform(studyset)` returns `(train_dataset, test_dataset)` with `test_dataset is None` when `test_size=None` in `nimare/tests/test_ml.py`
- [ ] T016 [US1] Add failing tests that `MAFeatureExtractor` exposes no public `fit` or `fit_transform` attributes in `nimare/tests/test_ml.py`
- [ ] T017 [US1] Add failing tests that map-only conversion creates an `MAFeatureDataset` with public `ids`, `study_ids`, `features`, `feature_names`, `target`, and `provenance` attributes in `nimare/tests/test_ml.py`
- [ ] T018 [US1] Add failing tests that internal `_map_features` is sparse, has shape `n_retained_analyses x n_masked_voxels`, and backs public `features` when no descriptors are extracted in `nimare/tests/test_ml.py`
- [ ] T019 [US1] Add failing tests that `missing_coordinates="drop"` is the constructor default, drops coordinate-less analyses before row construction, and records `dropped_ids` in provenance in `nimare/tests/test_ml.py`
- [ ] T020 [US1] Add failing tests that `missing_coordinates="include"` retains coordinate-less analyses as all-zero sparse map rows in `nimare/tests/test_ml.py`
- [ ] T021 [US1] Add failing tests that conversion uses Studyset-provided `ids` and `study_ids` without duplicate-ID reconciliation or analysis-ID fallback logic in `nimare/tests/test_ml.py`
- [ ] T022 [US1] Add failing tests that `MAFeatureExtractor.to_sklearn(studyset)` returns `(train_bunch, test_bunch)` with `test_bunch is None` when no split is requested in `nimare/tests/test_ml.py`
- [ ] T023 [US1] Add failing tests that map-only Bunch export contains sparse `data`, `target is None`, `groups == study_ids`, and separate `feature_names` in `nimare/tests/test_ml.py`
- [ ] T024 [US1] Add failing compatibility tests that conversion relies on existing Studyset and kernel transformer public behavior without changing latest-release APIs in `nimare/tests/test_ml.py`
- [ ] T025 [P] [US1] Fill the conversion example with `MAFeatureExtractor.to_sklearn(studyset)`, `missing_coordinates="drop"`, sparse `train_bunch.data`, `groups`, and provenance inspection in `examples/05_machine_learning/01_plot_ma_feature_dataset.py`

### Implementation for User Story 1

- [ ] T026 [US1] Implement `MAFeatureDataset.__init__` with public `ids`, `study_ids`, `features`, `feature_names`, `target`, and `provenance` attributes in `nimare/ml.py`
- [ ] T027 [US1] Implement `MAFeatureDataset.__init__` private `_map_features`, `_descriptor_features`, and `_masker` storage without exposing them as required public fields in `nimare/ml.py`
- [ ] T028 [US1] Implement `MAFeatureDataset` validation for feature row counts, ID/group lengths, sparse unreduced voxelwise features, target length, and provenance shape in `nimare/ml.py`
- [ ] T029 [US1] Implement `MAFeatureDataset.copy()` for map-only datasets with independent sparse matrix and provenance copies in `nimare/ml.py`
- [ ] T030 [US1] Implement `MAFeatureDataset.to_sklearn()` returning a `sklearn.utils.Bunch` with `data`, `target`, `groups`, and `feature_names` while preserving sparse unreduced data in `nimare/ml.py`
- [ ] T031 [US1] Implement `MAFeatureExtractor.__init__` storage for `kernel_transformer`, field selectors, `missing_coordinates="drop"`, `test_size`, `random_state`, `cache_maps`, `memory`, and `memory_level` in `nimare/ml.py`
- [ ] T032 [US1] Implement Studyset-native access to `ids`, `study_ids`, `coordinates`, `metadata`, `annotations_df`, `texts`, `masker`, `space`, and `basepath` in `nimare/ml.py`
- [ ] T033 [US1] Implement sparse MA feature extraction with `KernelTransformer.transform(..., return_type="sparse")` and row alignment to retained Studyset `ids` in `nimare/ml.py`
- [ ] T034 [US1] Implement `missing_coordinates` handling for default `drop`, explicit `include`, invalid option errors, and `dropped_ids` provenance in `nimare/ml.py`
- [ ] T035 [US1] Implement `MAFeatureExtractor.transform(studyset)` as cached extraction plus optional split returning `(train_dataset, test_dataset)` in `nimare/ml.py`
- [ ] T036 [US1] Implement `MAFeatureExtractor.to_sklearn(studyset, map_reducer=None, map_reducer_params=None)` as the one-call sklearn export wrapper in `nimare/ml.py`
- [ ] T037 [US1] Implement extractor-level map cache keys that invalidate on Studyset identity/content, kernel configuration, mask space/order, and missing-coordinate policy in `nimare/ml.py`
- [ ] T038 [US1] Add explicit `ValueError` messages for missing masker, missing Studyset IDs, missing study groups, invalid map rows, and incompatible sparse map shape in `nimare/ml.py`
- [ ] T039 [US1] Add Numpydoc docstrings for `MAFeatureDataset`, `MAFeatureExtractor`, `MAFeatureDataset.to_sklearn`, `MAFeatureDataset.copy`, `MAFeatureExtractor.transform`, and `MAFeatureExtractor.to_sklearn` in `nimare/ml.py`
- [ ] T040 [US1] Finalize initial `nimare.ml` autosummary and API documentation for conversion and map-only export in `docs/api.rst`

**Checkpoint**: User Story 1 is independently testable with `python -m pytest nimare/tests/test_ml.py -k "conversion or map_only or missing_coordinates"`.

---

## Phase 4: User Story 2 - Split Data Without Study Leakage (Priority: P1)

**Goal**: Provide reproducible grouped train/test splits that keep all analyses from each study in one partition and return sliced `MAFeatureDataset` objects.

**Independent Test**: Given a converted dataset with repeated study IDs, repeated splits with the same settings are identical, no study appears in both partitions, and each split exports `groups` equal to `study_ids`.

### Tests and Examples for User Story 2

> Write these tests first and confirm they fail before implementation.

- [ ] T041 [US2] Add failing tests that `MAFeatureDataset.split(test_size, random_state)` uses `study_ids` as sklearn `groups` and no study ID appears in both outputs in `nimare/tests/test_ml.py`
- [ ] T042 [US2] Add failing tests that identical random state and split settings produce identical train/test `ids` in `nimare/tests/test_ml.py`
- [ ] T043 [US2] Add failing tests that train/test slices preserve row alignment for `features`, `ids`, `study_ids`, `target`, and `provenance` in `nimare/tests/test_ml.py`
- [ ] T044 [US2] Add failing tests that each split's `to_sklearn()` export has `groups` equal to `study_ids` and sparse `data` aligned to retained rows in `nimare/tests/test_ml.py`
- [ ] T045 [US2] Add failing tests that `MAFeatureExtractor.transform(studyset)` performs grouped splitting when `test_size` is set and returns `(train_dataset, test_dataset)` in `nimare/tests/test_ml.py`
- [ ] T046 [US2] Add failing tests that too-few-study cases fail with a clear `ValueError` before returning partial splits in `nimare/tests/test_ml.py`
- [ ] T047 [P] [US2] Extend the dataset workflow example with `test_size`, grouped train/test outputs, and `groups` versus `study_ids` consistency in `examples/05_machine_learning/01_plot_ma_feature_dataset.py`

### Implementation for User Story 2

- [ ] T048 [US2] Implement row slicing internals for `MAFeatureDataset` that slice sparse `_map_features`, `_descriptor_features`, public `features`, `target`, `ids`, `study_ids`, and `provenance` consistently in `nimare/ml.py`
- [ ] T049 [US2] Implement `MAFeatureDataset.split` using scikit-learn `GroupShuffleSplit` with `study_ids` as groups and train/test `MAFeatureDataset` return values in `nimare/ml.py`
- [ ] T050 [US2] Implement grouped CV support for the `cv` parameter using scikit-learn group splitters while preserving study-group disjointness and row alignment in `nimare/ml.py`
- [ ] T051 [US2] Integrate `test_size` and `random_state` from `MAFeatureExtractor` into `transform(studyset)` and `to_sklearn(studyset, ...)` in `nimare/ml.py`
- [ ] T052 [US2] Implement split diagnostics and `ValueError` messages for missing groups, too few groups, invalid `test_size`, and unsupported `cv` configurations in `nimare/ml.py`
- [ ] T053 [US2] Add split-specific Numpydoc examples and API documentation for `groups`, `study_ids`, and tuple returns in `nimare/ml.py` and `docs/api.rst`

**Checkpoint**: User Stories 1 and 2 are independently testable with `python -m pytest nimare/tests/test_ml.py -k "split or leakage or groups"`.

---

## Phase 5: User Story 3 - Combine Activation Maps With Study Information (Priority: P2)

**Goal**: Add selected numeric metadata and annotation fields as descriptor columns, reject non-numeric descriptors by default, and support explicit descriptor transformers or vectorizers fit on training rows only when a split is requested.

**Independent Test**: Given selected descriptor fields, numeric descriptors are appended as numeric columns aligned to analysis rows; non-numeric fields fail without explicit preprocessing; transformed descriptor columns remain separate from internal `_map_features`.

### Tests and Examples for User Story 3

> Write these tests first and confirm they fail before implementation.

- [ ] T054 [US3] Add failing tests for descriptor selector normalization for metadata, annotations, and texts using dictionary selectors in `nimare/tests/test_ml.py`
- [ ] T055 [US3] Add failing tests that numeric metadata and annotation descriptors align to `ids`, produce descriptor feature names, and append to `features` and `to_sklearn().data` in `nimare/tests/test_ml.py`
- [ ] T056 [US3] Add failing tests that text and categorical descriptor fields raise without explicit transformers or vectorizers in `nimare/tests/test_ml.py`
- [ ] T057 [US3] Add failing tests that explicit descriptor vectorizers produce numeric descriptor columns and feature names in `nimare/tests/test_ml.py`
- [ ] T058 [US3] Add failing tests that descriptor transformers fit only on training rows and transform held-out rows when `test_size` is set in `nimare/tests/test_ml.py`
- [ ] T059 [US3] Add failing tests for missing descriptor reports listing affected `ids` and field selectors without silent row dropping or filling in `nimare/tests/test_ml.py`
- [ ] T060 [US3] Add failing tests that a field selected as target is not silently reused as a descriptor feature by default in `nimare/tests/test_ml.py`
- [ ] T061 [P] [US3] Extend the dataset workflow example with numeric descriptors and explicit text vectorization for descriptor fields in `examples/05_machine_learning/01_plot_ma_feature_dataset.py`

### Implementation for User Story 3

- [ ] T062 [US3] Implement descriptor field selector normalization and validation for `metadata`, `annotations`, and `texts` sources in `nimare/ml.py`
- [ ] T063 [US3] Implement Studyset table lookup for descriptor fields with row alignment to retained `ids` in `nimare/ml.py`
- [ ] T064 [US3] Implement numeric descriptor extraction, `_descriptor_features` construction, descriptor feature names, and sparse-compatible hstack into public `features` in `nimare/ml.py`
- [ ] T065 [US3] Implement non-numeric descriptor rejection with explicit messages that identify source and field in `nimare/ml.py`
- [ ] T066 [US3] Implement explicit descriptor transformer/vectorizer execution and transformed feature naming in `nimare/ml.py`
- [ ] T067 [US3] Implement train-only descriptor transformer fitting when `test_size` is set and held-out descriptor transformation without leakage in `nimare/ml.py`
- [ ] T068 [US3] Implement missing descriptor diagnostics in `provenance` without silent row dropping or filling in `nimare/ml.py`
- [ ] T069 [US3] Implement guards that prevent target fields from being duplicated as descriptors by default in `nimare/ml.py`
- [ ] T070 [US3] Add descriptor-feature Numpydoc docs and API documentation for selectors, missing diagnostics, explicit transformers, and sparse export in `nimare/ml.py` and `docs/api.rst`

**Checkpoint**: User Story 3 is independently testable with `python -m pytest nimare/tests/test_ml.py -k "descriptor or non_numeric or vectorizer"`.

---

## Phase 6: User Story 4 - Predict Selected Study Information (Priority: P2)

**Goal**: Export one selected scalar numeric or categorical metadata/annotation value, or an explicitly transformed text-derived value, as the prediction target `y`.

**Independent Test**: Given a selected outcome field, target values align to feature rows; scalar numeric and categorical targets export correctly; missing, constant, raw free-text, and multi-label targets are diagnosed or rejected as specified.

### Tests and Examples for User Story 4

> Write these tests first and confirm they fail before implementation.

- [ ] T071 [US4] Add failing tests for scalar numeric target extraction, one-dimensional `target`, target diagnostics in `provenance`, and row alignment with `features` in `nimare/tests/test_ml.py`
- [ ] T072 [US4] Add failing tests for scalar categorical targets, including string labels accepted by downstream sklearn classifiers in `nimare/tests/test_ml.py`
- [ ] T073 [US4] Add failing tests that study-level targets may repeat across analyses and grouped splits keep repeated study-level targets in one partition in `nimare/tests/test_ml.py`
- [ ] T074 [US4] Add failing tests for missing target reports and constant target diagnostics in `nimare/tests/test_ml.py`
- [ ] T075 [US4] Add failing tests that raw free-text and multi-label targets raise unless an explicit target transformer or label extractor is supplied in `nimare/tests/test_ml.py`
- [ ] T076 [US4] Add failing tests that explicit target transformers or label extractors produce a one-dimensional target aligned to retained `ids` in `nimare/tests/test_ml.py`
- [ ] T077 [US4] Add failing tests that target values survive `split`, dataset slicing, `to_sklearn()`, and reduced dataset copies without row-order changes in `nimare/tests/test_ml.py`
- [ ] T078 [P] [US4] Extend the dataset workflow example with target extraction, exported `y`, and a minimal sklearn estimator fit on train data in `examples/05_machine_learning/01_plot_ma_feature_dataset.py`

### Implementation for User Story 4

- [ ] T079 [US4] Implement target field selector normalization and lookup for metadata, annotations, and texts in `nimare/ml.py`
- [ ] T080 [US4] Implement scalar numeric and scalar categorical target validation and one-dimensional target storage in `MAFeatureDataset.target` in `nimare/ml.py`
- [ ] T081 [US4] Implement missing target reports and constant target diagnostics in `provenance` in `nimare/ml.py`
- [ ] T082 [US4] Implement raw free-text and multi-label target rejection with explicit messages in `nimare/ml.py`
- [ ] T083 [US4] Implement explicit target transformer or label extractor support with retained-row alignment in `nimare/ml.py`
- [ ] T084 [US4] Preserve target alignment through `to_sklearn`, `split`, row slicing, and reduced dataset copies in `nimare/ml.py`
- [ ] T085 [US4] Add target-extraction Numpydoc docs and API documentation for target selectors, target diagnostics, and sklearn `target` export in `nimare/ml.py` and `docs/api.rst`

**Checkpoint**: User Story 4 is independently testable with `python -m pytest nimare/tests/test_ml.py -k "target or outcome or y"`.

---

## Phase 7: User Story 5 - Reduce Voxelwise Feature Dimensionality (Priority: P3)

**Goal**: Provide sparse-safe reduction workflows that operate only on internal `_map_features`, use the stored `_masker` for atlas/label aggregation, preserve descriptors and targets, and optionally run through `MAFeatureExtractor.to_sklearn(studyset, map_reducer=...)`.

**Independent Test**: Given a feature dataset, variance thresholding, sparse-compatible low-rank reduction, and atlas/label aggregation reduce map features correctly without densifying unreduced voxel matrices; reducers fit only on training data and transform held-out analyses without leakage.

### Tests and Examples for User Story 5

> Write these tests first and confirm they fail before implementation.

- [ ] T086 [US5] Add failing reducer factory tests for `variance_threshold`, `truncated_svd`, and `atlas_aggregation` method names and returned sklearn-compatible transformers in `nimare/tests/test_ml.py`
- [ ] T087 [US5] Add failing tests that reducers operate on internal `_map_features` only and never select descriptor columns from public `features` or exported sklearn `data` in `nimare/tests/test_ml.py`
- [ ] T088 [US5] Add failing tests that `MAFeatureDataset.apply_map_reducer(fit=True)` fits on training map features and `fit=False` transforms held-out map features with the same fitted reducer in `nimare/tests/test_ml.py`
- [ ] T089 [US5] Add failing tests that `MAFeatureExtractor.to_sklearn(studyset, map_reducer=...)` fits the reducer on train rows only and applies it to held-out rows when `test_size` is set in `nimare/tests/test_ml.py`
- [ ] T090 [US5] Add failing tests that sparse low-rank reduction accepts sparse input, does not densify the unreduced voxel matrix, and may return a reduced dense component matrix in `nimare/tests/test_ml.py`
- [ ] T091 [US5] Add failing atlas/label aggregation tests that use the stored dataset `_masker` to align atlas labels to voxel columns and produce analysis-by-parcel outputs in `nimare/tests/test_ml.py`
- [ ] T092 [US5] Add failing atlas/label aggregation tests for incompatible atlas space, incompatible label image shape, empty parcels, and missing masker errors in `nimare/tests/test_ml.py`
- [ ] T093 [US5] Add failing tests that reduced datasets preserve `ids`, `study_ids`, `target`, descriptor features, and `provenance` in `nimare/tests/test_ml.py`
- [ ] T094 [US5] Add failing tests that reduced `feature_names` distinguish variance-threshold, SVD component, and atlas parcel outputs in `nimare/tests/test_ml.py`
- [ ] T095 [US5] Add a `performance_smoke` test for 1,000-study sparse conversion, grouped split, and sparse-safe reduction within <=3 minutes and <=5 GB peak memory in `nimare/tests/test_ml.py`
- [ ] T096 [P] [US5] Fill the reduction workflow example with grouped split, train-fit truncated SVD, held-out transform, and atlas aggregation using the dataset masker in `examples/05_machine_learning/02_plot_ma_feature_reduction.py`

### Implementation for User Story 5

- [ ] T097 [US5] Implement `make_map_reducer("variance_threshold", **kwargs)` using sparse-compatible variance filtering in `nimare/ml.py`
- [ ] T098 [US5] Implement `make_map_reducer("truncated_svd", **kwargs)` using scikit-learn `TruncatedSVD` or an equivalent sparse-safe low-rank reducer in `nimare/ml.py`
- [ ] T099 [US5] Implement `make_map_reducer("atlas_aggregation", atlas=..., masker=..., **kwargs)` contract and validation in `nimare/ml.py`
- [ ] T100 [US5] Implement atlas-to-mask alignment using the stored dataset `_masker`, voxel ordering, and a sparse voxel-by-parcel aggregation matrix in `nimare/ml.py`
- [ ] T101 [US5] Implement `MAFeatureDataset.apply_map_reducer` so reducers transform only `_map_features` and descriptor features remain separate in `nimare/ml.py`
- [ ] T102 [US5] Implement `apply_map_reducer` fit semantics, held-out transform semantics, row-count validation, output sparse/dense handling, and reduced feature names in `nimare/ml.py`
- [ ] T103 [US5] Integrate `map_reducer` and `map_reducer_params` into `MAFeatureExtractor.to_sklearn(studyset, ...)` with train-only fitting when split in `nimare/ml.py`
- [ ] T104 [US5] Implement diagnostics for unintended densification of unreduced voxelwise data and performance-budget risk in `nimare/ml.py`
- [ ] T105 [US5] Add reducer Numpydoc docs and API documentation for sparse-only unreduced data, reduced dense outputs, and masker-based atlas aggregation in `nimare/ml.py` and `docs/api.rst`

**Checkpoint**: User Story 5 is independently testable with `python -m pytest nimare/tests/test_ml.py -k "reducer or reduction or atlas or performance_smoke"`.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Final verification, documentation consistency, and compatibility review across all selected user stories.

- [ ] T106 [P] Finalize Numpydoc parameter docs, return docs, examples, warnings, and errors for every public class and function in `nimare/ml.py`
- [ ] T107 [P] Verify both Sphinx-Gallery examples use only public APIs and match `specs/001-ma-feature-dataset/quickstart.md` in `examples/05_machine_learning/01_plot_ma_feature_dataset.py` and `examples/05_machine_learning/02_plot_ma_feature_reduction.py`
- [ ] T108 [P] Update API documentation text for `nimare.ml`, `MAFeatureDataset`, `MAFeatureExtractor`, and `make_map_reducer` in `docs/api.rst`
- [ ] T109 Run the targeted feature test suite with `python -m pytest nimare/tests/test_ml.py` for `nimare/tests/test_ml.py`
- [ ] T110 Run the performance smoke check with `python -m pytest -m performance_smoke nimare/tests/test_ml.py` for `nimare/tests/test_ml.py`
- [ ] T111 Run the documentation build with `make -C docs html` for `docs/api.rst` and `examples/05_machine_learning/`
- [ ] T112 Run linting with `make lint` for `nimare/ml.py`, `nimare/tests/test_ml.py`, `docs/api.rst`, and `examples/05_machine_learning/`
- [ ] T113 Review public API compatibility against baseline `0.16.0` for `nimare/__init__.py`, `nimare/ml.py`, and `docs/api.rst`
- [ ] T114 Review generated artifacts for consistency with `specs/001-ma-feature-dataset/spec.md`, `specs/001-ma-feature-dataset/contracts/public-api.md`, and `specs/001-ma-feature-dataset/contracts/sklearn-compatibility.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies; can start immediately.
- **Foundational (Phase 2)**: Depends on Setup completion; blocks all user stories.
- **User Story 1 and User Story 2 (P1)**: Depend on Foundational completion.
- **User Story 3 and User Story 4 (P2)**: Depend on Foundational completion and integrate with the dataset container from User Story 1.
- **User Story 5 (P3)**: Depends on Foundational completion, map features from User Story 1, split semantics from User Story 2, and preservation checks from User Stories 3 and 4 when descriptors or targets are present.
- **Polish (Phase 8)**: Depends on all desired user stories being complete.

### User Story Dependencies

- **US1 Convert Studysets into feature data**: MVP; no dependency on other user stories after Foundational.
- **US2 Split data without study leakage**: Requires `MAFeatureDataset` rows, IDs, study groups, and tuple-return extractor behavior from US1.
- **US3 Combine activation maps with study information**: Requires `MAFeatureDataset` and dataset-level sklearn export from US1.
- **US4 Predict selected study information**: Requires `MAFeatureDataset` and dataset-level sklearn export from US1; split alignment integrates with US2.
- **US5 Reduce voxelwise feature dimensionality**: Requires internal `_map_features` and `_masker` from US1; train-fit/held-out-transform workflows integrate with US2; descriptor/target preservation integrates with US3 and US4 when present.

### Within Each User Story

- Tests must be written and fail before implementation.
- Public examples must be created or edited before the story is considered complete.
- Core implementation in `nimare/ml.py` follows the story tests.
- Documentation and docstrings close each story.

---

## Parallel Opportunities

- T002, T003, T004, and T005 can run in parallel after T001 is understood because they touch different files.
- T006, T007, T008, T009, and T010 can be drafted in parallel within `nimare/tests/test_ml.py` if coordinated to avoid duplicate fixture names.
- T025 can run in parallel with US1 test authoring because it touches `examples/05_machine_learning/01_plot_ma_feature_dataset.py`.
- T047, T061, and T078 can run in parallel with their story test tasks if prior edits to `examples/05_machine_learning/01_plot_ma_feature_dataset.py` are merged.
- T096 can run in parallel with US5 test authoring because it touches `examples/05_machine_learning/02_plot_ma_feature_reduction.py`.
- T106, T107, and T108 can run in parallel near the end because they touch `nimare/ml.py`, example files, and `docs/api.rst`.

## Parallel Example: User Story 1

```text
Task: "Add failing sparse conversion and missing-coordinate tests in nimare/tests/test_ml.py"
Task: "Fill conversion workflow example in examples/05_machine_learning/01_plot_ma_feature_dataset.py"
```

## Parallel Example: User Story 3

```text
Task: "Add failing descriptor diagnostics tests in nimare/tests/test_ml.py"
Task: "Extend descriptor feature examples in examples/05_machine_learning/01_plot_ma_feature_dataset.py"
```

## Parallel Example: User Story 5

```text
Task: "Add failing map reducer and atlas aggregation tests in nimare/tests/test_ml.py"
Task: "Fill reduction workflow example in examples/05_machine_learning/02_plot_ma_feature_reduction.py"
```

---

## Implementation Strategy

### MVP First

1. Complete Phase 1 and Phase 2.
2. Complete Phase 3 for User Story 1.
3. Run `python -m pytest nimare/tests/test_ml.py -k "conversion or map_only or missing_coordinates"`.
4. Stop and review the MVP before adding split, descriptor, target, or reduction behavior.

### Incremental Delivery

1. Add US1 conversion and map-only export.
2. Add US2 grouped splitting.
3. Add US3 descriptor features.
4. Add US4 target extraction.
5. Add US5 reduction workflows and performance smoke coverage.
6. Run Phase 8 verification.

### Parallel Team Strategy

After Phase 2, one developer can work on US1/US2 core dataset behavior while
another prepares examples and tests for US3/US4/US5. Production edits to
`nimare/ml.py` should be serialized or coordinated because all stories touch
that public module.

## Notes

- Preserve released public behavior from tag `0.16.0`.
- Prefer existing NiMARE utilities first, then nilearn, then scikit-learn before adding helpers.
- Keep unreduced voxelwise masked activation matrices sparse; dense output is allowed only after explicit reduction.
- Treat Studyset as the only supported input type for this MVP.
