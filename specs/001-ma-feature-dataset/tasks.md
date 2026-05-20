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

- [ ] T001 Create additive module scaffold with `__all__`, imports, and `NotImplementedError` placeholders for `MAFeatureDataset`, `MAFeatureExtractor`, and `make_map_reducer` in `nimare/ml.py`
- [ ] T002 [P] Create the machine-learning test module with imports, pytest markers, sparse-matrix assertions, and placeholder fixture names in `nimare/tests/test_ml.py`
- [ ] T003 [P] Create the Sphinx-Gallery dataset workflow example scaffold with public imports and placeholder narrative in `examples/05_machine_learning/01_plot_ma_feature_dataset.py`
- [ ] T004 [P] Create the Sphinx-Gallery reduction workflow example scaffold with public imports and placeholder narrative in `examples/05_machine_learning/02_plot_ma_feature_reduction.py`
- [ ] T005 Add the `nimare.ml` autosummary target placeholder and module description in `docs/api.rst`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish fixtures, identity assumptions, and API wiring used by every user story.

**CRITICAL**: No user story implementation should begin until this phase is complete.

- [ ] T006 Add shared Studyset/Dataset fixture builders with valid coordinates, a masker, metadata, annotations, texts, unique study IDs, and unique analysis IDs in `nimare/tests/test_ml.py`
- [ ] T007 Add shared fixture helpers for missing maps, missing descriptor fields, missing targets, and too-few-study-group cases in `nimare/tests/test_ml.py`
- [ ] T008 Add shared test helpers that assert row alignment across `map_features`, `sample_ids`, `study_ids`, `sample_metadata`, `target`, and descriptor matrices in `nimare/tests/test_ml.py`
- [ ] T009 Add shared test helpers that assert unreduced voxelwise matrices remain scipy sparse and are never silently densified in `nimare/tests/test_ml.py`
- [ ] T010 Export the additive `ml` module from `nimare/__init__.py` without changing existing released imports in `nimare/__init__.py`
- [ ] T011 Confirm the compatibility baseline remains `0.16.0` and record any discrepancy in `specs/001-ma-feature-dataset/plan.md`
- [ ] T012 Add module-level Numpydoc stubs documenting the analysis-row, voxel-column layout and unique-ID MVP input contract in `nimare/ml.py`

**Checkpoint**: Foundation ready; user story tests and implementation can now begin.

---

## Phase 3: User Story 1 - Convert Collections Into Feature Data (Priority: P1) MVP

**Goal**: Convert a NiMARE Studyset or Dataset into an `MAFeatureDataset` with one row per eligible analysis, sparse voxelwise map features, stored masker, sample provenance, and study groups.

**Independent Test**: Given a collection with multiple studies and analyses, conversion produces one row per eligible analysis, aligned sparse map features, unique sample IDs, collection-provided study IDs, masker provenance, and an exclusion report.

### Tests and Examples for User Story 1

> Write these tests first and confirm they fail before implementation.

- [ ] T013 [US1] Add failing tests that `MAFeatureExtractor.transform(collection)` exists, returns `MAFeatureDataset`, and `MAFeatureExtractor` has no public `fit` or `fit_transform` in `nimare/tests/test_ml.py`
- [ ] T014 [US1] Add failing tests that conversion produces `map_features` with shape `n_analyses x n_masked_voxels`, scipy sparse type, one `sample_id` per row, and the source masker in `nimare/tests/test_ml.py`
- [ ] T015 [US1] Add failing tests that `sample_metadata` contains row-aligned `study_id`, `analysis_id`, `sample_id`, source collection fields, and activation-map generation settings in `nimare/tests/test_ml.py`
- [ ] T016 [US1] Add failing tests that MVP inputs use collection-provided unique study IDs and analysis IDs for grouping without analysis-ID fallback logic in `nimare/tests/test_ml.py`
- [ ] T017 [US1] Add failing tests that `to_sklearn()` returns a Bunch with sparse `data`, `target=None`, `groups`, `sample_metadata`, `feature_names`, and `masker` metadata for map-only export in `nimare/tests/test_ml.py`
- [ ] T018 [US1] Add failing tests for missing-map exclusion reports that list excluded analysis IDs and reasons without changing row order for included analyses in `nimare/tests/test_ml.py`
- [ ] T019 [US1] Add failing compatibility tests that Dataset and Studyset conversion use existing collection/kernel behavior and preserve latest-release collection APIs in `nimare/tests/test_ml.py`
- [ ] T020 [P] [US1] Fill the conversion/provenance example showing `extractor.transform(collection)`, sparse `sklearn_data.data`, `groups`, and `sample_metadata` in `examples/05_machine_learning/01_plot_ma_feature_dataset.py`

### Implementation for User Story 1

- [ ] T021 [US1] Implement `MAFeatureDataset.__init__` with explicit fields for `map_features`, `sample_ids`, `study_ids`, `sample_metadata`, `masker`, `feature_names`, `exclusion_report`, and `provenance` in `nimare/ml.py`
- [ ] T022 [US1] Implement `MAFeatureDataset` dimension validation for row counts, sparse unreduced map features, sample ID alignment, study ID alignment, and sample metadata alignment in `nimare/ml.py`
- [ ] T023 [US1] Implement `MAFeatureDataset.copy()` and `MAFeatureDataset.get_feature_names()` for map-only datasets in `nimare/ml.py`
- [ ] T024 [US1] Implement `MAFeatureDataset.to_sklearn(include_descriptors=True, include_target=True)` returning a sklearn-compatible Bunch with sparse map data and no dense option in `nimare/ml.py`
- [ ] T025 [US1] Implement `MAFeatureExtractor.__init__` parameter storage for kernel transformer, field selectors, missing policy, memory, and memory level in `nimare/ml.py`
- [ ] T026 [US1] Implement `MAFeatureExtractor.transform(collection)` using `KernelTransformer.transform(..., return_type="sparse")` and row alignment to collection analysis IDs in `nimare/ml.py`
- [ ] T027 [US1] Implement Studyset-native and Dataset-compatible access to IDs, coordinates, masker, metadata, annotations, and texts without changing existing collection APIs in `nimare/ml.py`
- [ ] T028 [US1] Implement sample metadata construction, collection-provided study ID grouping, unique-ID MVP assumptions, and exclusion-report assembly in `nimare/ml.py`
- [ ] T029 [US1] Add conversion-specific `ValueError` messages for missing masker, missing study IDs, missing analysis IDs, incompatible map rows, and invalid sparse map shape in `nimare/ml.py`
- [ ] T030 [US1] Add Numpydoc docstrings for `MAFeatureDataset`, `MAFeatureExtractor`, `transform`, `to_sklearn`, `copy`, and `get_feature_names` in `nimare/ml.py`
- [ ] T031 [US1] Finalize initial `nimare.ml` autosummary and API documentation for conversion and map-only export in `docs/api.rst`

**Checkpoint**: User Story 1 is independently testable with `python -m pytest nimare/tests/test_ml.py -k "feature or conversion or provenance or map_only"`.

---

## Phase 4: User Story 2 - Split Data Without Study Leakage (Priority: P1)

**Goal**: Provide reproducible grouped train/test splits that keep all analyses from each study in the same partition and return sliced `MAFeatureDataset` objects.

**Independent Test**: Given a converted dataset with repeated study IDs, repeated splits with the same settings are identical, no study appears in both partitions, and each split can be exported through `to_sklearn()`.

### Tests and Examples for User Story 2

> Write these tests first and confirm they fail before implementation.

- [ ] T032 [US2] Add failing tests that `MAFeatureDataset.split(test_size, random_state)` uses study IDs as sklearn `groups` and no study ID appears in both outputs in `nimare/tests/test_ml.py`
- [ ] T033 [US2] Add failing tests that split reproducibility holds for identical random state and split settings in `nimare/tests/test_ml.py`
- [ ] T034 [US2] Add failing tests that train/test slices preserve row alignment for `map_features`, `sample_ids`, `study_ids`, `sample_metadata`, `target`, descriptors, and provenance in `nimare/tests/test_ml.py`
- [ ] T035 [US2] Add failing tests that each split's `to_sklearn()` exports `groups` equal to `sample_metadata["study_id"]` and preserves `analysis_id` provenance in `nimare/tests/test_ml.py`
- [ ] T036 [US2] Add failing tests that too-few-study cases fail with a clear `ValueError` before returning partial splits in `nimare/tests/test_ml.py`
- [ ] T037 [P] [US2] Extend the dataset workflow example with `feature_dataset.split(...)`, train/test `to_sklearn()`, and `groups` versus `sample_metadata["study_id"]` consistency in `examples/05_machine_learning/01_plot_ma_feature_dataset.py`

### Implementation for User Story 2

- [ ] T038 [US2] Implement row slicing internals for `MAFeatureDataset` that slice sparse map matrices, descriptors, target, study IDs, sample IDs, metadata, and provenance consistently in `nimare/ml.py`
- [ ] T039 [US2] Implement `MAFeatureDataset.split` using scikit-learn group splitters with `study_ids` as groups and returning train/test `MAFeatureDataset` slices in `nimare/ml.py`
- [ ] T040 [US2] Implement grouped CV support for the `cv` parameter while preserving study-group disjointness and row alignment in `nimare/ml.py`
- [ ] T041 [US2] Implement split diagnostics and `ValueError` messages for missing groups, too few groups, invalid `test_size`, and unsupported `cv` configurations in `nimare/ml.py`
- [ ] T042 [US2] Add split-specific Numpydoc examples and API documentation for `groups` and `sample_metadata["study_id"]` consistency in `nimare/ml.py` and `docs/api.rst`

**Checkpoint**: User Stories 1 and 2 are independently testable with `python -m pytest nimare/tests/test_ml.py -k "split or leakage or groups"`.

---

## Phase 5: User Story 3 - Combine Activation Maps With Study Information (Priority: P2)

**Goal**: Add selected numeric metadata and annotation fields as descriptor columns, reject non-numeric descriptors by default, and support explicit descriptor transformers or vectorizers.

**Independent Test**: Given selected descriptor fields, numeric descriptors are appended as numeric columns aligned to analysis rows; non-numeric fields fail without explicit preprocessing; transformed descriptor columns remain separate from `map_features` until sklearn export.

### Tests and Examples for User Story 3

> Write these tests first and confirm they fail before implementation.

- [ ] T043 [US3] Add failing tests for descriptor selector normalization for metadata, annotations, and texts using dictionary selectors in `nimare/tests/test_ml.py`
- [ ] T044 [US3] Add failing tests that numeric metadata and annotation descriptors align to `sample_ids`, produce descriptor feature names, and append to sparse `to_sklearn().data` in `nimare/tests/test_ml.py`
- [ ] T045 [US3] Add failing tests that `map_features` remains separate from descriptor features so map reducers never select descriptor columns in `nimare/tests/test_ml.py`
- [ ] T046 [US3] Add failing tests that a field selected as target is not silently reused as a descriptor feature by default in `nimare/tests/test_ml.py`
- [ ] T047 [US3] Add failing tests for missing descriptor reports listing affected `sample_id`s and fields under the selected missing policy in `nimare/tests/test_ml.py`
- [ ] T048 [US3] Add failing tests that categorical/text descriptor fields raise without explicit transformers and succeed with an explicit vectorizer producing numeric columns in `nimare/tests/test_ml.py`
- [ ] T049 [P] [US3] Extend the dataset workflow example with numeric descriptors and explicit text vectorization for descriptor fields in `examples/05_machine_learning/01_plot_ma_feature_dataset.py`

### Implementation for User Story 3

- [ ] T050 [US3] Implement descriptor field selector normalization and validation for metadata, annotations, and texts in `nimare/ml.py`
- [ ] T051 [US3] Implement collection table lookup for descriptor fields with row alignment to included `sample_ids` in `nimare/ml.py`
- [ ] T052 [US3] Implement numeric descriptor extraction, descriptor matrix construction, descriptor feature names, and sparse-compatible hstack behavior in `to_sklearn()` in `nimare/ml.py`
- [ ] T053 [US3] Implement non-numeric descriptor rejection, explicit descriptor transformer/vectorizer execution, and transformed feature naming in `nimare/ml.py`
- [ ] T054 [US3] Implement missing descriptor diagnostics and selected missing policy behavior without silent row dropping or filling in `nimare/ml.py`
- [ ] T055 [US3] Implement guards that prevent target fields from being duplicated as descriptors by default in `nimare/ml.py`
- [ ] T056 [US3] Add descriptor-feature Numpydoc docs and API documentation for selectors, missing policy, and explicit transformers in `nimare/ml.py` and `docs/api.rst`

**Checkpoint**: User Story 3 is independently testable with `python -m pytest nimare/tests/test_ml.py -k "descriptor or non_numeric or vectorizer"`.

---

## Phase 6: User Story 4 - Predict Selected Study Information (Priority: P2)

**Goal**: Export one selected scalar numeric or categorical metadata/annotation value, or an explicitly transformed text-derived value, as the prediction target `y`.

**Independent Test**: Given a selected outcome field, target values align to feature rows; scalar numeric and categorical targets export correctly; missing, constant, raw free-text, and multi-label targets are diagnosed or rejected as specified.

### Tests and Examples for User Story 4

> Write these tests first and confirm they fail before implementation.

- [ ] T057 [US4] Add failing tests for scalar numeric target extraction, one-dimensional `target`, target metadata, and row alignment with `data` in `nimare/tests/test_ml.py`
- [ ] T058 [US4] Add failing tests for scalar categorical targets, including string labels accepted by downstream sklearn classifiers in `nimare/tests/test_ml.py`
- [ ] T059 [US4] Add failing tests that study-level targets may repeat across analyses and grouped splits keep repeated study-level targets in one partition in `nimare/tests/test_ml.py`
- [ ] T060 [US4] Add failing tests for missing target reports, constant target diagnostics, and selected missing policy behavior in `nimare/tests/test_ml.py`
- [ ] T061 [US4] Add failing tests that raw free-text and multi-label targets raise unless an explicit target transformer or label extractor is supplied in `nimare/tests/test_ml.py`
- [ ] T062 [US4] Add failing tests that target values survive `split`, dataset slicing, and `to_sklearn(include_target=True)` without row-order changes in `nimare/tests/test_ml.py`
- [ ] T063 [P] [US4] Extend the dataset workflow example with target extraction, exported `y`, and a minimal sklearn estimator fit on train data in `examples/05_machine_learning/01_plot_ma_feature_dataset.py`

### Implementation for User Story 4

- [ ] T064 [US4] Implement target field selector normalization and lookup for metadata, annotations, and texts in `nimare/ml.py`
- [ ] T065 [US4] Implement scalar numeric and scalar categorical target validation, target type metadata, and one-dimensional target storage in `nimare/ml.py`
- [ ] T066 [US4] Implement missing target reports, constant target diagnostics, and selected missing policy behavior in `nimare/ml.py`
- [ ] T067 [US4] Implement raw free-text and multi-label target rejection plus explicit target transformer or label extractor support in `nimare/ml.py`
- [ ] T068 [US4] Preserve target alignment through `to_sklearn`, `split`, row slicing, and reduced dataset copies in `nimare/ml.py`
- [ ] T069 [US4] Add target-extraction Numpydoc docs and API documentation for target selectors and target diagnostics in `nimare/ml.py` and `docs/api.rst`

**Checkpoint**: User Story 4 is independently testable with `python -m pytest nimare/tests/test_ml.py -k "target or outcome or y"`.

---

## Phase 7: User Story 5 - Reduce Voxelwise Feature Dimensionality (Priority: P3)

**Goal**: Provide sparse-safe reduction workflows that operate only on `map_features`, use the stored masker for atlas/label aggregation, and preserve descriptors, targets, groups, and provenance.

**Independent Test**: Given a feature dataset, variance thresholding, sparse-compatible low-rank reduction, and atlas/label aggregation reduce map features correctly without densifying unreduced voxel matrices; reducers fit only on training data and transform held-out samples without leakage.

### Tests and Examples for User Story 5

> Write these tests first and confirm they fail before implementation.

- [ ] T070 [US5] Add failing reducer factory tests for `variance_threshold`, `truncated_svd`, and `atlas_aggregation` method names and returned sklearn-compatible transformers in `nimare/tests/test_ml.py`
- [ ] T071 [US5] Add failing tests that reducers operate on `MAFeatureDataset.map_features` only and never select descriptor columns from exported sklearn `data` in `nimare/tests/test_ml.py`
- [ ] T072 [US5] Add failing tests that `apply_map_reducer(fit=True)` fits on training map features and `fit=False` transforms held-out map features with the same fitted reducer in `nimare/tests/test_ml.py`
- [ ] T073 [US5] Add failing tests that sparse low-rank reduction uses sparse-compatible input, does not densify the unreduced voxel matrix, and may return a reduced dense component matrix in `nimare/tests/test_ml.py`
- [ ] T074 [US5] Add failing atlas/label aggregation tests that use the stored dataset masker to align atlas labels to voxel columns and produce analysis-by-parcel outputs in `nimare/tests/test_ml.py`
- [ ] T075 [US5] Add failing atlas/label aggregation tests for incompatible atlas space, incompatible label image shape, empty parcels, and missing masker errors in `nimare/tests/test_ml.py`
- [ ] T076 [US5] Add failing tests that reduced datasets preserve `sample_ids`, `study_ids`, `sample_metadata`, `target`, descriptor features, and provenance in `nimare/tests/test_ml.py`
- [ ] T077 [US5] Add failing tests that reduced feature names distinguish variance-threshold, SVD component, and atlas parcel outputs in `nimare/tests/test_ml.py`
- [ ] T078 [US5] Add a `performance_smoke` test for 1,000-study sparse conversion, grouped split, and sparse-safe reduction within <=3 minutes and <=5 GB peak memory in `nimare/tests/test_ml.py`
- [ ] T079 [P] [US5] Fill the reduction workflow example with grouped split, train-fit truncated SVD, held-out transform, and atlas aggregation using the dataset masker in `examples/05_machine_learning/02_plot_ma_feature_reduction.py`

### Implementation for User Story 5

- [ ] T080 [US5] Implement `make_map_reducer("variance_threshold", **kwargs)` using sparse-compatible variance filtering in `nimare/ml.py`
- [ ] T081 [US5] Implement `make_map_reducer("truncated_svd", **kwargs)` using scikit-learn `TruncatedSVD` or equivalent sparse-safe low-rank reducer in `nimare/ml.py`
- [ ] T082 [US5] Implement `make_map_reducer("atlas_aggregation", atlas=..., masker=..., **kwargs)` contract and validation in `nimare/ml.py`
- [ ] T083 [US5] Implement atlas-to-mask alignment using the stored dataset masker, voxel ordering, and a sparse voxel-by-parcel aggregation matrix in `nimare/ml.py`
- [ ] T084 [US5] Implement `MAFeatureDataset.apply_map_reducer` so it applies reducers only to `map_features` and preserves descriptor features separately in `nimare/ml.py`
- [ ] T085 [US5] Implement `apply_map_reducer` fit semantics, held-out transform semantics, row-count validation, output sparse/dense handling, and reduced feature names in `nimare/ml.py`
- [ ] T086 [US5] Implement diagnostics for unintended densification of unreduced voxelwise data and performance-budget risk in `nimare/ml.py`
- [ ] T087 [US5] Add reducer Numpydoc docs and API documentation for sparse-only unreduced data, reduced dense outputs, and masker-based atlas aggregation in `nimare/ml.py` and `docs/api.rst`

**Checkpoint**: User Story 5 is independently testable with `python -m pytest nimare/tests/test_ml.py -k "reducer or reduction or atlas or performance_smoke"`.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Final verification, documentation consistency, and compatibility review across all selected user stories.

- [ ] T088 [P] Finalize Numpydoc parameter docs, return docs, examples, warnings, and errors for every public class and function in `nimare/ml.py`
- [ ] T089 [P] Verify both Sphinx-Gallery examples use only public APIs and match `specs/001-ma-feature-dataset/quickstart.md` in `examples/05_machine_learning/01_plot_ma_feature_dataset.py` and `examples/05_machine_learning/02_plot_ma_feature_reduction.py`
- [ ] T090 [P] Update API documentation text for `nimare.ml`, `MAFeatureDataset`, `MAFeatureExtractor`, and `make_map_reducer` in `docs/api.rst`
- [ ] T091 Run the targeted feature test suite with `python -m pytest nimare/tests/test_ml.py` for `nimare/tests/test_ml.py`
- [ ] T092 Run the performance smoke check with `python -m pytest -m performance_smoke nimare/tests/test_ml.py` for `nimare/tests/test_ml.py`
- [ ] T093 Run the documentation build with `make -C docs html` for `docs/api.rst` and `examples/05_machine_learning/`
- [ ] T094 Run linting with `make lint` for `nimare/ml.py`, `nimare/tests/test_ml.py`, `docs/api.rst`, and `examples/05_machine_learning/`
- [ ] T095 Review public API compatibility against baseline `0.16.0` for `nimare/__init__.py`, `nimare/ml.py`, and `docs/api.rst`
- [ ] T096 Review generated artifacts for consistency with `specs/001-ma-feature-dataset/spec.md`, `specs/001-ma-feature-dataset/contracts/public-api.md`, and `specs/001-ma-feature-dataset/contracts/sklearn-compatibility.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies; can start immediately.
- **Foundational (Phase 2)**: Depends on Setup completion; blocks all user stories.
- **User Story 1 and User Story 2 (P1)**: Depend on Foundational completion.
- **User Story 3 and User Story 4 (P2)**: Depend on Foundational completion and integrate with the dataset container from User Story 1.
- **User Story 5 (P3)**: Depends on Foundational completion, map features from User Story 1, and split semantics from User Story 2.
- **Polish (Phase 8)**: Depends on all desired user stories being complete.

### User Story Dependencies

- **US1 Convert collections into feature data**: MVP; no dependency on other user stories after Foundational.
- **US2 Split data without study leakage**: Requires `MAFeatureDataset` rows, sample metadata, and study groups from US1.
- **US3 Combine activation maps with study information**: Requires `MAFeatureDataset` and `to_sklearn()` from US1.
- **US4 Predict selected study information**: Requires `MAFeatureDataset` and `to_sklearn()` from US1; split alignment integrates with US2.
- **US5 Reduce voxelwise feature dimensionality**: Requires map features and masker from US1; train-fit/held-out-transform workflows integrate with US2; descriptor/target preservation integrates with US3/US4 when present.

### Within Each User Story

- Tests must be written and fail before implementation.
- Public examples must be created or edited before the story is considered complete.
- Core implementation in `nimare/ml.py` follows the story tests.
- Documentation and docstrings close each story.

---

## Parallel Opportunities

- T002, T003, T004, and T005 can run in parallel after T001 is understood because they touch different files.
- T006, T007, T008, and T009 can be drafted in parallel within `nimare/tests/test_ml.py` if coordinated to avoid duplicate fixture names.
- T020 can run in parallel with US1 test authoring because it touches `examples/05_machine_learning/01_plot_ma_feature_dataset.py`.
- T037, T049, and T063 can run in parallel with their story test tasks if prior edits to `examples/05_machine_learning/01_plot_ma_feature_dataset.py` are merged.
- T079 can run in parallel with US5 test authoring because it touches `examples/05_machine_learning/02_plot_ma_feature_reduction.py`.
- T088, T089, and T090 can run in parallel near the end because they touch `nimare/ml.py`, example files, and `docs/api.rst`.

## Parallel Example: User Story 1

```text
Task: "Add failing sparse conversion and provenance tests in nimare/tests/test_ml.py"
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
3. Run `python -m pytest nimare/tests/test_ml.py -k "feature or conversion or provenance or map_only"`.
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
- Keep `map_features` separate from descriptor features so reducers only transform map-derived columns.
- Use the stored dataset masker for atlas/label aggregation and voxel ordering.
- Export sklearn `groups` from study IDs and also preserve `study_id` plus `analysis_id` in `sample_metadata`.
- Assume MVP inputs provide unique study IDs and unique analysis IDs.
- Reject non-numeric descriptor features by default without explicit preprocessing.
- Reject raw free-text and multi-label targets by default without explicit target handling.
- Create examples as `.py` Sphinx-Gallery sources; generated notebooks come from the docs build.
