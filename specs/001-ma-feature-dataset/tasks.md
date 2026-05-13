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

**Purpose**: Create the shared files needed for TDD, public API examples, and documentation.

- [ ] T001 Create the additive public module file `nimare/ml.py` with imports, `__all__`, and placeholders for `MAFeatureDataset`, `MAFeatureExtractor`, and `make_map_reducer`
- [ ] T002 [P] Create the machine-learning test module scaffold in `nimare/tests/test_ml.py`
- [ ] T003 [P] Create the Sphinx-Gallery dataset workflow example scaffold in `examples/05_machine_learning/01_plot_ma_feature_dataset.py`
- [ ] T004 [P] Create the Sphinx-Gallery reduction workflow example scaffold in `examples/05_machine_learning/02_plot_ma_feature_reduction.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish shared fixtures, public API wiring, and compatibility guardrails before user-story work begins.

**CRITICAL**: No user story implementation should begin until this phase is complete.

- [ ] T005 Add shared Studyset/Dataset fixture builders with coordinates, metadata, annotations, texts, explicit study IDs, and ambiguous-ID cases in `nimare/tests/test_ml.py`
- [ ] T006 Add Numpydoc-compatible public API skeletons that raise `NotImplementedError` for planned methods in `nimare/ml.py`
- [ ] T007 Export the additive `ml` module from `nimare/__init__.py`
- [ ] T008 Confirm the latest release compatibility baseline remains `0.16.0` in `specs/001-ma-feature-dataset/plan.md`

**Checkpoint**: Foundation ready; user story tests and implementation can now begin.

---

## Phase 3: User Story 1 - Convert collections into feature data (Priority: P1) MVP

**Goal**: Convert a NiMARE Studyset or Dataset into a scikit-learn-compatible feature dataset with one sample per eligible analysis, sparse map features, sample provenance, and study groups.

**Independent Test**: Given a collection with multiple studies and analyses, conversion produces one row per eligible analysis, aligned sparse map features, stable sample and study IDs, provenance, and an exclusion report.

### Tests and Examples for User Story 1

> Write these tests first and confirm they fail before implementation.

- [ ] T009 [US1] Add failing conversion contract tests for `MAFeatureExtractor.fit_transform`, `MAFeatureDataset.to_sklearn`, sparse map shape, provenance, and exclusion reports in `nimare/tests/test_ml.py`
- [ ] T010 [US1] Add failing compatibility tests for both Dataset and Studyset inputs without changing released collection or kernel behavior in `nimare/tests/test_ml.py`
- [ ] T011 [US1] Add failing study-group source tests for explicit study IDs, unambiguous analysis-ID derivation, and ambiguous grouping errors in `nimare/tests/test_ml.py`
- [ ] T012 [P] [US1] Fill the conversion and provenance workflow in `examples/05_machine_learning/01_plot_ma_feature_dataset.py`

### Implementation for User Story 1

- [ ] T013 [US1] Implement the `MAFeatureDataset` container, dimension validation, `copy`, `get_feature_names`, and map-only `to_sklearn` behavior in `nimare/ml.py`
- [ ] T014 [US1] Implement `MAFeatureExtractor.fit`, `transform`, and `fit_transform` using `KernelTransformer.transform(..., return_type="sparse")` in `nimare/ml.py`
- [ ] T015 [US1] Implement sample metadata, exclusion report, explicit study-ID grouping, unambiguous analysis-ID fallback, and ambiguous-grouping errors in `nimare/ml.py`
- [ ] T016 [US1] Add Numpydoc docstrings for conversion, provenance, grouping, and map-only export APIs in `nimare/ml.py`
- [ ] T017 [US1] Add the initial `nimare.ml` API autosummary entry and import documentation in `docs/api.rst`

**Checkpoint**: User Story 1 is independently testable with `python -m pytest nimare/tests/test_ml.py -k "feature or conversion or grouping"`.

---

## Phase 4: User Story 2 - Split data without study leakage (Priority: P1)

**Goal**: Provide reproducible grouped train/test splits that never place analyses from the same study in both train and test partitions.

**Independent Test**: Given a converted dataset with repeated study IDs, repeated splits with the same settings are identical, no study appears in both partitions, and too few study groups fails before returning a partial split.

### Tests and Examples for User Story 2

> Write these tests first and confirm they fail before implementation.

- [ ] T018 [US2] Add failing grouped split tests for no study leakage, reproducibility, aligned slicing, and too-few-study errors in `nimare/tests/test_ml.py`
- [ ] T019 [P] [US2] Extend the grouped split workflow in `examples/05_machine_learning/01_plot_ma_feature_dataset.py`

### Implementation for User Story 2

- [ ] T020 [US2] Implement `MAFeatureDataset.split` with scikit-learn `GroupShuffleSplit` and grouped CV support in `nimare/ml.py`
- [ ] T021 [US2] Implement dataset slicing for map features, sample metadata, target, descriptors, groups, feature names, and provenance in `nimare/ml.py`
- [ ] T022 [US2] Add split-specific diagnostics, `ValueError` messages, docstrings, and API docs in `nimare/ml.py` and `docs/api.rst`

**Checkpoint**: User Stories 1 and 2 are independently testable with `python -m pytest nimare/tests/test_ml.py -k "split or leakage or grouping"`.

---

## Phase 5: User Story 3 - Combine activation maps with study information (Priority: P2)

**Goal**: Add selected numeric metadata and annotation fields as descriptor features, reject non-numeric descriptors by default, and support explicit descriptor transformers or vectorizers.

**Independent Test**: Given selected descriptor fields, numeric descriptors are appended with aligned feature names; missing values are reported; non-numeric fields fail without an explicit transformer and succeed when one is provided.

### Tests and Examples for User Story 3

> Write these tests first and confirm they fail before implementation.

- [ ] T023 [US3] Add failing descriptor tests for numeric metadata/annotation append, descriptor feature names, sample alignment, and target-field-not-reused behavior in `nimare/tests/test_ml.py`
- [ ] T024 [US3] Add failing descriptor diagnostics tests for missing values, non-numeric field rejection, and explicit transformer/vectorizer handling in `nimare/tests/test_ml.py`
- [ ] T025 [P] [US3] Extend descriptor feature and explicit preprocessing examples in `examples/05_machine_learning/01_plot_ma_feature_dataset.py`

### Implementation for User Story 3

- [ ] T026 [US3] Implement descriptor field selector normalization and metadata/annotation/text table lookup in `nimare/ml.py`
- [ ] T027 [US3] Implement numeric descriptor extraction, sample alignment, descriptor feature names, and `to_sklearn(include_descriptors=True)` integration in `nimare/ml.py`
- [ ] T028 [US3] Implement non-numeric descriptor rejection, descriptor transformer/vectorizer execution, and missing descriptor reports in `nimare/ml.py`
- [ ] T029 [US3] Ensure descriptor transformers fit only on training data before held-out transformation in `nimare/ml.py`
- [ ] T030 [US3] Add descriptor-feature docstrings and API documentation in `nimare/ml.py` and `docs/api.rst`

**Checkpoint**: User Story 3 is independently testable with `python -m pytest nimare/tests/test_ml.py -k "descriptor or non_numeric"`.

---

## Phase 6: User Story 4 - Predict selected study information (Priority: P2)

**Goal**: Export one selected scalar numeric or categorical metadata/annotation value, or an explicitly transformed text-derived value, as the prediction target `y`.

**Independent Test**: Given a selected outcome field, target values align to feature rows; scalar numeric and categorical targets export correctly; missing, constant, raw free-text, and multi-label targets are diagnosed or rejected as specified.

### Tests and Examples for User Story 4

> Write these tests first and confirm they fail before implementation.

- [ ] T031 [US4] Add failing target tests for scalar numeric targets, scalar categorical targets, `y` alignment, and `to_sklearn(include_target=True)` behavior in `nimare/tests/test_ml.py`
- [ ] T032 [US4] Add failing target diagnostics tests for missing targets, constant targets, raw free-text rejection, multi-label rejection, and explicit target transformer handling in `nimare/tests/test_ml.py`
- [ ] T033 [P] [US4] Extend target extraction and exported `y` examples in `examples/05_machine_learning/01_plot_ma_feature_dataset.py`

### Implementation for User Story 4

- [ ] T034 [US4] Implement target field selector lookup, scalar numeric/categorical target validation, and target metadata in `nimare/ml.py`
- [ ] T035 [US4] Implement missing target reports, constant target diagnostics, and selected missing policy behavior in `nimare/ml.py`
- [ ] T036 [US4] Implement raw free-text and multi-label target rejection plus explicit target transformer or label extractor support in `nimare/ml.py`
- [ ] T037 [US4] Preserve target alignment through `to_sklearn`, `split`, and dataset slicing in `nimare/ml.py`
- [ ] T038 [US4] Add target-extraction docstrings and API documentation in `nimare/ml.py` and `docs/api.rst`

**Checkpoint**: User Story 4 is independently testable with `python -m pytest nimare/tests/test_ml.py -k "target or outcome"`.

---

## Phase 7: User Story 5 - Reduce voxelwise feature dimensionality (Priority: P3)

**Goal**: Provide reusable reduction workflows for voxelwise map features while preserving sample order, study groups, descriptors, and targets.

**Independent Test**: Given a feature dataset, variance thresholding, PCA/truncated SVD, and atlas/label aggregation reduce map features correctly; reducers fit only on training data and transform held-out samples without leakage.

### Tests and Examples for User Story 5

> Write these tests first and confirm they fail before implementation.

- [ ] T039 [US5] Add failing reducer factory tests for `variance_threshold`, dense `pca`, sparse `truncated_svd`, feature shapes, and feature names in `nimare/tests/test_ml.py`
- [ ] T040 [US5] Add failing atlas/label aggregation tests for compatible labels images, incompatible labels errors, and sample alignment in `nimare/tests/test_ml.py`
- [ ] T041 [US5] Add failing `apply_map_reducer` tests for train-fit, held-out-transform, row-count validation, descriptor preservation, and target preservation in `nimare/tests/test_ml.py`
- [ ] T042 [US5] Add a `performance_smoke` test for 1,000-study conversion and grouped split within <=3 minutes and <=5 GB peak memory in `nimare/tests/test_ml.py`
- [ ] T043 [P] [US5] Fill the reduction workflow example with truncated SVD plus references to variance thresholding, PCA, and atlas aggregation in `examples/05_machine_learning/02_plot_ma_feature_reduction.py`

### Implementation for User Story 5

- [ ] T044 [US5] Implement `make_map_reducer` factory support for `variance_threshold`, `pca`, and `truncated_svd` in `nimare/ml.py`
- [ ] T045 [US5] Implement atlas or label aggregation reducer support using nilearn-compatible masker or labels utilities in `nimare/ml.py`
- [ ] T046 [US5] Implement `MAFeatureDataset.apply_map_reducer` fit semantics, held-out transform behavior, row validation, and reduced feature names in `nimare/ml.py`
- [ ] T047 [US5] Preserve descriptor features, target values, study groups, sample metadata, and provenance across reduced dataset copies in `nimare/ml.py`
- [ ] T048 [US5] Optimize sparse reduction paths and add diagnostics for unintended densification or performance-budget risk in `nimare/ml.py`
- [ ] T049 [US5] Add reducer docstrings and API documentation in `nimare/ml.py` and `docs/api.rst`

**Checkpoint**: User Story 5 is independently testable with `python -m pytest nimare/tests/test_ml.py -k "reducer or reduction or performance_smoke"`.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Final verification, documentation consistency, and compatibility review across all user stories.

- [ ] T050 [P] Finalize Numpydoc examples, parameter docs, return docs, and error docs for every public class and function in `nimare/ml.py`
- [ ] T051 [P] Verify both Sphinx-Gallery examples use only public APIs and match the quickstart workflow in `examples/05_machine_learning/01_plot_ma_feature_dataset.py` and `examples/05_machine_learning/02_plot_ma_feature_reduction.py`
- [ ] T052 Run the targeted feature test suite with `python -m pytest nimare/tests/test_ml.py` for `nimare/tests/test_ml.py`
- [ ] T053 Run the performance smoke check with `python -m pytest -m performance_smoke nimare/tests/test_ml.py` for `nimare/tests/test_ml.py`
- [ ] T054 Run the documentation build with `make -C docs html` for `docs/api.rst` and `examples/05_machine_learning/`
- [ ] T055 Run linting with `make lint` for `nimare/ml.py`, `nimare/tests/test_ml.py`, `docs/api.rst`, and `examples/05_machine_learning/`
- [ ] T056 Review public API compatibility against baseline `0.16.0` for `nimare/__init__.py`, `nimare/ml.py`, and `docs/api.rst`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies; can start immediately.
- **Foundational (Phase 2)**: Depends on Setup completion; blocks all user stories.
- **User Story 1 and User Story 2 (P1)**: Depend on Foundational completion.
- **User Story 3 and User Story 4 (P2)**: Depend on Foundational completion and integrate with the dataset container from User Story 1.
- **User Story 5 (P3)**: Depends on Foundational completion and the dataset container from User Story 1; descriptor/target preservation checks depend on User Stories 3 and 4 when those stories are in scope.
- **Polish (Phase 8)**: Depends on all desired user stories being complete.

### User Story Dependencies

- **US1 Convert collections into feature data**: MVP; no dependency on other user stories after Foundational.
- **US2 Split data without study leakage**: Requires `MAFeatureDataset` rows, sample metadata, and study groups from US1.
- **US3 Combine activation maps with study information**: Requires the dataset container and export path from US1.
- **US4 Predict selected study information**: Requires the dataset container and export path from US1; split alignment integrates with US2.
- **US5 Reduce voxelwise feature dimensionality**: Requires map features from US1 and split semantics from US2; descriptor/target preservation integrates with US3/US4.

### Within Each User Story

- Tests must be written and fail before implementation.
- Public examples must be created or edited before the story is considered complete.
- Core implementation in `nimare/ml.py` follows the story tests.
- Documentation and docstrings close each story.

---

## Parallel Opportunities

- T002, T003, and T004 can run in parallel after T001 is understood because they touch different files.
- T012 can run in parallel with US1 test authoring because it touches `examples/05_machine_learning/01_plot_ma_feature_dataset.py`.
- T019, T025, and T033 can run in parallel with their story test tasks if prior edits to `examples/05_machine_learning/01_plot_ma_feature_dataset.py` are already merged.
- T043 can run in parallel with US5 test authoring because it touches `examples/05_machine_learning/02_plot_ma_feature_reduction.py`.
- T050 and T051 can run in parallel near the end because they touch `nimare/ml.py` and example files.

## Parallel Example: User Story 1

```text
Task: "Add failing conversion contract tests in nimare/tests/test_ml.py"
Task: "Fill conversion workflow example in examples/05_machine_learning/01_plot_ma_feature_dataset.py"
```

## Parallel Example: User Story 3

```text
Task: "Add failing descriptor diagnostics tests in nimare/tests/test_ml.py"
Task: "Extend descriptor feature examples in examples/05_machine_learning/01_plot_ma_feature_dataset.py"
```

## Parallel Example: User Story 5

```text
Task: "Add failing reducer factory tests in nimare/tests/test_ml.py"
Task: "Fill reduction workflow example in examples/05_machine_learning/02_plot_ma_feature_reduction.py"
```

---

## Implementation Strategy

### MVP First

1. Complete Phase 1 and Phase 2.
2. Complete Phase 3 for User Story 1.
3. Run `python -m pytest nimare/tests/test_ml.py -k "feature or conversion or grouping"`.
4. Stop and review the MVP before adding descriptor, target, or reduction behavior.

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
- Keep sparse masked activation matrices sparse unless the caller explicitly requests dense output.
- Reject non-numeric descriptor features by default without explicit preprocessing.
- Reject raw free-text and multi-label targets by default without explicit target handling.
- Create examples as `.py` Sphinx-Gallery sources; generated notebooks come from the docs build.
