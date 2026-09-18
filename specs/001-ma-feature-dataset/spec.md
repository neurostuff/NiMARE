# Feature Specification: Masked Activation Feature Dataset

**Feature Branch**: `001-ma-feature-dataset`  
**Created**: 2026-05-01  
**Status**: Draft  
**Input**: User description: "I want to create a new module in NiMARE that takes the Masked Activation maps generated from the kernel functions and treats them like features that can be used in scikit-learn estimators, I want to be able to take a Studyset and transform it into a compatible scikit-learn dataset that I can split into training/testing with appropriate splits (analyses stay with the studies/a study's analyses are not split between training and testing, they stay together). I also want metadata/annotations to be able to be added as additional features to the scikit-learn dataset (or an annotation/metadata/title/description could be used as an outcome measure (y) that the scikit-learn model will try to predict), and have some convienence data reduction pipelines for the voxelwise masked activation maps."

## Clarifications

### Session 2026-05-01

- Q: When metadata, annotations, titles, abstracts, or descriptions are added as additional features, how should non-numeric fields behave by default? -> A: Reject non-numeric descriptor fields by default unless the caller provides an explicit transformer or vectorizer.
- Q: For grouped splitting, how should the feature determine each analysis row's study group by default? -> A: Use Studyset-provided study IDs; MVP inputs are assumed to provide unique study IDs and unique analysis IDs.
- Q: When an annotation, metadata field, title, or description is used as the prediction target, what should the default target handling be? -> A: Export scalar numeric or categorical targets as y; reject free-text or multi-label targets unless an explicit target transformer or label extractor is provided.
- Q: Which convenience data reduction workflows should be required for the initial feature? -> A: Include variance thresholding, sparse-compatible low-rank reduction such as truncated SVD, and atlas or label aggregation when a masker or labels image is supplied.
- Q: What runtime and memory target should replace the current placeholder performance criterion? -> A: A representative Studyset with at least 1,000 studies must convert and split in <=3 minutes with <=5 GB peak memory.

### Session 2026-05-20

- Q: Should `MAFeatureExtractor` expose an sklearn-style `fit`/`fit_transform` API? -> A: No; use `transform(studyset)` for dataset-level conversion and `to_sklearn(studyset, ...)` for one-call sklearn export.
- Q: When may feature data be represented densely? -> A: Unreduced voxelwise feature data must remain sparse; dense output is allowed only after an explicit reducer creates a reduced representation, and dense PCA over unreduced voxels is not required.
- Q: How should the MVP handle study and analysis identifier ambiguity? -> A: Assume input Studysets provide unique study IDs and unique analysis IDs; duplicate or missing identifiers are out of scope for the MVP.
- Q: How should analyses with no coordinates be handled? -> A: Provide `missing_coordinates` with `include` and `drop` modes; default `drop` removes them before row construction and records dropped IDs in provenance, while `include` keeps them as all-zero sparse map rows.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Convert Studysets into feature data (Priority: P1)

A meta-analysis researcher wants to turn an existing NiMARE Studyset into a machine-learning-ready dataset where each analysis has masked activation map features, identifiers, and study grouping information.

**Why this priority**: This is the core value of the feature. Without reliable conversion from existing NiMARE Studysets, no downstream modeling workflow can use the generated activation map features.

**Independent Test**: Given a Studyset with multiple studies and analyses, the researcher can produce a feature dataset with one analysis row per retained analysis, voxelwise masked activation values, stable analysis identifiers, stable study identifiers, missing-coordinate option provenance, and enough provenance to trace each analysis row back to its source.
**First Failing Test**: A failing regression test demonstrates that converting a representative Studyset produces the expected analysis-row count, feature count, identifiers, study groups, and provenance.
**Public Example**: `examples/05_machine_learning/01_plot_ma_feature_dataset.py`

**Acceptance Scenarios**:

1. **Given** a Studyset with three studies and multiple analyses per study, **When** the researcher converts it into feature data, **Then** every analysis appears exactly once as an analysis row with its study identifier attached.
2. **Given** a Studyset and a selected activation-map generation strategy, **When** conversion completes, **Then** the feature data contains masked activation map values aligned to the same mask for every analysis row.
3. **Given** a converted feature dataset, **When** the researcher inspects analysis-row metadata, **Then** they can identify the original study, analysis, and activation-map generation settings for each analysis row.
4. **Given** a Studyset with analyses that have no coordinates, **When** the researcher converts with missing coordinates set to include, **Then** coordinate-less analyses remain as analysis rows with all-zero sparse map rows.
5. **Given** a Studyset with analyses that have no coordinates, **When** the researcher converts with missing coordinates set to drop, **Then** coordinate-less analyses are removed before row construction and their analysis IDs are recorded in provenance.

---

### User Story 2 - Split data without study leakage (Priority: P1)

A researcher wants training and testing partitions that keep all analyses from the same study together so that related analyses do not leak across model evaluation splits.

**Why this priority**: Leakage across studies would invalidate model evaluation and undermine trust in any predictive results.

**Independent Test**: Given a converted dataset with multiple analyses per study, repeated randomized splits always assign all analyses from each study to exactly one partition.
**First Failing Test**: A failing regression test shows that a grouped split rejects leakage and preserves study-level grouping across training and testing outputs.
**Public Example**: `examples/05_machine_learning/01_plot_ma_feature_dataset.py`

**Acceptance Scenarios**:

1. **Given** a feature dataset with repeated study identifiers, **When** the researcher requests a train/test split, **Then** no study identifier appears in both training and testing partitions.
2. **Given** the same dataset and split settings, **When** the split is repeated with the same reproducibility setting, **Then** the resulting partitions are identical.
3. **Given** a Studyset that has too few studies to create the requested split, **When** the researcher requests the split, **Then** the request fails with a clear explanation and no partial split is returned.

---

### User Story 3 - Combine activation maps with study information (Priority: P2)

A researcher wants to add selected metadata, annotations, titles, or descriptions as additional model features alongside masked activation map features.

**Why this priority**: Many useful prediction tasks need both neuroimaging-derived features and study-level or analysis-level descriptors.

**Independent Test**: Given a converted dataset and selected descriptive fields, the researcher can produce an augmented feature dataset with aligned non-image features and transparent missing-value handling.
**First Failing Test**: A failing regression test verifies that selected numeric descriptor fields are added only to matching analysis rows, non-numeric descriptor fields fail without an explicit transformer or vectorizer, and missing or unavailable fields are reported clearly.
**Public Example**: `examples/05_machine_learning/01_plot_ma_feature_dataset.py`

**Acceptance Scenarios**:

1. **Given** selected annotation and metadata fields, **When** the researcher augments the feature dataset, **Then** those fields appear as named features aligned to the correct analysis rows.
2. **Given** selected categorical, title, or description fields, **When** the researcher adds them as features without an explicit transformer or vectorizer, **Then** the request fails with a clear message identifying the non-numeric fields.
3. **Given** selected fields with missing values, **When** no missing-value strategy is selected, **Then** the feature reports the affected analyses and fields rather than silently dropping or filling them.

---

### User Story 4 - Predict selected study information (Priority: P2)

A researcher wants to use an annotation, metadata field, title-derived label, or description-derived label as the prediction target for a model trained from activation map and optional descriptor features.

**Why this priority**: The feature becomes useful for supervised modeling only when outcomes can be selected, aligned, validated, and exported with the same analysis-row grouping as the feature matrix.

**Independent Test**: Given a selected outcome field, the researcher can produce a target vector aligned to the feature analysis rows, with a clear report of target type, missing values, and retained analyses.
**First Failing Test**: A failing regression test verifies scalar target extraction, analysis-row alignment, missing target reporting, unsupported target-shape errors, and grouped splitting with the target attached.
**Public Example**: `examples/05_machine_learning/01_plot_ma_feature_dataset.py`

**Acceptance Scenarios**:

1. **Given** a scalar categorical annotation field, **When** the researcher selects it as the outcome, **Then** the target values are exported as an aligned y vector in the same analysis-row order as the feature dataset.
2. **Given** a scalar numeric metadata field, **When** the researcher selects it as the outcome, **Then** the target preserves numeric values and reports analyses where the outcome is unavailable.
3. **Given** a free-text or multi-label field, **When** the researcher selects it as the outcome without an explicit target transformer or label extractor, **Then** the request fails with a clear message identifying the unsupported target shape.
4. **Given** an outcome that exists only at the study level, **When** the dataset is split, **Then** all analyses sharing that study-level outcome remain in the same partition.

---

### User Story 5 - Reduce voxelwise feature dimensionality (Priority: P3)

A researcher wants convenient, reusable reduction workflows for high-dimensional masked activation map features before using them in a predictive model.

**Why this priority**: Voxelwise activation map features can be too large for efficient or stable modeling without common reduction options.

**Independent Test**: Given a feature dataset and a selected reduction workflow, the researcher can obtain a transformed feature dataset with fewer map-derived features while preserving analysis-row order, identifiers, study groups, and optional targets.
**First Failing Test**: A failing regression test verifies that each convenience reduction workflow preserves analysis-row alignment and produces the expected reduced feature shape.
**Public Example**: `examples/05_machine_learning/02_plot_ma_feature_reduction.py`

**Acceptance Scenarios**:

1. **Given** a high-dimensional activation map feature dataset, **When** the researcher applies a reduction workflow, **Then** the transformed dataset has fewer map-derived features and the same analysis-row order.
2. **Given** a fitted reduction workflow, **When** it is applied to a held-out partition, **Then** the held-out data is transformed without using held-out outcomes or refitting on held-out analyses.
3. **Given** optional descriptor features and targets, **When** map features are reduced, **Then** descriptor features and targets remain aligned with the transformed map features.
4. **Given** a feature dataset with low-variance map features, **When** variance thresholding is selected, **Then** low-variance map features are removed while analysis-row alignment is preserved.
5. **Given** a sparse voxelwise feature dataset, **When** sparse low-rank reduction is selected, **Then** the reducer uses a sparse-compatible decomposition such as truncated SVD and returns the requested number of components without densifying the unreduced voxel matrix.
6. **Given** a masker or labels image that defines regions, **When** atlas or label aggregation is selected, **Then** map features are aggregated into region-level features aligned to the original analysis rows.

### Edge Cases

- A Studyset contains studies with one analysis and studies with many analyses.
- A Studyset lacks unique study identifiers or unique analysis identifiers; this is outside the MVP input contract.
- A Studyset has too few studies to support the requested split ratio or cross-validation design.
- Analyses have no coordinates and must follow the selected missing-coordinate
  policy.
- Analyses with coordinates cannot produce a valid masked activation map.
- Input Studysets mix spaces or masks in a way that prevents aligned map features.
- Selected metadata, annotation, title, or description fields are absent, duplicated, or only available for some analyses.
- Selected outcomes have missing values, constant values, rare classes, or multiple labels per analysis.
- A study-level descriptor is repeated across analyses and could leak information if analyses were split independently.
- High-dimensional map features exceed the required <=3 minute conversion-and-split runtime or <=5 GB peak memory budget for a representative 1,000-study Studyset.
- Reduction workflows are requested before map features are available or after analysis-row order has changed.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The feature MUST accept existing NiMARE Studysets as input and create a machine-learning-ready dataset with one analysis row per analysis by default.
- **FR-002**: The feature MUST generate or consume masked activation map values for each analysis and align those values to a common feature space.
- **FR-003**: The feature MUST preserve source provenance for every analysis row, including study identifier, analysis identifier, and activation-map generation settings.
- **FR-004**: The feature MUST expose study grouping information so that all analyses from the same study can be kept together during data splitting, assuming the input Studyset provides unique study identifiers and unique analysis identifiers.
- **FR-005**: The feature MUST provide a train/test split workflow that prevents analyses from the same study from appearing in both training and testing partitions.
- **FR-006**: The feature MUST support reproducible splits when the researcher supplies the same split configuration.
- **FR-007**: The feature MUST allow selected numeric metadata and annotation fields to be added as additional model features.
- **FR-008**: The feature MUST reject non-numeric descriptor fields, including categorical metadata, annotations, titles, and descriptions, unless the researcher supplies an explicit transformer or vectorizer for converting them into numeric features.
- **FR-009**: The feature MUST allow one selected scalar numeric or categorical annotation or metadata value, or one value produced by an explicit target transformer or label extractor, to be exported as the prediction target y, and MUST reject raw free-text or multi-label targets unless such explicit target handling is supplied.
- **FR-010**: The feature MUST keep the feature data, target values, analysis identifiers, and study groups aligned through conversion, augmentation, splitting, and reduction.
- **FR-011**: The feature MUST report missing or unusable map, descriptor, and target values with enough detail for the researcher to correct inputs or choose an explicit handling strategy.
- **FR-012**: The feature MUST provide sparse-safe convenience reduction workflows for voxelwise masked activation map features while preserving analysis-row alignment and study grouping, including variance thresholding, truncated SVD or equivalent sparse-compatible low-rank reduction, and atlas or label aggregation when a masker or labels image is supplied.
- **FR-013**: The feature MUST prevent data leakage by ensuring any learned reduction or descriptor transformation is fit only on training data before being applied to held-out data.
- **FR-014**: The feature MUST expose Studyset conversion through `MAFeatureExtractor.transform(studyset)` as the advanced dataset-level pathway returning `(train_dataset, test_dataset)` where `test_dataset` is `None` when no split is requested; `MAFeatureExtractor.to_sklearn(studyset, ...)` MUST provide the one-call sklearn-ready export pathway. `MAFeatureExtractor` MUST NOT expose `fit` or `fit_transform` in the initial public API.
- **FR-015**: The feature MUST represent unreduced voxelwise feature data as sparse numeric matrices throughout conversion, export, and splitting; dense feature data is allowed only after an explicit reducer creates a reduced representation.
- **FR-016**: The feature MUST provide a user-facing example that demonstrates Studyset conversion, grouped splitting, descriptor features, target extraction, and at least one reduction workflow.
- **FR-017**: The feature MUST be additive with respect to released NiMARE public behavior; existing Studyset, kernel, metadata, annotation, and documentation workflows must continue to work.
- **FR-018**: The feature MUST provide an explicit `missing_coordinates` option on `MAFeatureExtractor` with `include` and `drop` modes. `drop` MUST be the default and MUST remove coordinate-less analyses before row construction while recording dropped IDs in provenance; `include` MUST retain coordinate-less analyses as all-zero sparse map rows.

### Key Entities *(include if feature involves data)*

- **Studyset**: A NiMARE Studyset containing studies and analyses that can provide coordinates, metadata, annotations, and text fields.
- **Analysis Row**: One analysis represented as a single machine-learning row with stable identifiers and provenance.
- **Masked Activation Feature Matrix**: The aligned map-derived feature values for all analysis rows.
- **Descriptor Feature Set**: Optional metadata, annotation, title, or description information added as additional numeric analysis-row features, with non-numeric inputs requiring an explicit transformer or vectorizer.
- **Prediction Target**: A selected scalar numeric or categorical metadata or annotation value, or a value produced from title or description text by an explicit target transformer or label extractor, aligned to analysis rows as y; raw free-text and multi-label targets require explicit target handling.
- **Study Group**: The grouping key that keeps all analyses from the same study together in splits.
- **Split Plan**: The train/test or validation partition assignment that preserves study groups.
- **Reduction Workflow**: A reusable transformation that reduces masked activation map feature dimensionality while preserving analysis-row order and metadata alignment; required initial workflows are variance thresholding, truncated SVD or equivalent sparse-compatible low-rank reduction, and atlas or label aggregation when a masker or labels image is supplied.

### Public API & Compatibility *(mandatory for code changes)*

- **Latest Release Baseline**: 0.16.0
- **Public API Surface**: New additive public surface for creating machine-learning-ready outputs from existing NiMARE Studysets through `MAFeatureExtractor.to_sklearn(studyset, ...)`, adding descriptor features, extracting targets, performing grouped splits, and applying reduction workflows. `MAFeatureExtractor.transform(studyset)` is retained as the advanced dataset-level pathway for reducer iteration.
- **Compatibility Requirement**: Preserve existing released public behavior for Studysets, masked activation map generation, metadata, annotations, and text access. New functionality is expected to be additive.
- **Migration/Deprecation Notes**: No migration or deprecation is expected for existing released APIs.
- **Sphinx-Gallery Example**: `examples/05_machine_learning/01_plot_ma_feature_dataset.py` and `examples/05_machine_learning/02_plot_ma_feature_reduction.py` created or edited.

### Scientific & Reproducibility Requirements *(include if results can change)*

- **Scientific Assumptions**: Analysis rows represent analyses by default; analyses from the same study are statistically related and must be grouped during model evaluation; masked activation map features are comparable only when aligned to a common mask/space.
- **Validation Evidence**: Validation must include fixture Studysets with multiple analyses per study, known study group assignments, selected descriptor fields, selected targets, missing-field cases, and expected reduced feature shapes.
- **Diagnostics**: Invalid or degraded inputs must produce clear messages for unavailable fields, incompatible masks/spaces, insufficient studies for splitting, missing or unusable outcomes, and analyses dropped by the missing-coordinate option.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For 100% of generated train/test splits in validation fixtures, no study identifier appears in more than one partition.
- **SC-002**: Converted feature datasets contain exactly one analysis row for every analysis when coordinate-less analyses are included, or exactly one analysis row for every retained analysis when coordinate-less analyses are dropped; invalid map rows unrelated to missing coordinates fail clearly.
- **SC-003**: Descriptor features and selected targets remain aligned with analysis rows after conversion, splitting, and reduction in all validation fixtures.
- **SC-004**: At least three common modeling workflows are demonstrated: map-only prediction, map-plus-descriptor prediction, and reduced-map prediction using variance thresholding, truncated SVD or equivalent sparse-compatible low-rank reduction, or atlas or label aggregation.
- **SC-005**: Users can complete the documented example workflow on a representative test Studyset without manually editing intermediate data.
- **SC-006**: Documentation examples convert successfully during the documentation build and produce rendered gallery outputs for the public workflow.
- **SC-007**: Missing descriptor or target data is never silently dropped or filled; validation fixtures report the affected analysis identifiers and fields.
- **SC-008**: A representative Studyset with at least 1,000 studies can be converted and split in <=3 minutes with <=5 GB peak memory in the standard development environment.

## Assumptions

- Each analysis is the default machine-learning row because the requested split rule specifically keeps analyses grouped by study.
- Coordinate-less analyses are dropped by default before row construction unless
  the researcher selects the include policy.
- The MVP assumes input Studysets provide unique study IDs and unique analysis IDs; deriving study groups from ambiguous or missing identifiers is out of scope.
- Study-level metadata may be repeated across that study's analyses, but grouped splitting prevents study-level leakage between training and testing partitions.
- "Scikit-learn-compatible" means the exported data can be consumed by common estimator workflows that expect aligned feature values, target values, and grouping labels.
- `MAFeatureExtractor` is a NiMARE conversion helper, not a trainable scikit-learn estimator; downstream scikit-learn models consume the output of `MAFeatureExtractor.to_sklearn(studyset, ...)`.
- Repeated reducer experiments should reuse cached MA map generation whenever Studyset and extraction settings are unchanged.
- Unreduced voxelwise feature data is sparse-only; dense data is permitted only after explicit reduction, such as a low-rank component matrix or parcel-level aggregate.
- The feature will not train or evaluate predictive models itself beyond providing data structures, split helpers, and reduction workflows needed by researchers.
- Missing descriptor or outcome values fail clearly by default unless the researcher explicitly selects a handling strategy.
- Study grouping is mandatory for split helpers and is read from the Studyset's study IDs.
- Existing kernel and Studyset behavior from the latest release remains the compatibility baseline.
- Coordinate-less analyses are not an error by themselves; they are handled by
  `MAFeatureExtractor.missing_coordinates`.
