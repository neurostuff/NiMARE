# Feature Specification: Masked Activation Feature Dataset

**Feature Branch**: `001-ma-feature-dataset`  
**Created**: 2026-05-01  
**Status**: Draft  
**Input**: User description: "I want to create a new module in NiMARE that takes the Masked Activation maps generated from the kernel functions and treats them like features that can be used in scikit-learn estimators, I want to be able to take any studyset/dataset and transform it into a compatible scikit-learn dataset that I can split into training/testing with appropriate splits (analyses stay with the studies/a study's analyses are not split between training and testing, they stay together). I also want metadata/annotations to be able to be added as additional features to the scikit-learn dataset (or an annotation/metadata/title/description could be used as an outcome measure (y) that the scikit-learn model will try to predict), and have some convienence data reduction pipelines for the voxelwise masked activation maps."

## Clarifications

### Session 2026-05-01

- Q: When metadata, annotations, titles, abstracts, or descriptions are added as additional features, how should non-numeric fields behave by default? -> A: Reject non-numeric descriptor fields by default unless the caller provides an explicit transformer or vectorizer.
- Q: For grouped splitting, how should the feature determine each sample's study group by default? -> A: Use explicit study IDs from the collection when available; otherwise derive groups from analysis IDs only when unambiguous; fail if grouping cannot be determined.
- Q: When an annotation, metadata field, title, or description is used as the prediction target, what should the default target handling be? -> A: Export scalar numeric or categorical targets as y; reject free-text or multi-label targets unless an explicit target transformer or label extractor is provided.
- Q: Which convenience data reduction workflows should be required for the initial feature? -> A: Include variance thresholding, PCA or truncated SVD as appropriate for the matrix type, and atlas or label aggregation when a masker or labels image is supplied.
- Q: What runtime and memory target should replace the current placeholder performance criterion? -> A: A representative collection with at least 1,000 studies must convert and split in <=3 minutes with <=5 GB peak memory.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Convert collections into feature data (Priority: P1)

A meta-analysis researcher wants to turn an existing NiMARE study collection into a machine-learning-ready dataset where each analysis has masked activation map features, identifiers, and study grouping information.

**Why this priority**: This is the core value of the feature. Without reliable conversion from existing NiMARE collections, no downstream modeling workflow can use the generated activation map features.

**Independent Test**: Given a collection with multiple studies and analyses, the researcher can produce a feature dataset with one sample per analysis, voxelwise masked activation values, stable analysis identifiers, stable study identifiers, and enough provenance to trace each sample back to its source.
**First Failing Test**: A failing regression test demonstrates that converting a representative collection produces the expected sample count, feature count, identifiers, study groups, and provenance.
**Public Example**: `examples/05_machine_learning/01_plot_ma_feature_dataset.py`

**Acceptance Scenarios**:

1. **Given** a collection with three studies and multiple analyses per study, **When** the researcher converts it into feature data, **Then** every eligible analysis appears exactly once as a sample with its study identifier attached.
2. **Given** a collection and a selected activation-map generation strategy, **When** conversion completes, **Then** the feature data contains masked activation map values aligned to the same mask for every sample.
3. **Given** a collection without explicit study identifiers but with unambiguous NiMARE analysis identifiers, **When** conversion completes, **Then** study groups are derived consistently from those analysis identifiers.
4. **Given** a collection where study grouping cannot be determined, **When** conversion is requested, **Then** the request fails with a clear explanation and no feature dataset is returned.
5. **Given** a converted feature dataset, **When** the researcher inspects sample metadata, **Then** they can identify the original study, analysis, and activation-map generation settings for each sample.

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
3. **Given** a collection that has too few studies to create the requested split, **When** the researcher requests the split, **Then** the request fails with a clear explanation and no partial split is returned.

---

### User Story 3 - Combine activation maps with study information (Priority: P2)

A researcher wants to add selected metadata, annotations, titles, or descriptions as additional model features alongside masked activation map features.

**Why this priority**: Many useful prediction tasks need both neuroimaging-derived features and study-level or analysis-level descriptors.

**Independent Test**: Given a converted dataset and selected descriptive fields, the researcher can produce an augmented feature dataset with aligned non-image features and transparent missing-value handling.
**First Failing Test**: A failing regression test verifies that selected numeric descriptor fields are added only to matching samples, non-numeric descriptor fields fail without an explicit transformer or vectorizer, and missing or unavailable fields are reported clearly.
**Public Example**: `examples/05_machine_learning/01_plot_ma_feature_dataset.py`

**Acceptance Scenarios**:

1. **Given** selected annotation and metadata fields, **When** the researcher augments the feature dataset, **Then** those fields appear as named features aligned to the correct analysis samples.
2. **Given** selected categorical, title, or description fields, **When** the researcher adds them as features without an explicit transformer or vectorizer, **Then** the request fails with a clear message identifying the non-numeric fields.
3. **Given** selected fields with missing values, **When** no missing-value strategy is selected, **Then** the feature reports the affected samples and fields rather than silently dropping or filling them.

---

### User Story 4 - Predict selected study information (Priority: P2)

A researcher wants to use an annotation, metadata field, title-derived label, or description-derived label as the prediction target for a model trained from activation map and optional descriptor features.

**Why this priority**: The feature becomes useful for supervised modeling only when outcomes can be selected, aligned, validated, and exported with the same sample grouping as the feature matrix.

**Independent Test**: Given a selected outcome field, the researcher can produce a target vector aligned to the feature samples, with a clear report of target type, missing values, and retained samples.
**First Failing Test**: A failing regression test verifies scalar target extraction, sample alignment, missing target reporting, unsupported target-shape errors, and grouped splitting with the target attached.
**Public Example**: `examples/05_machine_learning/01_plot_ma_feature_dataset.py`

**Acceptance Scenarios**:

1. **Given** a scalar categorical annotation field, **When** the researcher selects it as the outcome, **Then** the target values are exported as an aligned y vector in the same sample order as the feature dataset.
2. **Given** a scalar numeric metadata field, **When** the researcher selects it as the outcome, **Then** the target preserves numeric values and reports samples where the outcome is unavailable.
3. **Given** a free-text or multi-label field, **When** the researcher selects it as the outcome without an explicit target transformer or label extractor, **Then** the request fails with a clear message identifying the unsupported target shape.
4. **Given** an outcome that exists only at the study level, **When** the dataset is split, **Then** all analyses sharing that study-level outcome remain in the same partition.

---

### User Story 5 - Reduce voxelwise feature dimensionality (Priority: P3)

A researcher wants convenient, reusable reduction workflows for high-dimensional masked activation map features before using them in a predictive model.

**Why this priority**: Voxelwise activation map features can be too large for efficient or stable modeling without common reduction options.

**Independent Test**: Given a feature dataset and a selected reduction workflow, the researcher can obtain a transformed feature dataset with fewer map-derived features while preserving sample order, identifiers, study groups, and optional targets.
**First Failing Test**: A failing regression test verifies that each convenience reduction workflow preserves sample alignment and produces the expected reduced feature shape.
**Public Example**: `examples/05_machine_learning/02_plot_ma_feature_reduction.py`

**Acceptance Scenarios**:

1. **Given** a high-dimensional activation map feature dataset, **When** the researcher applies a reduction workflow, **Then** the transformed dataset has fewer map-derived features and the same sample order.
2. **Given** a fitted reduction workflow, **When** it is applied to a held-out partition, **Then** the held-out data is transformed without using held-out outcomes or refitting on held-out samples.
3. **Given** optional descriptor features and targets, **When** map features are reduced, **Then** descriptor features and targets remain aligned with the transformed map features.
4. **Given** a feature dataset with low-variance map features, **When** variance thresholding is selected, **Then** low-variance map features are removed while sample alignment is preserved.
5. **Given** a dense or sparse feature dataset, **When** PCA or truncated SVD is selected, **Then** the reducer uses the matrix-appropriate decomposition and returns the requested number of components.
6. **Given** a masker or labels image that defines regions, **When** atlas or label aggregation is selected, **Then** map features are aggregated into region-level features aligned to the original samples.

### Edge Cases

- A collection contains studies with one analysis and studies with many analyses.
- A collection lacks explicit study identifiers, and study groups must be derived from unambiguous analysis identifiers or fail clearly.
- A collection has too few studies to support the requested split ratio or cross-validation design.
- Analyses have no coordinates or cannot produce a valid masked activation map.
- Input collections mix spaces or masks in a way that prevents aligned map features.
- Selected metadata, annotation, title, or description fields are absent, duplicated, or only available for some samples.
- Selected outcomes have missing values, constant values, rare classes, or multiple labels per sample.
- A study-level descriptor is repeated across analyses and could leak information if analyses were split independently.
- High-dimensional map features exceed the required <=3 minute conversion-and-split runtime or <=5 GB peak memory budget for a representative 1,000-study collection.
- Reduction workflows are requested before map features are available or after sample order has changed.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The feature MUST accept existing NiMARE study collections as input and create a machine-learning-ready dataset with one sample per eligible analysis by default.
- **FR-002**: The feature MUST generate or consume masked activation map values for each eligible analysis and align those values to a common feature space.
- **FR-003**: The feature MUST preserve source provenance for every sample, including study identifier, analysis identifier, and activation-map generation settings.
- **FR-004**: The feature MUST expose study grouping information so that all analyses from the same study can be kept together during data splitting, using explicit study identifiers when available and deriving groups from analysis identifiers only when the derivation is unambiguous.
- **FR-005**: The feature MUST provide a train/test split workflow that prevents analyses from the same study from appearing in both training and testing partitions.
- **FR-006**: The feature MUST support reproducible splits when the researcher supplies the same split configuration.
- **FR-007**: The feature MUST allow selected numeric metadata and annotation fields to be added as additional model features.
- **FR-008**: The feature MUST reject non-numeric descriptor fields, including categorical metadata, annotations, titles, and descriptions, unless the researcher supplies an explicit transformer or vectorizer for converting them into numeric features.
- **FR-009**: The feature MUST allow one selected scalar numeric or categorical annotation or metadata value, or one value produced by an explicit target transformer or label extractor, to be exported as the prediction target y, and MUST reject raw free-text or multi-label targets unless such explicit target handling is supplied.
- **FR-010**: The feature MUST keep the feature data, target values, sample identifiers, and study groups aligned through conversion, augmentation, splitting, and reduction.
- **FR-011**: The feature MUST report missing or unusable map, descriptor, and target values with enough detail for the researcher to correct inputs or choose an explicit handling strategy.
- **FR-012**: The feature MUST provide convenience reduction workflows for voxelwise masked activation map features while preserving sample alignment and study grouping, including variance thresholding, matrix-appropriate PCA or truncated SVD, and atlas or label aggregation when a masker or labels image is supplied.
- **FR-013**: The feature MUST prevent data leakage by ensuring any learned reduction or descriptor transformation is fit only on training data before being applied to held-out data.
- **FR-014**: The feature MUST provide a user-facing example that demonstrates collection conversion, grouped splitting, descriptor features, target extraction, and at least one reduction workflow.
- **FR-015**: The feature MUST be additive with respect to released NiMARE public behavior; existing collection, kernel, metadata, annotation, and documentation workflows must continue to work.

### Key Entities *(include if feature involves data)*

- **Study Collection**: A NiMARE collection containing studies and analyses that can provide coordinates, metadata, annotations, and text fields.
- **Analysis Sample**: One eligible analysis represented as a single machine-learning sample with stable identifiers and provenance.
- **Masked Activation Feature Matrix**: The aligned map-derived feature values for all eligible analysis samples.
- **Descriptor Feature Set**: Optional metadata, annotation, title, or description information added as additional numeric sample features, with non-numeric inputs requiring an explicit transformer or vectorizer.
- **Prediction Target**: A selected scalar numeric or categorical metadata or annotation value, or a value produced from title or description text by an explicit target transformer or label extractor, aligned to analysis samples as y; raw free-text and multi-label targets require explicit target handling.
- **Study Group**: The grouping key that keeps all analyses from the same study together in splits.
- **Split Plan**: The train/test or validation partition assignment that preserves study groups.
- **Reduction Workflow**: A reusable transformation that reduces masked activation map feature dimensionality while preserving sample order and metadata alignment; required initial workflows are variance thresholding, matrix-appropriate PCA or truncated SVD, and atlas or label aggregation when a masker or labels image is supplied.

### Public API & Compatibility *(mandatory for code changes)*

- **Latest Release Baseline**: 0.16.0
- **Public API Surface**: New additive public surface for creating machine-learning-ready feature datasets from existing NiMARE study collections, adding descriptor features, extracting targets, performing grouped splits, and applying reduction workflows.
- **Compatibility Requirement**: Preserve existing released public behavior for study collections, masked activation map generation, metadata, annotations, and text access. New functionality is expected to be additive.
- **Migration/Deprecation Notes**: No migration or deprecation is expected for existing released APIs.
- **Sphinx-Gallery Example**: `examples/05_machine_learning/01_plot_ma_feature_dataset.py` and `examples/05_machine_learning/02_plot_ma_feature_reduction.py` created or edited.

### Scientific & Reproducibility Requirements *(include if results can change)*

- **Scientific Assumptions**: Samples represent analyses by default; analyses from the same study are statistically related and must be grouped during model evaluation; masked activation map features are comparable only when aligned to a common mask/space.
- **Validation Evidence**: Validation must include fixture collections with multiple analyses per study, known study group assignments, selected descriptor fields, selected targets, missing-field cases, and expected reduced feature shapes.
- **Diagnostics**: Invalid or degraded inputs must produce clear messages for missing maps, unavailable fields, incompatible masks/spaces, insufficient studies for splitting, and missing or unusable outcomes.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For 100% of generated train/test splits in validation fixtures, no study identifier appears in more than one partition.
- **SC-002**: Converted feature datasets contain exactly one sample for every eligible analysis and report every excluded analysis with a reason.
- **SC-003**: Descriptor features and selected targets remain aligned with analysis samples after conversion, splitting, and reduction in all validation fixtures.
- **SC-004**: At least three common modeling workflows are demonstrated: map-only prediction, map-plus-descriptor prediction, and reduced-map prediction using variance thresholding, PCA or truncated SVD, or atlas or label aggregation.
- **SC-005**: Users can complete the documented example workflow on a representative test collection without manually editing intermediate data.
- **SC-006**: Documentation examples convert successfully during the documentation build and produce rendered gallery outputs for the public workflow.
- **SC-007**: Missing descriptor or target data is never silently dropped or filled; validation fixtures report the affected sample identifiers and fields.
- **SC-008**: A representative collection with at least 1,000 studies can be converted and split in <=3 minutes with <=5 GB peak memory in the standard development environment.

## Assumptions

- Each analysis is the default machine-learning sample because the requested split rule specifically keeps analyses grouped by study.
- Study-level metadata may be repeated across that study's analyses, but grouped splitting prevents study-level leakage between training and testing partitions.
- "Scikit-learn-compatible" means the exported data can be consumed by common estimator workflows that expect aligned feature values, target values, and grouping labels.
- The feature will not train or evaluate predictive models itself beyond providing data structures, split helpers, and reduction workflows needed by researchers.
- Missing descriptor or outcome values fail clearly by default unless the researcher explicitly selects a handling strategy.
- Study grouping is mandatory for split helpers; conversion fails if neither explicit study identifiers nor unambiguous analysis identifiers can determine groups.
- Existing kernel and collection behavior from the latest release remains the compatibility baseline.
