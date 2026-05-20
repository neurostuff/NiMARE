# Data Model: Masked Activation Feature Dataset

## Entity: Study Collection

Represents an input NiMARE Studyset or Dataset.

**Fields**

- `ids`: ordered analysis identifiers exposed by the collection.
- `coordinates`: coordinate table used by kernel transformers.
- `metadata`: tabular metadata fields.
- `annotations`: tabular annotation fields.
- `texts`: title, abstract, description, or other text fields.
- `masker`: fitted masker defining masked voxel space.
- `study_id`: explicit study grouping field when present in collection tables.

**Relationships**

- Contains many Analysis Samples.
- Provides source tables for Descriptor Feature Set and Prediction Target.
- Provides coordinates and masker for Masked Activation Feature Matrix.

**Validation Rules**

- Must expose a masker for map feature generation.
- Must contain at least one eligible analysis with valid coordinates.
- Must provide stable identifiers for sample alignment.
- Must provide unique study identifiers and unique analysis identifiers.

## Entity: Analysis Sample

One eligible analysis represented as one machine-learning sample.

**Fields**

- `sample_id`: full analysis identifier used for row alignment.
- `study_id`: grouping identifier used for leakage-safe splits.
- `analysis_id`: analysis identifier within the source study when available.
- `group_source`: collection-provided study ID.
- `row_index`: integer position in the feature dataset.
- `status`: included or excluded.
- `exclusion_reason`: reason an analysis was excluded, if applicable.

**Relationships**

- Belongs to one Study Group.
- Has one row in Masked Activation Feature Matrix when included.
- May have descriptor features and one prediction target.

**Validation Rules**

- Included sample IDs and analysis IDs are assumed unique in MVP inputs.
- Included samples must have exactly one study ID from the collection.
- Included samples must have exactly one row in every aligned output.
- Excluded analyses must be reported with a reason.

## Entity: Masked Activation Feature Matrix

Map-derived feature values for included samples.

**Fields**

- `matrix`: sample-by-masked-voxel sparse matrix for unreduced voxelwise
  features; reduced features may be dense only after an explicit reducer
  creates a lower-dimensional representation.
- `feature_names`: voxel or reduced feature identifiers.
- `masker`: masker used to define voxel ordering.
- `kernel_transformer`: description of the MA map generator.
- `map_ids`: sample IDs corresponding to matrix rows.

**Relationships**

- One row per included Analysis Sample.
- May be transformed by a Reduction Workflow.

**Validation Rules**

- Row count must equal included sample count.
- Row order must match `sample_id`.
- Masker and feature count must remain stable until an explicit reduction is
  applied.

## Entity: Descriptor Feature Set

Optional non-map features extracted from metadata, annotations, or texts.

**Fields**

- `source`: one of metadata, annotations, or texts.
- `field`: selected field name.
- `values`: sample-aligned values.
- `dtype`: resolved field type, such as numeric, categorical, text, or derived.
- `feature_names`: names added to the exported dataset.
- `missing_report`: sample IDs and fields with missing values.
- `transformer`: optional preprocessing transformer used for non-numeric fields.

**Relationships**

- Adds columns to the exported feature data.
- May share source fields with Prediction Target, but a field used as a target
  must not be duplicated as a feature by default.

**Validation Rules**

- Values must align exactly to included sample IDs.
- Missing values must be reported unless an explicit handling strategy is set.
- Numeric fields may be appended directly.
- Text and categorical fields must be rejected unless an explicit transformer or
  vectorizer converts them into numeric features.
- A descriptor transformer must fit only on training samples before transforming
  held-out samples.

## Entity: Prediction Target

Optional value the user wants a downstream model to predict.

**Fields**

- `source`: metadata, annotations, or texts.
- `field`: selected target field.
- `values`: one-dimensional sample-aligned target values exported as `y`.
- `target_type`: inferred or user-declared target type.
- `missing_report`: sample IDs missing target values.
- `target_transformer`: optional transformer or label extractor for free-text or
  multi-label targets.

**Relationships**

- Aligns one target value to each included Analysis Sample after target filtering.
- Shares Study Group labels for grouped splits.

**Validation Rules**

- Target length must match exported sample count.
- Scalar numeric targets preserve numeric values.
- Scalar categorical targets may remain strings or encoded values.
- Raw free-text and multi-label targets must be rejected unless an explicit
  target transformer or label extractor is supplied.
- Missing or constant targets must be diagnosed.
- Study-level targets may repeat across analyses, but grouped splitting must
  keep those repeated values in one partition.

## Entity: Study Group

Grouping key that keeps analyses from one study together.

**Fields**

- `study_id`: group label.
- `sample_ids`: sample IDs in the group.
- `n_samples`: number of analyses in the group.
- `source`: collection-provided study ID.

**Relationships**

- Used by Split Plan.
- Contains one or more Analysis Samples.

**Validation Rules**

- Every included sample must have exactly one study group.
- No study group may appear in more than one split partition.
- Missing study groups are outside the MVP input contract and must be reported
  before any split is returned.

## Entity: Split Plan

Train/test or cross-validation partition assignment.

**Fields**

- `train_indices`: sample indices assigned to training data.
- `test_indices`: sample indices assigned to held-out data.
- `groups`: study group labels used for splitting.
- `random_state`: reproducibility setting when applicable.
- `splitter`: splitter type and parameters.

**Relationships**

- Slices MA features, descriptor features, targets, sample metadata, and groups.
- Feeds Reduction Workflow fitting.

**Validation Rules**

- Train and test partitions must be disjoint.
- No study group can occur in both partitions.
- Requested split must fail clearly if there are too few study groups.

## Entity: Reduction Workflow

Reusable transformation for high-dimensional map features.

**Fields**

- `name`: reduction workflow name.
- `steps`: ordered transformer steps.
- `method`: one of variance thresholding, sparse-compatible low-rank reduction
  such as truncated SVD, or atlas/label aggregation for the initial public
  workflows.
- `fit_state`: unfitted or fitted.
- `input_feature_names`: map feature names before reduction.
- `output_feature_names`: reduced feature names after transformation.
- `region_definition`: optional masker or labels image for atlas/label
  aggregation.

**Relationships**

- Fits on training MA features.
- Transforms training and held-out MA features.
- Preserves Descriptor Feature Set and Prediction Target alignment.

**Validation Rules**

- Must not fit on held-out data.
- Transformed row order must match input row order.
- Output feature count must be deterministic for fixed inputs and parameters.
- Reducers must not densify unreduced voxelwise inputs as an intermediary.
- Reduced outputs may be dense when the reducer explicitly returns a
  lower-dimensional component or parcel matrix.
- Atlas/label aggregation requires a supplied masker or labels image compatible
  with the map feature space.

## State Transitions

1. Study Collection -> Analysis Samples + Masked Activation Feature Matrix.
2. Feature Matrix + optional descriptors -> MAFeatureDataset.
3. MAFeatureDataset + optional target -> supervised MAFeatureDataset.
4. MAFeatureDataset -> Split Plan -> train/test MAFeatureDataset slices.
5. Training slice -> fitted Reduction Workflow -> transformed train/test slices.
