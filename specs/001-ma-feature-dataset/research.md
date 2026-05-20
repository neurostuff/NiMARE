# Research: Masked Activation Feature Dataset

## Decision: Reuse NiMARE kernel transformers for map feature extraction

Use `KernelTransformer.transform(collection, return_type="sparse")` as the
primary map feature source. Existing ALE, KDA, and MKDA kernels already produce
analysis-by-masked-voxel sparse matrices for modeled activation maps. The
extractor will align the returned rows with sample IDs and retain an exclusion
report for analyses that cannot produce maps.

**Rationale**: This directly satisfies the requirement to use existing NiMARE
utilities before writing new helpers. It also preserves current kernel semantics,
sample-size handling, masker handling, and sparse output behavior.

**Alternatives considered**:

- Rebuild kernel map generation in the new module: rejected because it would
  duplicate scientific logic and increase regression risk.
- Generate niimgs and mask them later: rejected as the default because sparse
  output is already available and more memory efficient.

## Decision: Normalize inputs through existing NiMARE collection interfaces

Accept Studyset and Dataset inputs through existing NiMARE normalization and
shared table conventions. Use `collection.ids`, `collection.coordinates`,
`collection.metadata`, `collection.annotations` or `collection.annotations_df`,
`collection.texts`, `collection.masker`, and `collection.slice()` instead of
building independent collection adapters.

**Rationale**: Studyset is the preferred current collection type, while Dataset
remains a released compatibility path. NiMARE already exposes enough Dataset-like
tables to avoid a parallel data access layer.

**Alternatives considered**:

- Support only Studyset: rejected because the spec explicitly requires
  Studyset/Dataset inputs and released Dataset behavior remains compatibility
  relevant.
- Convert every input to Dataset: rejected because new workflows must prefer
  Studyset where possible.

## Decision: Represent the exported dataset as a NiMARE container plus a sklearn Bunch

Create an `MAFeatureDataset` container for NiMARE-specific provenance, masker,
groups, descriptor tables, target metadata, and diagnostics. Provide
`to_sklearn()` to return a scikit-learn-style dataset object with `data`,
`target`, `groups`, `sample_metadata`, and `feature_names`.
Expose conversion through `MAFeatureExtractor.transform(collection)` only; do
not expose `fit` or `fit_transform` on `MAFeatureExtractor` in the initial
public API.

**Rationale**: A bare tuple or matrix would be scikit-learn compatible but would
lose NiMARE-specific provenance needed for scientific reproducibility. A
container keeps NiMARE context while still exposing conventional scikit-learn
inputs.

**Alternatives considered**:

- Return only `(X, y, groups)`: rejected because it drops provenance and missing
  data diagnostics.
- Subclass a scikit-learn estimator or transformer immediately: rejected for the
  initial API because the primary user need is dataset creation and splitting,
  not model fitting.
- Add `fit`/`fit_transform` to `MAFeatureExtractor`: rejected because the
  extractor's main job is one-shot conversion from a NiMARE collection into an
  sklearn-compatible dataset, while trainable behavior belongs in explicit
  descriptor transformers, target transformers, reducers, and downstream
  sklearn models.

## Decision: Use scikit-learn group splitters for leakage-safe splits

Use `GroupShuffleSplit` for train/test splits and `GroupKFold` for grouped
cross-validation helpers. Expose study IDs as `groups` so users can pass them to
any scikit-learn estimator workflow that accepts groups.

**Rationale**: Group-aware splitting is a mature scikit-learn capability and
directly addresses the leakage requirement. It is simpler and safer than custom
split logic.

**Alternatives considered**:

- Implement custom grouped split logic: rejected because scikit-learn already
  provides the needed behavior.
- Split by analysis ID only: rejected because it violates the no-study-leakage
  requirement.

## Decision: Use collection-provided unique study and analysis IDs for MVP

Determine each sample's study group from collection-provided study IDs. The MVP
assumes input collections provide unique study IDs and unique analysis IDs, so
duplicate identifier diagnostics and analysis-ID-derived study groups are out of
scope for the initial implementation.

**Rationale**: The no-leakage split requirement depends on reliable study
groups. Treating unique identifiers as an input contract keeps the initial API
and tests focused on conversion, alignment, sparse map features, and grouped
splits.

**Alternatives considered**:

- Require caller-supplied groups for every conversion: rejected because NiMARE
  collections usually already contain study identity.
- Always parse IDs by convention: rejected because silently guessing groups can
  invalidate model evaluation.
- Add duplicate-ID reconciliation and analysis-ID fallback logic: rejected for
  the MVP because the initial workflow can assume unique collection identifiers.

## Decision: Descriptor extraction uses numeric fields by default

Metadata, annotations, and texts will be selected from existing collection tables
using current getters and table attributes. Numeric annotation/metadata values
can be added directly as descriptor features. Non-numeric descriptor fields,
including categorical metadata, annotations, titles, abstracts, and descriptions,
will be rejected by default unless the caller supplies an explicit transformer or
vectorizer that produces numeric features.

**Rationale**: NiMARE already standardizes these tables for Dataset and Studyset.
The new module should handle alignment and diagnostics, not invent a new
annotation or metadata schema or silently expand categorical/text features.

**Alternatives considered**:

- Require users to pre-build descriptor matrices: rejected because the feature
  specifically asks for convenience extraction.
- Pass non-numeric descriptors through in `data`: rejected because scikit-learn
  estimators generally require numeric features.
- Automatically encode categorical fields and vectorize text: rejected because
  default preprocessing would silently change feature dimensionality and model
  semantics.

## Decision: Target extraction supports scalar numeric/categorical targets by default

Target fields may be scalar numeric or scalar categorical values aligned to the
exported sample order. Categorical targets may remain strings or encoded values
for downstream scikit-learn estimators. Raw free-text targets and multi-label
targets are rejected by default unless the caller supplies an explicit target
transformer or label extractor.

**Rationale**: Scalar numeric and categorical targets cover standard regression
and classification workflows without forcing a lossy encoding policy. Free-text
and multi-label prediction require domain-specific extraction choices that
should be explicit.

**Alternatives considered**:

- Require targets to be numeric only: rejected because scikit-learn classifiers
  can handle categorical labels and many NiMARE annotations are categorical.
- Automatically vectorize text targets: rejected because this would produce
  high-dimensional or task-specific outcomes without explicit user intent.

## Decision: Reduction helpers are reusable scikit-learn pipelines

Provide convenience constructors for map feature reduction using the order of
preference requested by the user: existing NiMARE helpers for masking and sparse
map matrices, nilearn maskers for atlas/image-aware reductions when applicable,
and scikit-learn reducers for matrix reductions. Required initial workflows are
variance thresholding, sparse-compatible low-rank decomposition such as
`TruncatedSVD`, and atlas or label aggregation when a masker or labels image is
supplied. Unreduced voxelwise maps stay sparse; reducers may emit dense reduced
component or parcel matrices only after the voxel space has been reduced.

**Rationale**: scikit-learn pipelines naturally enforce fitting on training data
before transforming held-out data, which is required to avoid leakage. The
specified reducers cover simple filtering, low-rank dimensionality reduction, and
region-level aggregation without introducing a bespoke modeling framework.

**Alternatives considered**:

- Add bespoke reducer classes: rejected unless a needed reduction is not
  available through NiMARE, nilearn, or scikit-learn.
- Densify unreduced voxelwise map matrices before reduction: rejected because
  sparse MA maps are expected to be high-dimensional.
- Require dense PCA as an initial workflow: rejected because the initial API
  must avoid dense unreduced voxelwise intermediates; sparse-compatible
  low-rank reduction covers the large-scale use case.
- Provide only user-supplied transformers: rejected because the spec requires
  convenience workflows.

## Decision: Add a performance smoke target for 1,000-study conversion and split

Plan for a performance-sensitive test or benchmark that converts and grouped
splits a representative collection with at least 1,000 studies in <=3 minutes
with <=5 GB peak memory in the standard development environment.

**Rationale**: The feature can produce high-dimensional voxelwise matrices, so
the plan needs an explicit budget to protect interactive research workflows and
avoid accidental densification.

**Alternatives considered**:

- Keep performance qualitative: rejected because the clarified success
  criterion requires a measurable budget.
- Use a looser 1,000-analysis budget: rejected because the clarified requirement
  is study-based and stricter.

## Decision: Use existing NiMARE testing and Sphinx-Gallery infrastructure

Add targeted pytest coverage under `nimare/tests/test_ml.py`, add public API
docs in `docs/api.rst`, and add `.py` examples under `examples/05_machine_learning/`.
Example conversion will be validated through the Sphinx documentation build.

**Rationale**: This satisfies the project constitution and the user's request to
use existing testing, documentation, and notebook-generation infrastructure.

**Alternatives considered**:

- Add standalone notebooks: rejected because NiMARE uses Sphinx-Gallery `.py`
  sources that generate gallery pages and notebooks during docs builds.
- Create a new test directory: rejected because existing tests live under
  `nimare/tests`.
