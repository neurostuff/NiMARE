# Research: Masked Activation Feature Dataset

## Decision: Reuse NiMARE kernel transformers for map feature extraction

Use `KernelTransformer.transform(studyset, return_type="sparse")` as the
primary map feature source. Existing ALE, KDA, and MKDA kernels already produce
analysis-by-masked-voxel sparse matrices for modeled activation maps. The
extractor will align the returned rows with analysis IDs. Analyses with no
coordinates are handled by the `MAFeatureExtractor.missing_coordinates` option:
retain them as all-zero sparse map rows or drop them before row construction.

**Rationale**: This directly satisfies the requirement to use existing NiMARE
utilities before writing new helpers. It also preserves current kernel semantics,
sample-size handling, masker handling, and sparse output behavior.

**Alternatives considered**:

- Rebuild kernel map generation in the new module: rejected because it would
  duplicate scientific logic and increase regression risk.
- Generate niimgs and mask them later: rejected as the default because sparse
  output is already available and more memory efficient.

## Decision: Normalize inputs through existing NiMARE Studyset interfaces

Accept Studyset inputs through existing NiMARE normalization and shared table
conventions. Use `studyset.ids`, `studyset.coordinates`, `studyset.metadata`,
`studyset.annotations` or `studyset.annotations_df`, `studyset.texts`,
`studyset.masker`, and `studyset.slice()` instead of building independent
adapters.

**Rationale**: Studyset is the preferred current input type for this feature.
Using its existing table and masker conventions avoids a parallel data access
layer.

**Alternatives considered**:

- Support additional NiMARE input object types: rejected for the MVP to keep the
  public input contract focused on Studyset semantics.
- Convert Studysets through another internal container first: rejected because
  new workflows should avoid an unnecessary compatibility adapter.

## Decision: Represent the exported dataset as a NiMARE container plus a sklearn Bunch

Create an `MAFeatureDataset` container for NiMARE-specific provenance, masker,
groups, descriptor tables, target metadata, and diagnostics. Provide
`to_sklearn()` to return a scikit-learn-style dataset object with `data`,
`target`, `groups`, and `feature_names`.
Expose one-call export through `MAFeatureExtractor.to_sklearn(studyset, ...)`
for the common workflow, and keep `MAFeatureExtractor.transform(studyset)` for
advanced dataset-level workflows. Do not
expose `fit` or `fit_transform` on `MAFeatureExtractor` in the initial public
API.

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
  extractor's main job is one-shot conversion from a NiMARE Studyset into an
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

## Decision: Use Studyset-provided unique study and analysis IDs for MVP

Determine each analysis row's study group from Studyset-provided study IDs. The MVP
assumes input Studysets provide unique study IDs and unique analysis IDs, so
duplicate identifier diagnostics and analysis-ID-derived study groups are out of
scope for the initial implementation.

**Rationale**: The no-leakage split requirement depends on reliable study
groups. Treating unique identifiers as an input contract keeps the initial API
and tests focused on conversion, alignment, sparse map features, and grouped
splits.

**Alternatives considered**:

- Require caller-supplied groups for every conversion: rejected because NiMARE
  Studysets usually already contain study identity.
- Always parse IDs by convention: rejected because silently guessing groups can
  invalidate model evaluation.
- Add duplicate-ID reconciliation and analysis-ID fallback logic: rejected for
  the MVP because the initial workflow can assume unique Studyset identifiers.

## Decision: Descriptor extraction uses numeric fields by default

Metadata, annotations, and texts will be selected from existing Studyset tables
using current getters and table attributes. Numeric annotation/metadata values
can be added directly as descriptor features. Non-numeric descriptor fields,
including categorical metadata, annotations, titles, abstracts, and descriptions,
will be rejected by default unless the caller supplies an explicit transformer or
vectorizer that produces numeric features.

**Rationale**: NiMARE already standardizes these tables for Studyset.
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
exported analysis-row order. Categorical targets may remain strings or encoded values
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
splits a representative Studyset with at least 1,000 studies in <=3 minutes
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

## Decision: Prioritize sparsity over automatic feature-name tracking in `to_sklearn()`

`to_sklearn()` returns a `sklearn.utils.Bunch` with sparse `data`, separate
`feature_names`, `target`, and `groups`. Feature names are not automatically
paired with `data` through a pandas DataFrame or custom wrapper; users preserve
pairing through explicit DataFrame construction when needed or manage names
separately when working with sparse matrices directly.

**Rationale**: Voxelwise map features can have thousands to millions of columns.
Converting to a dense pandas DataFrame for automatic name tracking would
multiply memory costs by 10-100x, making the feature impractical for large-scale
neuroimaging datasets. Sparsity is essential for scalability. Users who need
automatic name tracking can construct a DataFrame at the point of use
(`pd.DataFrame(data, columns=feature_names)`), which is a well-established
sklearn pattern and explicit about the memory trade-off.

**Alternatives considered**:

- Return a dense DataFrame by default: rejected because it defeats the purpose
  of sparse map features and makes the tool unusable for typical neuroimaging
  dataset sizes (100k+ voxels).
- Return a custom wrapper that pairs sparse data with names: rejected because
  it adds complexity, is not a standard sklearn convention, and users would
  still need to convert to dense/DataFrame or manually track names through
  transformations.
- Return a tuple `(data, target, groups, feature_names)`: rejected because it is
  less discoverable than a Bunch and still requires users to manage pairing.

Users who want automatic name tracking through sklearn pipelines should use
`get_feature_names_out()` from sklearn transformers or work with DataFrames
explicitly for datasets small enough to densify.

## Decision: Orchestrate the full pipeline in `MAFeatureExtractor` for ergonomic public API

`MAFeatureExtractor` will handle the complete conversion, splitting, and
reduction workflow via convenience wrappers. `to_sklearn(studyset, ...)` is the
one-call public path. `transform(studyset)` returns
`(train_dataset, test_dataset)` for advanced workflows where users iterate over
multiple reducers.

Internally, the extractor decomposes into separate concerns: cached kernel-map
extraction, descriptor/target extraction, splitting, descriptor transformer
fitting, reducer fitting, and application. Train and test are represented as a
plain tuple of independent `MAFeatureDataset` containers.

**Rationale**: Users who need the "happy path" (convert, split, reduce) can do
it in one call without manual leakage management. Fitting descriptor transformers
and reducers on train data only is automatic and enforced. Advanced users can
still use `MAFeatureDataset.split()` and `apply_map_reducer()` for manual control.
This design separates public ergonomics (simple pipeline) from internal
architecture (modular components), enabling both ease-of-use and flexibility.

**Alternatives considered**:

- Keep `MAFeatureExtractor` focused on extraction only, require users to call
  `split()` and `apply_map_reducer()` manually: rejected because it makes the
  common case (convert → split → reduce) verbose, and it shifts leakage-safety
  responsibility to the user.
- Return only `(train, test)` without optional splitting: rejected because users
  who don't want a split should be able to opt out, and returning `None` for
  missing splits is explicit and idiomatic Python.
- Fit descriptor/target transformers on train only without exposing fit control:
  rejected because this is automatic and non-negotiable for correctness, making
  explicit control unnecessary.

## Decision: Cache MA map generation in `MAFeatureExtractor`

`MAFeatureExtractor` caches sparse MA map generation keyed by Studyset identity
and extractor settings that affect map construction (kernel, mask/space, and
missing-coordinate policy). Re-running with a different reducer reuses cached
maps and skips kernel recomputation.

**Rationale**: MA map generation is the most expensive repeated step in reducer
comparison workflows. Caching preserves scientific equivalence while reducing
iteration latency.

**Alternatives considered**:

- Recompute maps for every reducer: rejected because it wastes compute and slows
  experimentation.
- Cache reduced outputs too: rejected for MVP because reductions are cheap
  relative to map generation and vary frequently across experiments.

## Discussion: Tuple vs custom split-pair bundle

The advanced `transform(studyset)` API needs to return both train and test
containers. Two candidate representations were considered:

- Tuple: `(train_dataset, test_dataset)`.
- Custom class: e.g., `MAFeatureBundle(train=..., test=...)`.

**Decision for now**: use tuple returns.

**Why tuple is preferred right now**:

- Lower API surface area for MVP and fewer concepts for users to learn.
- Idiomatic Python for paired returns and easy unpacking.
- No maintenance burden for a new container type (`copy`, serialization,
  equality, docs, tests).
- Keeps focus on core scientific concerns (alignment, leakage safety,
  sparsity, reducer correctness).

**Trade-offs**:

- Tuples are less self-descriptive than named attributes.
- Bundle methods (e.g., `with_map_reducer`) are not available as object methods.

**Why custom bundle is deferred**:

- A bundle object may become valuable if future workflows need richer lifecycle
  management (metadata about splits, pipeline state snapshots, chained
  operations). For MVP, tuple simplicity and lower overhead are preferred.
