# Implementation Plan: Masked Activation Feature Dataset

**Branch**: `001-ma-feature-dataset` | **Date**: 2026-05-01 | **Spec**: `specs/001-ma-feature-dataset/spec.md`
**Input**: Feature specification from `specs/001-ma-feature-dataset/spec.md`

**Note**: This plan was refreshed after the 2026-05-01 clarification pass.

## Summary

Add a new additive public `nimare.ml` module that converts NiMARE Studyset or
Dataset collections into scikit-learn-compatible masked activation feature
datasets. The module will use existing NiMARE kernel transformers to generate
sparse modeled activation map features, preserve sample provenance and study
groups, export `data`, `target`, `groups`, `sample_metadata`, and
`feature_names` for scikit-learn workflows, and provide leakage-safe grouped
splits plus convenience map-reduction workflows. Clarified defaults require
numeric descriptor features unless the caller supplies an explicit transformer,
scalar numeric or categorical targets unless the caller supplies explicit target
handling, explicit-or-unambiguous study grouping, and initial reducers for
variance thresholding, PCA/truncated SVD, and atlas/label aggregation.

## Technical Context

**Language/Version**: Python 3.10 through 3.14, matching `setup.cfg` metadata and CI support.  
**Primary Dependencies**: Existing NiMARE collection/kernel utilities first; nilearn `>=0.12.0,<0.14` for mask/image-aware operations; scikit-learn `>=1.0.0` for `Bunch`, group splitters, preprocessing, decomposition, and pipelines; numpy, pandas, scipy sparse, sparse, and joblib as already-declared runtime dependencies.  
**Data/Storage**: In-memory NiMARE Studyset and Dataset objects, NIMADS-derived tabular views, pandas metadata/annotation/text tables, nilearn maskers or labels images, and sample-by-feature sparse matrices. No new persistent storage format.  
**Testing**: Add targeted pytest coverage under `nimare/tests/test_ml.py` before implementation. First failing tests must cover conversion/provenance, grouped split leakage prevention, non-numeric descriptor rejection, scalar target export and unsupported target-shape rejection, missing-value diagnostics, reducer alignment, and the 1,000-study performance budget. Use existing markers, including `performance_smoke` for the scale check if needed.  
**Target Platform**: NiMARE-supported Python and OS matrix; no network-dependent tests.  
**Project Type**: Python scientific library public API plus Sphinx documentation examples.  
**Public API Impact**: New additive `nimare.ml` module with `MAFeatureDataset`, `MAFeatureExtractor`, field-selector handling, grouped split helpers, `to_sklearn()`, `apply_map_reducer()`, and map-reduction convenience constructors. Update `nimare/__init__.py`, `docs/api.rst`, and Numpydoc docstrings. No released public API is removed, renamed, or narrowed.  
**Compatibility Baseline**: `0.16.0` from `git describe --tags --abbrev=0`. Released Studyset, Dataset, kernel, metadata, annotation, and text access behavior must remain compatible.  
**Example Coverage**: Create Sphinx-Gallery examples `examples/05_machine_learning/01_plot_ma_feature_dataset.py` and `examples/05_machine_learning/02_plot_ma_feature_reduction.py`. Examples remain `.py` sources and are converted by the docs/Sphinx build.  
**Scientific Validation**: Validate that one sample represents one eligible analysis by default; all analyses from one study share a study group; map features are aligned to one mask/space; learned descriptor/reduction transforms fit only on training samples; held-out transformations do not use held-out targets; and all exclusions, missing values, and invalid targets are diagnosed.  
**Performance Goals**: A representative collection with at least 1,000 studies must convert and split in <=3 minutes with <=5 GB peak memory in the standard development environment.  
**Constraints**: Preserve latest-tag public behavior; prefer NiMARE utilities, then nilearn, then scikit-learn before new helpers; reject silent descriptor/target coercion; keep sparse map matrices sparse unless explicitly densified; tests and examples precede or accompany implementation; docs build must convert examples.  
**Scale/Scope**: New `nimare/ml.py`, tests under `nimare/tests`, API docs, two examples, and feature planning artifacts. Initial scope stops at dataset creation, target extraction, grouped splitting, and reusable reduction workflows; it does not train or evaluate predictive models.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Scientific validity**: PASS. The plan uses existing kernel transformers for
  modeled activation maps, records analysis-level sampling assumptions, enforces
  mask/space alignment, and requires validation fixtures for grouping,
  descriptors, targets, reductions, diagnostics, and performance.
- **TDD**: PASS. Targeted first failing tests are specified before
  implementation, including success paths, leakage prevention, error behavior,
  and performance-sensitive behavior.
- **API-first**: PASS. Public contracts are defined in
  `specs/001-ma-feature-dataset/contracts/public-api.md` and
  `specs/001-ma-feature-dataset/contracts/sklearn-compatibility.md` before
  implementation.
- **Example-driven**: PASS. Public API additions require
  `examples/05_machine_learning/01_plot_ma_feature_dataset.py` and
  `examples/05_machine_learning/02_plot_ma_feature_reduction.py`.
- **Public API stability**: PASS. Baseline tag is `0.16.0`; this feature is
  additive and does not alter released public methods/classes.
- **Simplicity**: PASS. The selected design is one public module plus simple
  containers/helpers, using existing NiMARE, nilearn, and scikit-learn utilities
  before local helpers.
- **Diagnostics**: PASS. Missing maps, unavailable fields, non-numeric
  descriptors, unsupported targets, incompatible masks/spaces, insufficient
  groups, and performance-budget failures are explicit test targets.
- **Environment**: PASS. Supported Python range, dependencies, docs extras,
  tests extras, and local verification commands are identified.
- **Documentation**: PASS. Public API docs and executable Sphinx-Gallery
  examples are required.

**Post-Design Re-check**: PASS. Phase 0 and Phase 1 artifacts preserve the
same additive API, test-first expectations, example coverage, compatibility
baseline, diagnostics, and reuse order. No constitution violations are known.

## Project Structure

### Documentation (this feature)

```text
specs/001-ma-feature-dataset/
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- contracts/
|   |-- public-api.md
|   `-- sklearn-compatibility.md
`-- tasks.md              # Created by /speckit-tasks, not /speckit-plan
```

### Source Code (repository root)

```text
nimare/
|-- __init__.py           # Export additive ml module
|-- ml.py                 # MAFeatureDataset, MAFeatureExtractor, reducers
`-- tests/
    `-- test_ml.py        # Contract, regression, diagnostics, performance tests

docs/
`-- api.rst               # Public API autosummary entry

examples/
`-- 05_machine_learning/
    |-- 01_plot_ma_feature_dataset.py
    `-- 02_plot_ma_feature_reduction.py
```

**Structure Decision**: Use a single additive `nimare/ml.py` module for the
initial feature. The API surface is focused enough that a package hierarchy
would add indirection without current need; if the unreleased implementation
grows too large before tagging, it can be split while preserving the public
`nimare.ml` import path.

## Phase 0: Research

Research decisions are captured in
`specs/001-ma-feature-dataset/research.md`. All clarification outcomes are
resolved there:

- Kernel transformers provide sparse modeled activation map features.
- Studyset/Dataset access uses existing collection interfaces.
- Study groups use explicit study IDs first, then only unambiguous analysis-ID
  derivation.
- Export uses a NiMARE container plus a scikit-learn `Bunch`.
- Splits use scikit-learn group splitters.
- Descriptor features are numeric by default; non-numeric descriptors require
  explicit transformers.
- Targets are scalar numeric/categorical by default; free-text and multi-label
  targets require explicit target handling.
- Reduction helpers cover variance thresholding, PCA/truncated SVD, and
  atlas/label aggregation.
- Performance is budgeted at <=3 minutes and <=5 GB peak memory for 1,000
  studies.

## Phase 1: Design & Contracts

Design artifacts are captured in:

- `specs/001-ma-feature-dataset/data-model.md`
- `specs/001-ma-feature-dataset/contracts/public-api.md`
- `specs/001-ma-feature-dataset/contracts/sklearn-compatibility.md`
- `specs/001-ma-feature-dataset/quickstart.md`

`AGENTS.md` already points to this active plan between the Spec Kit markers.
No `update-agent-context.sh` script exists in this checkout, so the agent
context was verified manually.

## Complexity Tracking

No constitution violations or required complexity exceptions.

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
