# Interface Design Review: `nimare.ml`

**Status**: Implemented on 2026-09-25. The contracts, `data-model.md`,
`quickstart.md` and `spec.md` were revised to match; §11 records where the
implementation departed from this proposal.
**Branch**: `001-ma-feature-dataset`
**Reviewed artifacts**: `spec.md`, `plan.md`, `research.md`, `data-model.md`,
`contracts/public-api.md`, `contracts/sklearn-compatibility.md`, `quickstart.md`,
`nimare/ml.py`, `nimare/tests/test_ml.py`, `examples/05_machine_learning/*`
**Baseline measured on**: the bundled `nback_vs_flanker_studyset_2026-07`
(906 analyses, 320 studies, 228,483 masked voxels), scikit-learn 1.9.1,
nilearn 0.13.1.

This document does three things: it states what the module is actually for,
it records where the contract and the implementation have drifted apart (and
which of the two is right), and it proposes one public interface, with the
options that were considered and rejected for each decision.

---

## 1. The use cases the interface has to serve

Five come from `spec.md`; the sixth and seventh come from the data the module
will actually be pointed at.

| # | Use case | What the interface owes it |
|---|----------|----------------------------|
| U1 | Turn a Studyset into an ML-ready matrix of modeled activation (MA) features | one call, sparse `X`, stable `ids`, per-row `study_id`, provenance |
| U2 | Evaluate without study leakage | `groups` usable by any sklearn splitter; a one-line holdout split |
| U3 | Add study-level/analysis-level descriptors as extra features | selection by field, numeric by default, loud about non-numeric and missing |
| U4 | Predict a metadata/annotation value | aligned `y`, scalar numeric or categorical, diagnosed |
| U5 | Reduce 228k voxel columns | reducers that stay sparse, are fit on train rows only, and compose |
| U6 | Compare several reducers on the same maps | MA generation paid for once, not once per reducer |
| U7 | Work against real NeuroStore metadata | ~80 metadata columns, mostly missing, mixed dtypes, list-valued `sample_sizes` |

U6 and U7 are the ones that decide the shape of the API. U6 says extraction and
evaluation are separate phases with different lifetimes. U7 says descriptor
handling cannot be a bespoke mini-framework; it has to hand off to sklearn.

### The fact the whole design rests on

Kernel transformation is **row-independent**: an analysis's MA map is a
deterministic function of that analysis's own foci (plus its own sample size,
for `ALEKernel`). It never sees `y` and never sees another row. Therefore
computing the full map matrix *before* splitting is not leakage, and the
container can be built eagerly and split afterwards.

The converse is just as important: everything that *does* learn across rows —
`TruncatedSVD`, `VarianceThreshold`, imputers, scalers, text vectorizers — must
be fit on training rows only. That is the line the interface has to draw, and
the only safe place to draw it is inside a `Pipeline`, where scikit-learn's
cross-validation machinery enforces it for us.

**Design rule.** NiMARE owns *extraction* (Studyset → aligned sparse `X`, `y`,
`groups`, provenance). scikit-learn owns *evaluation* (splitting, fitting,
reduction, scoring). The interface's job is to make the hand-off clean, and to
refuse to reimplement the far side of it.

---

## 2. Where the contract and the implementation stand today

Both have merit and they no longer agree. The implementation drifted toward
"export a `Bunch` plus an unfitted preprocessor, let sklearn drive"; the
contract still describes "a NiMARE container that splits and reduces itself".

| Element | `contracts/public-api.md` | `nimare/ml.py` today | Verdict |
|---|---|---|---|
| `MAFeatureExtractor.transform` | returns `(train_dataset, test_dataset)` | returns a single `Bunch` (+ `preprocessor`) | **implementation is closer**; neither is right (§5.1) |
| `MAFeatureExtractor.to_sklearn` | required, one-call export | **missing** | contract right |
| `test_size` / `random_state` on the extractor | required | **absent** | implementation right (§5.2) |
| `MAFeatureDataset.split()` | required | **missing** | contract right (§5.2) |
| `MAFeatureDataset.apply_map_reducer(reducer, fit=False)` | required | **missing**, replaced by `make_preprocessor()` | implementation right, but incomplete (§5.3) |
| `make_map_reducer("variance_threshold")` | required (FR-012) | **raises `NotImplementedError`** | contract right; one-line fix (§5.3) |
| `descriptor_transformers`, `target_transformer`, `memory`, `memory_level` | required | **all four raise `NotImplementedError` in `__init__`** | see §5.4, §5.5, §5.6 |
| `texts` as a descriptor/target source | required | **rejected**: `Unsupported field selector source: texts` | contract right (§5.4) |
| Non-numeric descriptor rejection with a clear message (FR-008) | required | raises numpy's `could not convert string to float: 'flanker'` | contract right (§5.4) |
| Missing-value reporting (FR-011, SC-007) | required | **absent** | contract right (§5.4) |
| `MAFeatureDataset` exported | `docs/api.rst` lists `ml.MAFeatureDataset` | **not in `ml.__all__`** | inconsistency (§5.1) |
| Sparse-only unreduced features (FR-015) | required | held on export, **not** through `make_preprocessor` | gap (§5.3) |
| 1,000-study budget (SC-008) | ≤3 min, ≤5 GB | no test, no benchmark | gap (§9) |

### Defects found while reading, with evidence

**D-1 (correctness, silent). Map rows are aligned to the Studyset by position,
and the positions are not guaranteed to match.**
`KernelTransformer.transform(..., return_type="sparse")` returns rows ordered by
`np.unique(coordinates["id"])` — sorted ids — and discards the `exp_ids` that
name them. `MAFeatureExtractor.transform` pairs those rows with `studyset.ids`
positionally. That holds only while the view is in canonical (sorted) order.
`Studyset.select_analyses` is public and documented to accept "an array of
positions", which produces a non-canonical view:

```text
ids:              ['study_1-t1' 'study_0-t1' 'study_0-t2']
kernel row order: ['study_0-t1' 'study_0-t2' 'study_1-t1']
```

Every row's maps, target, group and id are then mismatched, with no error. The
fix is two lines (reindex by `np.unique(studyset.coordinates["id"])`, assert the
row count) and it belongs in the interface as a stated guarantee, not an
accident of store layout.

**D-2. The documented quickstart descriptor does not work.**
`{"source": "metadata", "field": "sample_sizes"}` — the example in
`quickstart.md` — raises `ValueError: setting an array element with a sequence`,
because `sample_sizes` is list-valued in NIMADS. NiMARE already solves this:
`nimare.studyset.requirements.PerAnalysis("sample_sizes", reduce="mean")`
normalises per-group sizes *and* inherits study-level values.

**D-3. Non-numeric descriptors fail as numpy errors**, not as the FR-008
diagnostic naming the field.

**D-4. `feature_names` is eager and costs ~45% of the payload it names.**
For the bundled Studyset the MA matrix is 3,894,860 non-zeros (~32 MB of
`data` + `indices`); the `["feature_0", …]` list is **14.3 MB**, rebuilt on
every `transform()` and copied by every slice. It should be lazy.

**D-5. The map cache is unbounded and copies twice.** `self._map_cache` is a
plain dict keyed by a joblib hash, never evicted, and each hit/miss does
`.copy()` twice (transient 3× the matrix). Meanwhile `KernelTransformer`
already inherits nilearn's `CacheMixin`: `MKDAKernel(r=10, memory=…,
memory_level=1)` gives cross-process caching for free (measured 0.4 s warm vs
1.0 s cold, including from a fresh kernel instance in a fresh process).

**D-6. `make_preprocessor()` can silently densify.** `ColumnTransformer`
defaults to `sparse_threshold=0.3`: a sparse block whose density exceeds 30%
is returned dense. With 228k voxel columns that is the failure mode FR-015
exists to prevent. Verified: a 100%-dense-valued sparse block comes back as a
dense array under the default.

**D-7. "Masked activation" is not NiMARE's term.** The kernels produce
*modeled* activation (MA) maps (`nimare/meta/kernel.py` throughout). The spec
title, the module docstring and several docstrings say "masked". The class
names (`MAFeature*`) are fine; the prose should be corrected.

---

## 3. Constraints the design has to respect

- **Additive only.** Baseline 0.16.0; nothing released may change (FR-017).
- **Reuse order** (contract §"Utility Preference Order"): NiMARE → nilearn →
  scikit-learn → new code. §5.4 and §5.6 below are where this bites hardest.
- **`setup.cfg` floor is `scikit-learn>=1.0.0`** (CI min-version job pins
  1.4.0). Nothing proposed here needs >1.0 behaviour; `set_output`/metadata
  routing are deliberately avoided.
- **Sparse in, sparse through** until an explicit reducer says otherwise.
- **`MAFeatureExtractor` is not a scikit-learn estimator** and must not grow
  `fit`/`fit_transform` (FR-014).

---

## 4. The architectural choice: where does the sklearn boundary sit?

Four candidates were considered for the primary hand-off.

### Option A — NiMARE-owned lifecycle (what the contract describes)

`extractor.transform(studyset) -> (train_ds, test_ds)`; datasets split
themselves and reduce themselves via `apply_map_reducer(reducer, fit=False)`.

- ✅ One mental model, no sklearn knowledge required for the simple case.
- ❌ Reimplements `Pipeline` badly. `fit=False` is a *boolean that decides
  whether you leak*; every manual `fit_transform` on train / `transform` on
  test pair is a place to get it wrong, and it cannot be used inside
  `cross_val_score` at all — the split has already happened.
- ❌ `(dataset, None)` tuples in the common no-split case.
- ❌ Descriptor transformers must be fit at extraction time, i.e. on all rows,
  which contradicts FR-013 unless the container re-implements fold-aware
  fitting too.

### Option B — sklearn-owned lifecycle (what the code drifted to)

`extractor.transform(studyset) -> Bunch(data, target, groups, …, preprocessor)`;
the user drops `preprocessor` into a `Pipeline` and hands `groups` to a group
splitter.

- ✅ Leakage safety comes from `Pipeline` + `GroupKFold`, which are correct by
  construction and already understood by the audience.
- ✅ Cheap: `make_preprocessor` is ~10 lines around `ColumnTransformer`.
- ✅ Matches U6: one extraction, many reducers.
- ❌ `transform()` returning a `Bunch` conflates NiMARE's container with the
  sklearn export (and strands `MAFeatureDataset`, which the current code does
  not even export).
- ❌ No holdout convenience at all — U2's "just give me a train/test split"
  needs six lines of `GroupShuffleSplit` boilerplate.
- ❌ `preprocessor` attached to the data bundle is an odd place for a *model*
  object to live.

### Option C — id-indexed pipeline source (`X = ids`)

A `MAFeatures(studyset, kernel)` sklearn transformer whose `transform(ids)`
looks rows up in a cached matrix; `X` passed to `cross_val_score` is the id
array (or a descriptor DataFrame indexed by id).

- ✅ Strictly leak-free, including text vectorizers and imputers, because
  *everything* is inside the pipeline. The precedent is good: nilearn's
  `Decoder` takes a list of images as `X`.
- ✅ Composes with `ColumnTransformer`/`FeatureUnion` for U7's messy frames.
- ❌ Unfamiliar: `X` no longer looks like data.
- ❌ Doesn't satisfy U1 on its own — researchers still want the matrix in hand.
- ❌ More surface than the MVP justifies.

### Option D — B for the mainline, C as the documented extension

Recommended. Extraction returns a NiMARE container; the container exports an
sklearn bundle and a preprocessor; a thin holdout helper covers U2; and the
id-indexed transformer (C) is deferred to v1.1 as the answer for text and
categorical descriptors, with v1 internals keyed by analysis id so it drops in
without a rewrite.

**Recommendation: Option D.** It keeps the implementation's best insight (don't
reimplement `Pipeline`), restores what the contract was right about (a named
container, a one-call export, a grouped split), and gives non-numeric
descriptors a real answer instead of a `NotImplementedError`.

---

## 5. Decisions

### 5.1 `transform()` returns the container; `to_sklearn()` returns the bundle

- **Options**: (a) `transform → (train, test)` tuple [contract]; (b)
  `transform → Bunch` [current code]; (c) `transform → MAFeatureDataset`,
  `to_sklearn → Bunch`; (d) one method with `return_type=` in the style of
  `KernelTransformer.transform`.
- **Recommendation: (c).** One object per method, no `None`-in-a-tuple, and the
  container/bundle split is exactly the one `research.md` argued for. (d) is
  tempting for house consistency, but a string switch that changes the return
  *type* is worse for discoverability and type checking than two named methods;
  `KernelTransformer` needs it only because it has four output shapes.
- `MAFeatureDataset` goes into `ml.__all__` (it is already in `docs/api.rst`).
- `to_sklearn(studyset, return_X_y=False)` mirrors the scikit-learn dataset
  loader convention (`Bunch` by default, `(X, y)` on request).

### 5.2 Splitting lives on the dataset, not the extractor

- **Options**: (a) `test_size`/`random_state` on the extractor [contract];
  (b) `dataset.split()`; (c) neither — export `groups` and document
  `GroupShuffleSplit`.
- **Recommendation: (b) + (c).** Split configuration is an *evaluation*
  parameter; putting it on the extractor's constructor means the same extractor
  cannot be reused for CV and for a holdout, and it forces the `(ds, None)`
  tuple on everyone who doesn't want a split. `split()` is ~15 lines over
  `GroupShuffleSplit` and satisfies FR-005/FR-006 with one obvious call.
- `split()` must pre-validate the group count: with one study, sklearn reports
  `With n_samples=1, test_size=0.25 …`, which names neither studies nor the real
  problem. NiMARE raises first, naming the study count and the request.
- Document explicitly that `test_size` is a fraction **of studies**, not of
  analyses — `GroupShuffleSplit` splits groups, so analysis counts only
  approximate the ratio.
- Drop the contract's `cv=` parameter on `split()`: it has no coherent return
  type. `GroupKFold`/`StratifiedGroupKFold` over `groups` is the answer, and
  belongs in the docs and the gallery example.

### 5.3 Reduction: a preprocessor for pipelines, fitted-state for manual work

- **Options**: (a) `apply_map_reducer(reducer, fit=False)` [contract];
  (b) `make_preprocessor()` only [current code]; (c) `make_preprocessor()` plus
  `fit_transform_maps` / `transform_maps`.
- **Recommendation: (c).** Drop the `fit=` boolean — the fitted-ness belongs to
  the reducer, not to a flag on the data. `fit_transform_maps(reducer)` fits and
  returns a reduced dataset; `transform_maps(reducer)` requires an already
  fitted reducer and raises `NotFittedError` otherwise. Leakage becomes the
  *error* case rather than a keyword argument.
- `make_preprocessor` gains `sparse_threshold=1.0` (fixes D-6) and a
  `descriptor_transformer=` slot (so an imputer or scaler can be applied to the
  descriptor block per fold — the common need under U7, given how much
  NeuroStore metadata is missing).
- `make_map_reducer` grows `variance_threshold` (`VarianceThreshold` is
  sparse-preserving — verified) so FR-012 is actually met.
- The atlas reducer becomes public (`AtlasAggregator`, from
  `_NilearnMaskerReducer`) with `get_feature_names_out()` returning region
  labels, and a sparse fast path: for `NiftiLabelsMasker` with `strategy` in
  `{"mean", "sum"}`, once the labels image is resampled into the map masker's
  voxel space, the aggregation is an exact sparse matmul against a
  voxel-by-region indicator, with no `unmask` round-trip at all. `NiftiMapsMasker`
  (least-squares) keeps the batched dense path.

### 5.4 Field selection reuses NiMARE's own vocabulary

`{"source": ..., "field": ...}` dicts are a second vocabulary for something
NiMARE already has: `_required_inputs`' `(kind, field)` pairs, translated by
`nimare.studyset.inputs.requirement_for` into `Coordinates` / `PerAnalysis` /
`Labels` / `Texts` requirements. Every estimator in the library describes itself
that way.

- **Options**: (a) dict selectors [contract]; (b) `(source, field)` tuples and
  bare strings, resolved through `requirements`; (c) accept a caller-built
  DataFrame.
- **Recommendation: (b)**, with `"field"` alone resolved against metadata →
  annotations → texts and an ambiguity error naming the collision. Dicts stay
  accepted for one release as an undocumented alias (the surface is unreleased,
  so this is courtesy, not compatibility).
- Numeric metadata resolves via `PerAnalysis(field, reduce="mean")`, which fixes
  D-2 and gets study-level inheritance for free.
- Non-numeric fields raise a NiMARE message naming the source, the field, the
  observed dtype and the two supported routes (§5.5) — fixing D-3.
- `texts` becomes a valid source for selection, and always non-numeric.
- **Missing values** get one knob, `missing_values="raise" | "drop" | "keep"`,
  default `"raise"` (SC-007: never silently dropped or filled). The error names
  the affected analysis ids (capped, with a count) and the fields. `"drop"`
  records dropped ids in provenance; `"keep"` leaves NaN and records the count,
  for users who will impute inside their pipeline.
- `missing_coordinates="drop" | "include"` stays exactly as specified, and is
  implemented over `nimare.studyset.requirements.Coordinates` — `Studyset.resolve(...,
  drop_invalid=True)` for `"drop"`, its `validity(view)` mask for `"include"` —
  rather than hand-rolled coordinate-id set arithmetic.

### 5.5 Non-numeric descriptors: reject in v1, with a real answer attached

- **Options**: (a) `descriptor_transformers` applied eagerly at extraction
  [contract]; (b) reject, and document two routes; (c) carry raw descriptors in
  a heterogeneous `X`.
- **Recommendation: (b) for v1.** (a) is the one place the contract is
  internally inconsistent: fitting a `TfidfVectorizer` or a `OneHotEncoder` at
  extraction time fits it on *all* rows, which is precisely what FR-013 and the
  sklearn-compatibility contract forbid; the contract's "fit descriptor
  transformers on training data, then apply to train and test separately"
  requires the extractor to own fold-aware fitting — i.e. to reimplement
  `Pipeline` (Option A's failure). (c) is Option C, deferred.
- v1 therefore ships: numeric descriptors in `X`; a clear rejection for the
  rest; `dataset.descriptors` (a small raw DataFrame indexed by id, which we
  already have in hand) so route 1 works today:
  1. encode it yourself and hand the numeric column back in, or
  2. use the documented pipeline recipe — `ColumnTransformer` over
     `dataset.descriptors` union the map block — which becomes a supported
     class (`MAFeatures`) in v1.1.
- **Runner-up worth a decision**: allow `descriptor_transformers` eagerly with a
  documented leakage note (the way NLP practice tolerates a shared
  vocabulary). It ships U3's text case sooner at the cost of contradicting
  SC-007's spirit. I recommend against it, but it is the call to make if text
  descriptors are wanted before v1.1.
- `target_transformer` **stays** in v1, restricted to a callable or a stateless
  transformer applied to the raw target column (FR-009's "label extractor"): it
  is row-wise, so it cannot leak. Leakage-sensitive target transforms are
  pointed at `TransformedTargetRegressor`.

### 5.6 Caching delegates to NiMARE's existing mechanism

- **Options**: (a) the current unbounded dict keyed by a joblib hash of the
  frames; (b) `memory`/`memory_level` through `CacheMixin`, which `NiMAREBase`
  already provides and `KernelTransformer` already uses; (c) nothing.
- **Recommendation: (b)**, plus a **one-entry** in-process memo of the last
  `(studyset identity, kernel params)` result for U6's reducer loop. This
  honours the two constructor parameters that currently raise
  `NotImplementedError`, gives cross-process reuse (measured 0.4 s warm), and
  bounds memory at one matrix instead of one per distinct configuration.
- The double `.copy()` goes: the memo hands back the matrix it holds, and
  `MAFeatureDataset` treats map features as read-only (documented; `copy()`
  exists for callers who need to mutate).

### 5.7 Smaller decisions

| Decision | Recommendation | Why |
|---|---|---|
| `feature_names` | lazy cached property; voxels named `voxel_<i>` | fixes D-4 (14.3 MB built per call); `feature_0` is ambiguous once descriptors are also features |
| `_masker` | public read-only `masker` property | reducers, examples and tests all need it; a leading underscore that everything touches is not private |
| Row selection | `select_analyses(mask_or_positions)` | mirrors `Studyset.select_analyses` |
| Provenance | plain JSON-serialisable dict with a documented schema | FR-003; a dataclass buys little and complicates `save`/`load` |
| Alignment | reindex map rows by id and assert the count | fixes D-1; also catches duplicate ids, which `np.unique` would otherwise collapse silently |
| Naming | "modeled activation" in all prose | fixes D-7; class names unchanged |

---

## 6. Proposed public surface

```python
# nimare/ml.py
__all__ = ["MAFeatureDataset", "MAFeatureExtractor", "AtlasAggregator", "make_map_reducer"]


class MAFeatureExtractor(NiMAREBase):
    """Convert a Studyset into modeled-activation feature data.

    Not a scikit-learn estimator: it has no ``fit``/``fit_transform`` (FR-014).
    """

    def __init__(
        self,
        kernel_transformer,                 # instance or class; no scientific default
        *,
        descriptor_fields=None,             # str | (source, field) | sequence of either
        target_field=None,                  # str | (source, field)
        target_transformer=None,            # callable or stateless transformer
        missing_coordinates="drop",         # {"drop", "include"}
        missing_values="raise",             # {"raise", "drop", "keep"}
        cache_maps=True,                    # one-entry in-process memo
        memory=Memory(location=None, verbose=0),
        memory_level=0,
    ): ...

    def transform(self, studyset) -> "MAFeatureDataset": ...

    def to_sklearn(self, studyset, *, return_X_y=False):
        """Bunch(data, target, groups, ids, feature_names, provenance,
        map_columns, descriptor_columns) -- or ``(X, y)`` when return_X_y."""


class MAFeatureDataset(NiMAREBase):
    # --- data (all row-aligned, in one fixed row order) -----------------
    ids: np.ndarray            # "<study_id>-<analysis_id>"
    study_ids: np.ndarray      # sklearn ``groups``
    features                   # sparse (n_rows, n_map + n_descriptor)
    target                     # 1-D or None
    provenance: dict
    descriptors                # DataFrame of raw selected values, indexed by ids
    masker                     # voxel order for the map block

    @property
    def feature_names(self) -> list[str]: ...      # lazy
    @property
    def map_columns(self) -> slice: ...
    @property
    def descriptor_columns(self) -> slice: ...

    # --- export ---------------------------------------------------------
    def to_sklearn(self, *, return_X_y=False): ...

    # --- evaluation helpers --------------------------------------------
    def split(self, test_size=0.25, random_state=None):
        """Grouped holdout by study. ``test_size`` is a fraction of *studies*.
        Raises before returning anything if the study count cannot serve it."""

    def make_preprocessor(
        self,
        map_reducer="truncated_svd",
        descriptor_transformer="passthrough",
        **reducer_kwargs,
    ):
        """Unfitted ColumnTransformer: reduce map columns, handle descriptor
        columns, ``sparse_threshold=1.0``. Put it in a Pipeline."""

    # --- manual reduction (fitted-state, not a boolean) -----------------
    def fit_transform_maps(self, reducer) -> "MAFeatureDataset": ...
    def transform_maps(self, fitted_reducer) -> "MAFeatureDataset": ...

    # --- plumbing -------------------------------------------------------
    def select_analyses(self, mask_or_positions) -> "MAFeatureDataset": ...
    def copy(self) -> "MAFeatureDataset": ...
    def __len__(self) -> int: ...


def make_map_reducer(method, masker=None, **kwargs):
    """{"variance_threshold", "truncated_svd", "atlas_aggregation"}."""


class AtlasAggregator(TransformerMixin, BaseEstimator):
    """Aggregate masked voxel features into atlas regions.

    Sparse matmul fast path for NiftiLabelsMasker (mean/sum); batched
    unmask + nilearn transform otherwise. Implements get_feature_names_out.
    """
```

### Guarantees the interface states (and tests)

1. Row order is fixed at construction and identical across `features`,
   `target`, `ids`, `study_ids`, `descriptors`, and every derived dataset.
2. Map rows are matched to analysis ids **by id**, never by position (D-1).
3. Unreduced voxel features are sparse everywhere: in `features`, in the
   exported `data`, and out of `make_preprocessor()` (D-6).
4. `groups` is `study_ids`; no study appears in both sides of any split.
5. Nothing is silently imputed, coerced, or dropped: missing descriptors and
   targets raise by default and name the analyses involved.

---

## 7. What the workflows look like

**U1 + U4 + U5 + U2 (gallery example 01):**

```python
extractor = MAFeatureExtractor(
    kernel_transformer=MKDAKernel(r=10),
    target_field=("metadata", "comparison_task"),
)
data = extractor.transform(studyset)          # MAFeatureDataset
bunch = data.to_sklearn()                     # or extractor.to_sklearn(studyset)

pipeline = make_pipeline(
    data.make_preprocessor("truncated_svd", n_components=50, random_state=13),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)
scores = cross_val_score(
    pipeline, bunch.data, bunch.target, cv=GroupKFold(5), groups=bunch.groups,
)
```

**U2, simple holdout:**

```python
train, test = data.split(test_size=0.25, random_state=13)
assert set(train.study_ids).isdisjoint(test.study_ids)
```

**U6 (gallery example 02), and manual reduction without a pipeline:**

```python
data = extractor.transform(studyset)          # maps computed once
for name, reducer in {
    "Truncated SVD": make_map_reducer("truncated_svd", n_components=64, random_state=13),
    "DiFuMo atlas":  make_map_reducer("atlas_aggregation", masker=data.masker,
                                      atlas_masker=difumo_masker),
}.items():
    train_r = train.fit_transform_maps(reducer)   # fits on train rows only
    test_r = test.transform_maps(reducer)         # NotFittedError if you skip the line above
```

**U3, today (numeric) and the v1.1 route (text/categorical):**

```python
extractor = MAFeatureExtractor(
    kernel_transformer=MKDAKernel(r=10),
    descriptor_fields=[("metadata", "sample_sizes"), "n_subjects"],
    missing_values="keep",                    # imputed per fold, below
)
data = extractor.transform(studyset)
pipeline = make_pipeline(
    data.make_preprocessor("truncated_svd", n_components=50,
                           descriptor_transformer=SimpleImputer()),
    Ridge(),
)
# v1.1: raw text/categorical stay out of X and are encoded per fold
# ColumnTransformer over data.descriptors, union MAFeatures(studyset, kernel)
```

---

## 8. Amendments this requires to the contract

`contracts/public-api.md`, `contracts/sklearn-compatibility.md`, `data-model.md`
and `quickstart.md` need these edits; `spec.md`'s functional requirements are
all still met, and three of them (FR-008, FR-011, FR-012) become *true* for the
first time.

1. `MAFeatureExtractor.transform(studyset)` returns **one** `MAFeatureDataset`,
   not `(train, test)`. [FR-014's substance — no `fit`/`fit_transform`, a
   one-call `to_sklearn` — is unchanged.]
2. `test_size` and `random_state` move off the extractor onto
   `MAFeatureDataset.split()`; `split()` loses `cv=`.
3. `apply_map_reducer(reducer, fit=False)` → `fit_transform_maps(reducer)` /
   `transform_maps(fitted_reducer)`.
4. `make_preprocessor(...)` is added to the container's required methods, with
   `sparse_threshold=1.0` named as the FR-015 guarantee.
5. Field selectors become `str` / `(source, field)`; the dict form is an alias.
   `texts` is a valid source.
6. `missing_values="raise"|"drop"|"keep"` is added (SC-007).
7. `descriptor_transformers` is deferred to v1.1 and documented as the
   pipeline route; `MAFeatures` is named as the v1.1 extension point.
8. `memory`/`memory_level` are wired to `CacheMixin`; `cache_maps` is
   redefined as a one-entry memo.
9. `MAFeatureDataset` is exported from `ml.__all__`; `masker` and
   `descriptors` become public.
10. Prose says "modeled activation" throughout (D-7).

---

## 9. Implementation plan

Ordered so each step is independently reviewable, and so the two defects that
can corrupt results land first.

| Step | Work | Tests |
|---|---|---|
| 1 | Align map rows by id; reject duplicate/missing ids | regression test built on `select_analyses([2, 0, 1])`, asserting each row's map matches its own foci |
| 2 | Lazy `feature_names`; public `masker`; `MAFeatureDataset` in `__all__` | name/lazy assertions; memory assertion on a synthetic wide mask |
| 3 | `transform → MAFeatureDataset`, `to_sklearn(studyset, return_X_y=)` on both classes | export-contract assertions (existing helper covers most) |
| 4 | `split()` with group-count pre-validation | disjointness, determinism, and the clear too-few-studies error |
| 5 | Selectors via `requirements`; `texts`; FR-008 message; `missing_values` | `sample_sizes`, categorical rejection, missing reporting (names the ids) |
| 6 | `make_preprocessor` (`sparse_threshold=1.0`, `descriptor_transformer=`); `variance_threshold`; public `AtlasAggregator` + labels fast path + `get_feature_names_out` | sparsity preserved through CT; labels fast path == nilearn's output; pipeline + `GroupKFold` end-to-end |
| 7 | `fit_transform_maps` / `transform_maps` | `NotFittedError` on the unfitted path; alignment preserved |
| 8 | Caching via `CacheMixin` + one-entry memo | kernel called once across reducers; invalidation on changed coordinates/params |
| 9 | Rewrite both gallery examples; `docs/api.rst`; provenance schema | docs build converts both examples |
| 10 | SC-008 budget: 1,000-study synthetic Studyset, `performance_smoke` marker | ≤3 min, ≤5 GB (current baseline: 906 analyses in 6.2 s, 110 MB peak) |

Steps 1–2 are strictly bug fixes against the current code and could land ahead
of the interface decision.

**All ten steps landed together.** The 1,000-study budget is met with room to
spare: conversion and a grouped split take **0.6 s** and **0.58 GB** peak RSS
against budgets of 3 minutes and 5 GB, and gallery example 01 reproduced its
previous cross-validation accuracy (0.625 ± 0.019) through the new API. After
merging the branch's upstream, which rounds millimetre coordinates to the
nearest voxel (#1142), the same example reads 0.623 ± 0.030: a different map,
not a different pipeline.

---

## 10. Deferred, and explicitly not in v1

- `MAFeatures` (Option C) and per-fold descriptor encoding — v1.1.
- Studysets too large to materialise (NeuroStore's 115k analyses ≈ 6 GB of MKDA
  non-zeros): out of scope, but the id-keyed internals from step 1 are what a
  chunked or memory-mapped backend would need, so nothing here forecloses it.
- Duplicate study/analysis ids: the MVP input contract assumes uniqueness, but
  step 1's assertion turns a silent misalignment into a clear error.
- The module trains no models, and will not (spec Assumptions).

---

## Appendix: how the measurements were taken

All against `nimare/resources/nback_vs_flanker_studyset_2026-07` (906 analyses,
320 studies, 228,483 voxels), Python 3.12, scikit-learn 1.9.1, nilearn 0.13.1.

| Measurement | Value |
|---|---|
| `MKDAKernel(r=10)` + `transform`, cold | 6.2 s, 110 MB `tracemalloc` peak |
| MA matrix | 3,894,860 nnz, 16 MB values (~32 MB with indices) |
| `feature_names` list | 14.3 MB |
| Extractor memo hit | 0.3 s |
| `KernelTransformer` joblib memory, warm, fresh process | 0.4 s |
| Gallery example 01, end to end | 21.9 s, CV accuracy 0.625 ± 0.019 |
| `VarianceThreshold` on CSR | stays sparse |
| `ColumnTransformer` default `sparse_threshold=0.3` | densifies a dense-valued sparse block |
| `GroupShuffleSplit` with 1 group | `ValueError: With n_samples=1, …` (does not mention studies) |
| Kernel row order vs `select_analyses([2, 0, 1])` | mismatched (D-1) |

---

## 11. Where the implementation departed from this proposal

Written after the fact, so the proposal and the code can be read together.

1. **The atlas reducer keeps nilearn's slow path.** §5.3 proposed a sparse
   matmul fast path for `NiftiLabelsMasker` with mean/sum. It was dropped:
   no requirement asks for it, and reproducing nilearn's resampling, background
   handling, region ordering and strategy semantics exactly is how a
   performance optimisation turns into a silent numerical divergence from the
   library the tests compare against. `AtlasAggregator` batches rows back
   through nilearn and `batch_size` is the documented memory/speed dial
   (gallery example 02 raises it to 64, which pays nilearn's per-call
   least-squares setup 6× less often than the default 10).
2. **`MAFeatureDataset` takes the blocks, not the combined matrix.** The
   constructor takes `map_features` and `descriptor_features` and derives
   `features` and `feature_names` from them on first access. Passing a
   prebuilt `features` *and* its parts, as the old code did, makes
   "the matrix disagrees with its blocks" a state the container has to
   validate against; deriving it makes that state unrepresentable.
3. **`map_features` and `descriptor_features` are public too**, not just
   `masker`. The reduction methods, the examples and the tests all read them;
   an underscore that everything touches is not privacy.
4. **An unknown reducer name raises `ValueError`, not `NotImplementedError`.**
   With all three workflows implemented, an unrecognised name is a typo, and
   the message names the alternatives.
5. **Selector forms follow Python's own convention**: a tuple is one
   `(source, field)` selector, a list holds several. A bare field name is
   resolved across metadata, annotations and texts, and an ambiguous one raises
   naming the sources it matched. Mappings are still accepted.
6. **Caching hands a location to a copy of the kernel transformer** rather than
   adding a second cache to the extractor. `KernelTransformer` already inherits
   nilearn's `CacheMixin`, so `memory`/`memory_level` reuse the cache NiMARE
   already has, across processes; `cache_maps` is the one-entry in-process memo
   for the reducer-comparison loop. The caller's kernel is never mutated.
7. **A constant target raises** rather than warning: there is nothing to
   predict, and finding out after a cross-validation run is worse than finding
   out now.
8. **Extra guardrails**, each with a test: an empty Studyset, a field selected
   twice, a reducer that changes the row count, a boolean mask of the wrong
   length, and a target transformer that returns something other than one value
   per analysis.
9. **`descriptors` travels in the exported Bunch**, so the pipeline route for
   categorical and text fields is reachable from the bundle alone.
10. **Gallery example 02 evaluates on one grouped holdout**, as its predecessor
    did, instead of three splits: nilearn's maps-masker least squares makes
    each additional split expensive in a documentation build.

---

## 12. Follow-up: generalising the two reducer slots (2026-09-25)

The reducers shipped as three named workflows plus an `AtlasAggregator` that
required a caller-built nilearn masker, which made DiFuMo and truncated SVD
read as *the* two choices rather than as two examples. Both slots are now open:

- **Any scikit-learn transformer.** `make_map_reducer` takes a workflow name, a
  transformer, or a transformer class it builds from `**kwargs`;
  `make_preprocessor` passes whatever it is given through the same resolver.
  The named workflows stay, because a name is the shortest way to say the
  common thing, and an unknown name now says what the alternatives to a name
  are. Sparse-readability is documented rather than guarded: scikit-learn's own
  message for `PCA` on sparse input names the fix (`TruncatedSVD`), and which
  reducer suits a question is the researcher's call, not the module's.
- **Any atlas nilearn can load.** `AtlasAggregator(atlas=...)` accepts a fetched
  atlas `Bunch`, a 3D or 4D image, a path, the name of a `fetch_atlas_*`
  function with its `atlas_kwargs`, or a masker the caller configured. It picks
  `NiftiMapsMasker` for 4D and `NiftiLabelsMasker` for 3D, and reads the
  atlas's `labels` -- a list, or a frame like DiFuMo's -- for the reduced
  feature names, dropping a leading `Background` entry when the region count
  says to. A `NiftiMasker` as the atlas is refused: it extracts voxels, not
  regions.

The dispatch is on what nilearn's fetchers actually return (`maps` plus
`labels`, `maps` being a path more often than an image), so it works for the
fetchers as they are rather than for a normalised form none of them produce.
Strings mean a workflow name in the reducer slot and a file or fetcher in the
`atlas` slot, which keeps one overload per slot instead of one string that
could be three things.

Generalising the atlas slot surfaced one cross-version difference worth
recording. Regions that fall outside the mask are kept by nilearn 0.12 and
dropped by 0.13, and for a probabilistic atlas 0.13 drops them from the
transform output without dropping them from `maps_img_` or `n_elements_`, so no
fitted attribute predicts the width. `AtlasAggregator.get_feature_names_out()`
counts the columns instead -- transforming a single all-zero row when nothing
has been transformed yet -- and falls back to positional names when the atlas's
labels cannot be matched to the regions that survived. Guessing would have
meant labelling a column with a region that is not in it. The tests run against
both nilearn 0.12.0 and 0.13.1.

---

## 13. Follow-up: one function and one container (2026-09-25)

The names above (`MAFeatureExtractor`, `MAFeatureDataset`) were reviewed once
more and replaced, for two reasons that are worth separating.

`MAFeatureDataset` collided with `nimare.dataset.Dataset`, the legacy studyset
container users already know, and `MA` was jargon no other public class in the
library uses as a prefix. The container is now **`FeatureSet`**.

The extractor went further than a rename. It was a configure-then-call-once
object, and the only things a class bought were applying identical settings to
several Studysets and holding the in-process map memo. Conversion became a call rather than a
configure-then-call pair, first as a module-level `extract_features` function
and then, on reflection, as the named constructor
**`FeatureSet.from_studyset(studyset, kernel_transformer, ...)`**: it belongs to
the thing it builds, and it leaves one public name instead of two. It is a
classmethod rather than `__init__` because `split`, `select_analyses`, `copy`
and the map-reduction methods build the same container from blocks that already
exist; a constructor that ran a kernel would push all of them onto a back door
and take the container's validation with it. The conversion logic stays in an
internal `_FeatureExtractor` class so the stages remain separate methods over
shared configuration, but users never meet it. That also settles the question the
naming review kept circling: with one class there is no near-identical pair to
confuse, and nothing public takes a Studyset and exposes `fit`.

Two consequences:

- `cache_maps` is gone. A per-instance memo cannot be hit when every call
  builds its own extractor, and the workflow it existed for -- comparing
  reducers over one Studyset -- reuses the returned `FeatureSet` instead.
  `memory` remains as the one caching story.
- Wiring `memory` up turned out never to have worked. Kernel transformers cache
  their maps through nilearn's `CacheMixin` at `func_memory_level=2`, and the
  extractor was handing them `memory_level=1`, which the cache reads as a
  request not to cache; two conversions of one Studyset regenerated every map.
  `memory_level` now defaults to 2 and is passed through unchanged, and a test
  counts the kernel's own `_transform` calls rather than asserting that the
  cache directory exists, which is what let the gap through the first time.

The two gallery examples were merged into one,
`examples/05_machine_learning/01_plot_machine_learning_in_nimare.py`: the
workflow reads as one story -- convert, export, split, classify, reduce -- and
splitting it across two pages made the second repeat the first's setup. The
numeric prefix stays because the gallery only executes files matching
`NN_plot_`.

---

## 14. Follow-up: stop re-naming scikit-learn (2026-09-25)

`make_map_reducer("truncated_svd", n_components=50)` was a second vocabulary
for `TruncatedSVD(n_components=50)`, and `features.make_preprocessor(...)` read
as required ceremony in the example even where it wrapped a single transformer
in a `ColumnTransformer` over every column. Both are gone:

- **`make_map_reducer` is removed**, along with the workflow names. Map
  features are a sparse matrix; scikit-learn's transformers reduce them, under
  scikit-learn's names, imported from scikit-learn. The module's public surface
  is `FeatureSet` and `AtlasAggregator` -- the latter because it is the one
  reducer that has to know which voxel each column is, and because resolving an
  atlas (fetched Bunch, image, path, fetcher name, masker) is work nilearn does
  not do for you.
- **`make_preprocessor` returns the reducer itself when there are no descriptor
  columns.** It is a `ColumnTransformer` helper, and a `ColumnTransformer` over
  one block is not worth the indirection. It keeps the two things worth keeping
  for the two-block case: the column boundary, and `sparse_threshold=1.0`,
  since scikit-learn's default of 0.3 would densify a map block above 30%
  density -- about 1.6 GB at 228k columns. `map_columns` and
  `descriptor_columns` stay public so the recipe can be written by hand, and
  the docstring writes it out.
- `fit_transform_maps` rejects a bare atlas with a message naming
  `AtlasAggregator(atlas, masker=features.masker)`, since `transform_maps`
  needs the fitted aggregator for the held-out rows.

The example now leads with `make_pipeline(TruncatedSVD(50), LogisticRegression())`
on `bunch.data`, and reaches for `make_preprocessor` only in the section that
has descriptor columns to protect.

---

## 15. Follow-up: annotations as first-class descriptors (2026-09-25)

`Neurosynth_TFIDF__pain` is one of 3,228 labels in the bundled Neurosynth
Studyset, and one of 794 in the NeuroStore release over 115,747 analyses.
Naming them one at a time was the only way to select them, which is to say
there was no way to use an annotation at all. Three things changed:

- **A field that reads as a glob pattern selects labels.**
  `("annotations", "Neurosynth_TFIDF__*")` takes every match, in the Studyset's
  order, each under its own name. A pattern matching nothing raises and shows
  what the Studyset does annotate with; a pattern against metadata or texts
  says patterns are for annotations.
- **A label selection is read from the sparse `LabelBlock`**, not from
  `annotations_df`, and the descriptor block is allowed to be sparse from there
  on: through `features`, `split`, `select_analyses` and `to_sklearn`. The
  bundled annotation is 1,167 non-zeros in 54,876 cells; the release annotation
  is 3 million in 92 million. `FeatureSet.__init__` no longer coerces the
  descriptor block through `np.asarray`.
- **Absence means zero for a pattern selection**, which is what a sparse
  annotation means, so `missing_values` has nothing to report about one. An
  exactly named field keeps value semantics, where absent means missing. The
  rule is *pattern gives a label matrix, name gives a value*, which is
  predictable without knowing how many labels matched.

Provenance keeps the selectors as given plus `n_descriptor_features`, rather
than thousands of expanded names.

Two things fell out of writing it. `FeatureSet.descriptors` -- the frame of
raw descriptor values -- was **removed**: it held the same numbers as
`descriptor_features` under the same names, it would have densified a label
block, and the story it was documented for was wrong. A rejected categorical
field never reached it, because rejection happens before any container exists,
so the message telling users to read raw values from it could not be followed.
The message now names the Studyset table the values are actually in. And the
constant-target guard earned its place unprompted: asking for
`Neurosynth_TFIDF__pain` as the target of the bundled 17-study set fails,
because that set is motor and language studies and the label is zero
throughout.

---

## 16. Review, and what it found (2026-09-26)

The branch was reviewed as a whole before merge, with an independent pass over
the diff as well as a read of my own. Eight correctness bugs were confirmed by
reproduction, six of them in the two features added last:

1. An exact label whose name contains a glob metacharacter could not be
   selected at all -- 878 of the bundled NeuroStore studyset's own labels are
   named `...groups[0].BMI`. An exact name now wins over pattern matching.
2. A pattern that matched a non-numeric label produced a silent column of
   zeros, where the same label named exactly was refused. 868 of that
   studyset's 926 labels are non-numeric, so this was not a corner case.
   Patterns now refuse them, naming them.
3. Pattern selection crashed on any Studyset carrying more than one
   annotation, because selection went through `annotations_df` (which merges
   them) and extraction through `label_block()` (which refuses to). Both now go
   through `label_block_for`.
4. `missing_values="raise"` fired for analyses that `missing_coordinates="drop"`
   had already removed, aborting a conversion over a row that would not have
   been in the output.
5. `split` handed train, test and parent the same provenance dict, so a
   mutation leaked between them and every part reported the parent's row count.
6. `split`, `copy` and `select_analyses` returned a plain `FeatureSet`,
   dropping a subclass's type despite the `container=cls` mechanism added for
   exactly that.
7. The constant-target guard ran before row retention, so a single-class target
   survived when the minority class was dropped for want of coordinates.
8. Study-level metadata was not inherited by an analysis whose sibling declared
   the same field, because `PerAnalysis` falls back to the study level only
   when no analysis declares it. The metadata frame merges correctly, and is
   now what the module reads.

The first three share a cause worth naming: there were two sources of truth for
what an annotation contains, and they disagreed about naming, dtype and
multi-annotation merging. They are now one object, `_Fields`, which answers
what exists, resolves a selector against it, and reads the values -- so a
label's name, its numeric-ness and its values cannot come from different
places. That also removed the dense `annotations_df` read that selection used
to perform on every pattern.

The review's other finding was that design rationale had accumulated in
docstrings -- 33% of the module -- because `nimare.ml` had no narrative
documentation page at all, only an autosummary stub. `docs/machine_learning.rst`
now sits beside `cbma` and `decoding` in the methods toctree and holds the
reasoning; the docstrings are back to parameters, returns and raises.
