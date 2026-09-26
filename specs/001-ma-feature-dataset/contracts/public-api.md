# Public API Contract: `nimare.ml`

This contract defines the additive public surface for converting NiMARE
Studysets into scikit-learn-compatible feature datasets.

**Revised 2026-09-25** against `interface-design.md`, which records the options
weighed for each decision below. Terminology: each dataset row is an analysis
row.

## Module

`nimare.ml`

Conversion is `Studyset.to_bunch`, a method on the collection being converted.
`nimare.ml` holds only what a Studyset cannot answer on its own, and is
exported from `nimare/__init__.py` and documented in `docs/api.rst`: its public
names are `MaskerTransformer`, `describe_fields` and `make_nimare_column_transformer`. There is
no container class. Every other reduction is an ordinary scikit-learn
transformer: the module must not re-export scikit-learn under NiMARE names.

`to_bunch` MUST import scikit-learn when called rather than at module load, so
that `nimare.studyset` does not depend on it.

## Division of Responsibility

NiMARE owns extraction: reading the Studyset, generating modeled activation
(MA) maps, aligning every row to its analysis, and reporting what is missing.
scikit-learn owns evaluation: splitting, fitting, reduction and scoring.

Kernel transformation is row-independent, so building the whole matrix before
splitting leaks nothing. Everything that learns *across* rows must be fit on
training rows only, which is what a `Pipeline` is for. The module must not
reimplement that machinery.

## Utility Preference Order

For every task, implementation must look for reusable functionality in this
order before adding local helpers:

1. Existing NiMARE utilities, classes, fixtures, and documentation conventions.
2. nilearn utilities for mask/image-aware operations.
3. scikit-learn utilities for dataset containers, splitters, preprocessing,
   decomposition, and pipelines.
4. New local helpers only when none of the above provides the needed behavior.

## `Studyset.to_bunch`

`studyset.to_bunch(kernel_transformer, **options)` is the public entry point.
It converts one Studyset and returns one `sklearn.utils.Bunch`.
Conversion is a single call, not a configure-then-call pair: the settings are
arguments, and what comes back is the thing the researcher works with.

It is a method on the Studyset because that is the thing being converted, and
the bundle it returns is inert data. Unlike the other `to_*` methods it
computes rather than reformats, which the docstring MUST state: it runs a
kernel over every analysis.

The conversion logic lives in an internal `_FeatureExtractor` class, so that the
stages -- field selection, target handling, row retention, map generation,
provenance -- stay separate methods over shared configuration. That class is not
exported, not documented, and not part of the public surface. No public object
in this module takes a Studyset and exposes `fit` or `fit_transform`.

### Signature

Required, positionally:

- `studyset`: the Studyset to convert. One analysis becomes one row.
- `kernel_transformer`: existing NiMARE kernel transformer instance or class.
  No implicit scientific default is selected; public examples must pass an
  explicit kernel transformer.

Keyword-only:

- `descriptor_fields`: field selectors appended to the feature matrix as extra
  numeric columns.
- `target_field`: field selector for `y`.
- `target_transformer`: callable or stateless transformer applied to the raw
  target values, for free-text and multi-label fields that need a label
  extractor.
- `missing_coordinates`: `"drop"` (default) or `"include"`.
- `missing_values`: `"raise"` (default), `"drop"` or `"keep"`, for missing
  descriptor and target values.
- `memory`, `memory_level`: joblib cache location for MA map generation,
  applied to a copy of the kernel transformer when it does not define its own,
  following the NiMARE `CacheMixin` convention. Kernel transformers cache their
  maps at memory level 2, so `memory_level` defaults to 2 here; a lower level
  is a request not to cache. Repeated conversions of the same Studyset then
  reuse the maps, across processes as well as within one.

The option vocabulary is validated before any work is done.

### Required behavior

- Use Studyset-native access for IDs, coordinates, masker, metadata,
  annotations, and texts.
- Generate MA features through `KernelTransformer.transform(..., return_type="sparse")`.
- **Align map rows to analyses by id.** Kernel transformers return maps ordered
  by analysis id and discard the ids that name them, so positional pairing is
  only correct while the Studyset happens to be in sorted order. A row count
  that cannot be reconciled must raise.
- Reject duplicate analysis identifiers, which would otherwise collapse into
  one map row unnoticed.
- Determine study groups from the Studyset's per-analysis study ids.
- When `missing_coordinates="include"`, analyses with no coordinates are
  all-zero sparse map rows. When `"drop"`, they are removed before row
  construction and recorded in provenance.
- Append numeric descriptor fields directly. Reject categorical and text
  descriptor fields, naming the field, its kind, and the two supported routes:
  encode it and select the numeric result, its raw values being in the Studyset
  tables the message names.
- Export scalar numeric and scalar categorical targets as a one-dimensional
  `y`. Reject raw free-text and multi-label targets unless `target_transformer`
  is supplied, and reject a target that has one value for every analysis.
- Report missing descriptor and target values under `missing_values="raise"`,
  naming the fields and the affected analysis ids; record them in provenance
  under `"drop"` and `"keep"`. Only analyses that survive `missing_coordinates`
  can be missing anything: the rest are not in the output to be missing from.
- Refuse a target that is constant over the analyses that were **kept**, since
  a minority class can leave with the analyses that had no coordinates.
- Inherit study-level metadata even when a sibling analysis declares the same
  field.
- Never mutate the caller's kernel transformer.

## The bundle

The aligned container, and the module's one class. Row `i` is analysis `ids[i]` from study `study_ids[i]`,
and that order is preserved by every method.

### Required keys

- `data`: the analysis-by-feature matrix, map columns then descriptor columns.
  Sparse while the map features are unreduced.
- `target`: optional row-aligned prediction target.
- `groups`: one study label per row, for a group-aware splitter.
- `ids`: full Studyset analysis identifiers, `<study_id>-<analysis_id>`.
- `feature_names`: names for `data` in column order.
- `map_columns`, `descriptor_columns`: the column slices of `data`. A slice is
  what a `ColumnTransformer` takes, so these are the column spec, and no other
  marking of the voxel columns is possible: scikit-learn hands every
  transformer the whole of `X`.
- `descriptor_names`: the descriptor columns, in order, under their real names.
- `masker`: the masker defining voxel order for unreduced map features. An
  atlas reducer cannot work this out from `data`, which is why the bundle
  carries it.
- `provenance`: conversion settings and source Studyset details, including
  `missing_coordinates`, `dropped_ids`, `missing_value_ids`, and the kernel
  transformer and its parameters.
- `train`, `test`: row positions of a grouped holdout. Present only when
  `test_size` was given.

Row selection happens on the Studyset, before conversion, with the
`select_analyses` / `slice` pair it already has. A bundle is subset by indexing
its aligned keys together, which is scikit-learn's own idiom and needs no
NiMARE API.

### Errors

- Raise `ValueError` when feature, target, descriptor, group, or row dimensions
  do not align.
- Raise `ValueError` when a split cannot be created from the available number of
  study groups, or when `test_size` cannot serve one.
- Raise `ValueError` when a reducer returns a row count that does not match the
  input dataset.

## Field Selectors

A selector names one descriptor or target field, or a set of annotation
labels, using the vocabulary NiMARE estimators already use in
`_required_inputs`:

- a bare field name (`"sample_sizes"`), looked up in metadata, annotations and
  texts in turn; an ambiguous name raises and asks for the explicit form;
- a `(source, field)` tuple (`("annotations", "motor_label")`), for the names a
  bare selector cannot express.

These are one spelling per situation, not two for the same one, and there MUST
NOT be a third: the `{"source": ..., "field": ...}` mapping §5.4 kept as an
undocumented alias is removed, since `nimare.ml` has never been released and
the alias was courtesy to nobody.

A tuple is one selector; a list holds several. Sources are `"metadata"`,
`"annotations"` and `"texts"`, with `"annotations_df"` and `"text"` accepted as
aliases.

### Annotation labels

A field that reads as a glob pattern -- `("annotations", "Neurosynth_TFIDF__*")`
-- selects every annotation label matching it, in the Studyset's order, each
under its own name. An exact name MUST win over pattern matching, so that a
label called `ParticipantDemographicsExtractor.groups[0].BMI` is selectable at
all; 878 of the labels in the bundled NeuroStore studyset are named that way.
A pattern MUST refuse the non-numeric labels it matches, the way an exactly
named field is refused, rather than reading them as zeros. This is how an annotation is used at all: the Neurosynth
release annotates 115,747 analyses with 794 labels, and naming them one at a
time is not a workflow; the 2026-09 release carries 924.

Such a selection MUST be read through `label_block_for`, which is also what
decides what the labels are called, so that selection and extraction cannot
disagree about a Studyset carrying several annotations. It MUST stay sparse through extraction, splitting and export, because a dense read of that
annotation is 92 million cells holding 3 million values. Label names MUST
survive whole, double underscores included, since `Neurosynth_TFIDF__pain` is
the name of the thing; where a name has to be softened for a scikit-learn step
name, the softening is internal and `descriptor_names` keeps the real one.

A label no analysis carries is a zero, not a gap, so `missing_values` has
nothing to report about a pattern selection. An exactly named field keeps the
value semantics it has today, where an absent value is missing.

A pattern matching no label MUST raise and show what the Studyset does
annotate with. Patterns are for annotations; a pattern against another source
MUST say so.

A pattern's brackets are an `fnmatch` character class, which is not what they
mean in an extractor's repeated fields (`groups[0].count`). A pattern matching
no label MUST therefore be retried with its brackets taken literally before it
raises; a pattern that already matches MUST be left alone, so that a deliberate
character class such as `[mt]*` keeps working. Where a pattern spans both kinds,
the refusal MUST name `describe_fields` as the way to select the numeric ones.

Provenance records the selectors as they were given, plus
`n_descriptor_features`, rather than thousands of expanded names.

Numeric metadata is read through `nimare.studyset.requirements.PerAnalysis`, so
study-level fields are inherited by their analyses and list-valued fields such
as `sample_sizes` are reduced the way the rest of NiMARE reduces them.

Default field behavior:

- Descriptor fields must resolve to numeric values; there is no implicit
  encoding.
- Target fields may resolve to scalar numeric or scalar categorical values.
- Raw title, abstract, description, and other free-text fields, and multi-label
  fields, require an explicit `target_transformer` to become a target, and
  cannot become descriptor features.

### The train/test split

`to_bunch` MUST accept `test_size` and `random_state`. With `test_size`, the
bundle MUST also carry `train` and `test` row positions from a
`GroupShuffleSplit` over `groups`, so that no study appears in both. Without
it, both keys MUST be absent: the return type is one `Bunch` either way, never
a tuple, and never `(bundle, None)`.

`test_size` counts studies, as a fraction or a count. A value that would leave
a partition empty, or a Studyset with fewer than two studies, MUST raise and
say so.

This exists because the grouping is domain knowledge, not because splitting
needs a NiMARE API: a plain `train_test_split` on the bundled Studyset puts 112
of its 320 studies on both sides. Splitting is milliseconds against a
conversion that runs a kernel, so the docstring MUST point at `groups` and a
scikit-learn group splitter for repeated splits and for cross-validation.

### Missing values per role

`missing_values` MUST accept a mapping from role to policy, with roles
`target` and `descriptors` and the same three policies. A role the mapping
does not name is `raise`. An unknown role or policy MUST be refused by name.

This exists because the roles differ in what can be done about a gap: a
descriptor gap is imputable inside a pipeline and a target gap is not, so a
single setting forces either dropping rows that only lack a descriptor, or
producing a `NaN` target that fails inside the estimator rather than at
extraction. The refusal under `raise` MUST mention the mapping form.

## `describe_fields`

```python
describe_fields(studyset, source=None, min_coverage=0.0) -> pandas.DataFrame
```

Reports the fields a selector may name, so that choosing one on a
release-scale Studyset is a query rather than a search. The 2026-09 NeuroStore
release offers 997 of them, and 875 are reported by fewer than one analysis in
a hundred.

### Required behavior

- Returns one row per field, with columns `source`, `field`, `kind`,
  `coverage`, `n_unique` and `example`, ordered by descending coverage.
- `kind` MUST be the kind `from_studyset` reads the field as. Both MUST share
  one reader, so a field described as numeric cannot then be refused as
  non-numeric by extraction.
- `coverage` is the fraction of analyses reporting a value, by the same
  definition of absence `missing_values` uses.
- `source` restricts the report to one source; `min_coverage` drops fields
  below a coverage.

## Reduction

Map features are an ordinary sparse matrix, so ordinary scikit-learn
transformers reduce them: `TruncatedSVD`, `VarianceThreshold`,
`SparseRandomProjection` and anything else that accepts sparse input, used as
scikit-learn documents them and imported from scikit-learn. A reducer that
cannot read sparse input (`PCA`, for one) fails when it is fitted, with
scikit-learn's own message. There is no NiMARE vocabulary for any of this, and
no factory that re-names it.

`MaskerTransformer` is the one transformer the module adds, because it is the
one that has to know which voxel each column is. A nilearn masker is already a
scikit-learn transformer; what it is not is one that takes an array, since it
takes images. This bridges that: rows are unmasked into images in the
`source_masker`'s space, handed to the masker, and returned as an array.

It accepts any nilearn masker, cloned rather than modified, or anything nilearn
loads as an atlas:

- a `Bunch` from a `nilearn.datasets.fetch_atlas_*` function, read for its
  `maps` and, when present, its `labels`;
- a 3D (deterministic) or 4D (probabilistic) atlas image, or a path to one;
- the name of a nilearn fetcher, such as `"harvard_oxford"`, with any arguments
  it needs in `masker_kwargs`.

A 4D atlas is summarised with a `NiftiMapsMasker` and a 3D one with a
`NiftiLabelsMasker`, both with `resampling_target="data"`. A `NiftiMasker`
returns voxels rather than regions, which MUST be allowed: it is how nilearn's
smoothing, standardizing and detrending reach these features, and the bridge
carries it unchanged. An image that is neither 3D nor 4D, and a string that
names neither a file nor a fetcher, must each raise and say what was expected. Region definitions, resampling and the aggregation strategy stay
nilearn's business; `get_feature_names_out()` reports region names from the
atlas when it carries them, and from the masker otherwise.

How many regions an atlas yields is nilearn's answer, not NiMARE's, and it
differs by version: a region falling outside the mask is kept by nilearn 0.12
and dropped by 0.13, and for a maps atlas 0.13 drops it from the output without
dropping it from `maps_img_` or `n_elements_`. `get_feature_names_out()` must
therefore report exactly as many names as the masker returns columns, counting
them rather than trusting either attribute, and must fall back to positional
names when the atlas's own labels cannot be matched to the surviving regions.
A feature matrix is comparable across environments only when the nilearn
version is, which the docstring says.

### Applying a reducer to the map columns only

```python
make_nimare_column_transformer(bunch, *transformers, remainder="drop",
                               sparse_threshold=1.0, n_jobs=None, verbose=False,
                               verbose_feature_names_out=True)
```

`sklearn.compose.make_column_transformer` with the bundle filled in. It MUST
take the same `(transformer, columns)` pairs, name the steps the way
`_name_estimators` does, and pass `remainder`, `n_jobs`, `verbose` and
`verbose_feature_names_out` through unchanged. Anything it does not cover is
written out as a `ColumnTransformer` over `bunch.map_columns` and
`bunch.descriptor_columns`.

What it adds MUST be limited to what scikit-learn cannot derive from an array:

- `columns` may be a name or a list of names, as it may be for a
  ColumnTransformer reading a frame. `"maps"` and `"descriptors"` name the
  blocks; a descriptor may be named by its field name. A name that is neither
  MUST raise and list both.
- An atlas in the transformer slot MUST be resolved to an `MaskerTransformer`
  bound to the bundle's masker, and the step named after the aggregator rather
  than the atlas object.
- Each step MUST report the names of the columns it was given, so that a fitted
  coefficient can be read back to its field; a ColumnTransformer selects by
  position here, because the feature matrix is an array rather than a frame.
- A block MUST stay sparse unless the transformer cannot take sparse input,
  which is asked by fitting a clone on a sparse probe.
- `sparse_threshold` MUST default to 1.0 rather than scikit-learn's 0.3, which
  would densify an unreduced map block.
- Leaving descriptor columns unclaimed under `remainder="drop"` MUST raise.
  Dropping them is available, but only by saying so.

A string in the transformer slot MUST be `"passthrough"` or `"drop"`, as for a
ColumnTransformer; `"passthrough"` keeps the block's sparsity and its names.

## Documentation Contract

Required public example, one page covering the whole workflow:

- `examples/05_machine_learning/01_plot_machine_learning_in_nimare.py`

The gallery only executes files matching `NN_plot_`, so the numeric prefix
stays even with a single example.

Required docs:

- API autosummary entries in `docs/api.rst`.
- Numpydoc docstrings for all public classes and functions.
- The 1,000-study conversion-and-split budget of <=3 minutes and <=5 GB peak
  memory, checked by a `performance_smoke` test.
