# GCLDA Rust Port — Design

Date: 2026-08-15
Branch: `rs-gclda`
Status: approved, ready for implementation planning

## Goal

Port NiMARE's Generalized Correspondence LDA (`nimare/annotate/gclda.py`) to Rust as a
standalone training binary. Training runs in Rust; the resulting probability matrices are
consumed by NiMARE's existing Python decoding and encoding functions.

Two success criteria, weighted equally:

1. **Correctness.** The Rust trainer reproduces the Python implementation bit-for-bit, given
   the same inputs and seed.
2. **Efficiency.** Lower peak memory and lower wall-clock time than the Python implementation,
   with measured evidence attributing the difference to specific causes.

## Context: the baseline is not naive Python

The existing implementation is already `numba`-JIT compiled, with `parallel=True` on the
spatial kernels (`_jit_get_peak_probs`, `_jit_get_spatial_dists`, `_jit_spatial_pdf`). The
sampling inner loops are compiled by LLVM, the same backend Rust uses. A port therefore cannot
expect large gains on raw scalar arithmetic, and the design does not claim any.

The gains this design targets are **structural**: allocations that should not exist, an
asymptotically wrong log-likelihood computation, and a dense intermediate that destroys cache
locality.

### Where the work actually is, per iteration

For full Neurosynth v7 (~14k documents, ~500k coordinates, ~3k terms, T=100, R=2):

| Phase | Work | Parallel-safe? |
|---|---|---|
| `_update_word_topic_assignments` | O(N_wtokens x T) | No — sequential, RNG-ordered |
| `_get_peak_probs` | O(N_peaks x T x R) Gaussian evals, **allocates ~800 MB** | Yes |
| `_update_peak_assignments` | O(N_peaks x T x R), streams the 800 MB array | No — sequential, RNG-ordered |
| `_update_regions` | O(N_peaks) accumulate + T x R matrix inversions | Yes |
| `compute_log_likelihood` | O(D x W x T) dense matmul, **allocates ~340 MB** | Yes |

Collapsed Gibbs sampling is inherently sequential per token, so parallelism is only legitimate
where the work is already order-independent. Fortunately, that is exactly where the largest
allocations are, so bit-exactness and parallelism do not conflict.

## Decisions

| Decision | Choice |
|---|---|
| Delivery | Standalone CLI binary; crate structured as library + thin binary so PyO3 remains possible later |
| Reproducibility | Bit-exact replication of NumPy's legacy MT19937 |
| Mask input | Rust reads NIfTI directly, mirroring the Python `mask` argument |
| Benchmark data | Real Neurosynth v7 for headline numbers, synthetic generator for scaling curves |
| Optimization strategy | Approach A: faithful arithmetic, structural fixes (see below) |

## Verified foundations

Two assumptions underpinning the design were tested before it was written, because the whole
approach collapses if either is false.

### 1. numba and NumPy share an RNG stream — CONFIRMED

`np.random.seed(s)` followed by `np.random.random()` produces identical `float64` bit patterns
inside and outside `@njit`, for scalar seeds. Verified across seeds 1, 42, and 12345.

Consequence: a single MT19937 implementation in Rust covers both the pure-NumPy initialization
(`np.random.randint` in `__init__`) and every `@njit` sampling call.

### 2. LAPACK's 3x3 inverse is NOT reproducible in scalar arithmetic — CONFIRMED PROBLEM

A naive scalar LU with partial pivoting matched `np.linalg.inv` bit-for-bit in only **104 of
3000** trials (worst relative difference 4.0e-13); `slogdet` matched in 2717 of 3000. NumPy
dispatches to OpenBLAS, whose blocked and vectorized kernels depend on build configuration, CPU
features, and threading. This is not portably reproducible from Rust.

Because these precision matrices feed sampling probabilities directly, a single 1-ulp difference
can flip a sampled index, and the divergence then cascades through every subsequent iteration.

**Resolution — change both sides to a closed-form 3x3 adjugate inverse with a fully specified
operation order.** Peak coordinates are always `(x, y, z)`, so the covariance is always 3x3.
Measured against LAPACK over 3000 random regularized covariances:

| Property | Closed-form adjugate |
|---|---|
| Accuracy vs LAPACK | 4.4e-13 relative (same class as any LU) |
| Self-reproducibility | 3000 / 3000 identical bits |
| Symmetry of result | **Exactly symmetric** (LAPACK's result is not) |
| Speed, 40000 inverse+logdet | **0.064 s vs 0.158 s — 2.5x faster** |

This makes bit-exactness a designed property rather than a coincidence. It also removes 200
LAPACK calls per iteration from the Python hot path and produces the exactly-symmetric inverse
that a covariance inverse mathematically should be. It is a genuine improvement to shipped
NiMARE behavior, not merely a porting convenience.

## Pre-port Python fixes

Two changes land in `nimare/annotate/gclda.py` **before** the port, each as its own commit, so
the Rust work targets a stable and correct reference.

### Fix 1 — log-likelihood off-by-one (correctness bug)

`compute_log_likelihood` contains, at `gclda.py:984`, `:1020`, and `:1022`:

```python
doc = self.data["ptoken_doc_idx"][i_ptoken] - 1   # "convert didx from 1-idx to 0-idx"
word_token = self.data["wtoken_word_idx"][i_wtoken] - 1
doc = self.data["wtoken_doc_idx"][i_wtoken] - 1
```

These indices are built by `docidx_mapper = {id_: i for (i, id_) in enumerate(ids)}` and are
**already 0-indexed**. Subtracting 1 maps document 0 to index -1, which wraps to the last
document. Every reported log-likelihood is computed against shifted indices.

This does not corrupt the fitted model — log-likelihood never feeds back into sampling — but
every logged, stored, and plotted likelihood value is wrong. Rust implements the correct
indexing; the regression harness compares against the fixed Python reference.

### Fix 2 — closed-form 3x3 inverse

Replace `np.linalg.inv` / `np.linalg.slogdet` in `_cache_region_pdf_params` with the closed-form
adjugate described above. Rationale and measurements as given in Verified Foundations.

## Approach A: faithful arithmetic, structural fixes

Bit-exactness constrains optimization more than it first appears. Reproducing `float64` results
exactly requires preserving **arithmetic operation order**. Common micro-optimizations are
therefore unavailable — for example, hoisting a loop-invariant division into a
multiply-by-reciprocal is not bit-preserving and must not be done. Every optimization below is
either allocation-structural or applies to order-independent work.

### A1. Fuse the peak PDF into the peak sampler

`_get_peak_probs` materializes an `n_peaks x n_topics x n_regions` `float64` array every
iteration (~800 MB at Neurosynth scale), which `_update_peak_assignments` then streams through
exactly once, reading only `peak_probs[i, :, :]` for the current token.

Rust computes that `T x R` block for one peak at a time into a small reusable buffer. Memory for
this structure drops from O(N x T x R) to O(T x R). The arithmetic is unchanged, so results are
identical. The same fusion applies to the log-likelihood's use of the peak PDFs.

This phase is currently memory-bandwidth-bound; the `perf stat` counters in the profiling plan
exist to confirm that diagnosis rather than assume it.

### A2. Sparse log-likelihood

`compute_log_likelihood` builds `p_wtoken_g_doc = np.dot(docprobs_z, wordprobs.T)`, a dense
`D x W` matrix (~340 MB, ~4.2 GFLOP at Neurosynth scale), then reads only the entries
corresponding to observed `(doc, word)` pairs. Rust computes those dot products directly:
O(nnz x T) time and O(1) additional memory.

### A3. Sparse ingest

The Python constructor materializes `count_df.to_numpy()`, a dense `D x W` `int64` matrix
(~340 MB), then calls `np.nonzero` on it. Rust streams the TSV directly into token-level index
arrays, never building the dense form.

### A4. Parallelism via rayon, restricted to order-independent work

Region statistic accumulation, the per-topic region parameter updates, and the final
`get_probability_distributions` sweep (V x T x R Gaussian evaluations; ~228k x 100 x 2 for a
2 mm MNI mask). The sequential samplers are left sequential — this is required for correctness,
not merely for exactness.

### A5. Streamed output

Output matrices are written to `.npy` incrementally rather than assembled in memory first.

### Expected gains — stated honestly, to be replaced by measurements

| Component | Expectation | Reason |
|---|---|---|
| Peak memory | Large reduction; target O(outputs) | A1, A2, A3 remove essentially all transient allocation |
| Peak-sampling phase | 2-4x | A1 eliminates an 800 MB/iteration bandwidth-bound stream |
| Log-likelihood | Order of magnitude | A2 replaces O(D x W x T) with O(nnz x T) |
| Word-sampling inner loop | 1.0-1.5x | numba already compiles this well; little headroom |

If a measured figure lands below its expectation, the measured figure is what gets reported.

## Architecture

### Crate layout

```
rust/gclda/
  Cargo.toml
  src/
    lib.rs               public API
    rng.rs               MT19937: seed, random(), legacy bounded randint
    gaussian.rs          closed-form 3x3 inverse, logdet, PDF evaluation
    model.rs             params, data, count matrices, assignments, initialization
    sampler/
      words.rs           z-assignments      (sequential, RNG-ordered)
      peaks.rs           y,r-assignments    (sequential; peak PDF fused in)
      regions.rs         spatial parameters (rayon over topics)
    loglik.rs            sparse log-likelihood
    io/
      tsv.rs             streaming count and coordinate readers
      nifti.rs           mask loading -> boolean volume + affine -> mask_xyz
      npy.rs             .npy writer (f64 / f32 / i64)
    output.rs            output directory writer
  src/bin/gclda-train.rs CLI (clap)
```

Library plus thin binary, so a PyO3 layer can be added later without restructuring.

### CLI

Mirrors the Python signature exactly: `--n-topics`, `--n-regions`, `--symmetric`, `--alpha`,
`--beta`, `--gamma`, `--delta`, `--dobs`, `--roi-size`, `--seed-init`, plus `fit` parameters
`--n-iters` and `--loglikely-freq`, plus `--counts`, `--coordinates`, `--mask`, `--out-dir`, and
`--output-dtype`.

### Index-determining details that must be replicated exactly

These define the axes of every output file. Any deviation silently misaligns results.

- `ids` is the **lexicographically sorted** intersection of count IDs and coordinate IDs, with
  IDs compared as strings. Rust's byte-order `String` ordering agrees with Python's code-point
  ordering for all inputs here.
- All-zero term columns are dropped **in place**, preserving the order of surviving columns.
- Word tokens are expanded in `np.nonzero` row-major order over the `D x W` count matrix. Note
  that the rows of that matrix are in **`count_df` row order**, which is not necessarily
  `docidx` order — `docidx` is derived from the sorted `ids` list, while row order comes from the
  input file. Rust must iterate rows in input-file order and emit `docidx` values through the
  mapping, not iterate in `docidx` order. Within a row, tokens are emitted in ascending word
  index, each repeated by its count.
- `_mask_img_to_bool` is `np.asanyarray(dataobj).astype(bool)` — plain nonzero, not a threshold.
- Symmetric initialization assigns subregion by `(random_pair * 2) + (x > 0)`.

### Output directory

The four required matrices, as `.npy` — a format Rust writes in a few lines and Python opens
with `np.load(..., mmap_mode="r")`, so a 183 MB matrix need never be resident:

- `p_topic_g_voxel.npy` (V x T), `p_voxel_g_topic.npy` (V x T)
- `p_topic_g_word.npy` (W x T), `p_word_g_topic.npy` (W x T)
- `vocabulary.txt` — one term per line, defining the W axis
- `model.json` — parameters, document ID order, mask path/affine/shape, iteration count, timings

Additional outputs required for GCLDA-based decoding, encoding, inspection, and resumption:

- `n_word_tokens_word_by_topic.npy` (W x T) — the raw counts the `p_*` matrices derive from;
  referenced by commented-out code paths in `nimare/decode/`
- `n_peak_tokens_doc_by_topic.npy` (D x T) — per-document topic loadings, needed for any
  document-level annotation
- `regions_mu.npy` (T x R x 3), `regions_sigma.npy` (T x R x 3 x 3),
  `n_peak_tokens_region_by_topic.npy` (R x T) — the spatial model, required to evaluate topics
  at arbitrary coordinates
- `loglikelihood.tsv` — columns `iter`, `x`, `w`, `total`
- `wtoken_topic_idx.npy`, `peak_topic_idx.npy`, `peak_region_idx.npy` — assignment state,
  enabling checkpoint and resume for multi-hour runs

`--output-dtype float32` halves the two large voxel matrices. It affects serialization only and
never the training computation.

### Python integration — `nimare/annotate/gclda_rs.py`

`nimare/decode/{continuous,discrete,encode}.py` touch only `model.mask`, `model.vocabulary`, and
the four `p_*_` attributes. An object exposing exactly those therefore works with all three
existing decoders **with no changes to `nimare/decode/`**.

- `export_gclda_tsvs(count_df, coordinates_df, out_dir)` — write trainer inputs
- `load_gclda_model(dir, mask=...)` — memory-map outputs into a `GCLDAResult`
- `train_gclda_rust(...)` — optional convenience wrapper that invokes the binary

## Regression testing

Four layers, ordered so that a failure localizes to a component rather than to "the port."

**Level 1 — components.** Small golden fixtures generated from Python and committed: MT19937
`random()` streams across several seeds; legacy bounded `randint` at several bounds, including
non-powers-of-2 to exercise the masked-rejection path; `_sample_from_unnormalized` on fixed
weight vectors; 3x3 inverse and logdet on fixed matrices; `mask_xyz` from the bundled mask; and
the complete TSV-to-token-array ingest (`ids`, `vocabulary`, `wtoken_*`, `ptoken_*`). Consumed by
Rust `#[test]`s.

**Level 2 — per-iteration state equality.** The core harness. Both implementations dump complete
state — every count matrix, every assignment vector, `regions_mu`, `regions_sigma` — after each
iteration, and the harness asserts bit-identity at every step, reporting the first differing
iteration and the first differing element. Without this, a port bug introduced at iteration 3 is
invisible until the endpoint, where it is untraceable.

**Level 3 — end-to-end outputs.** All four `p_*` matrices and every auxiliary output,
bit-identical, across: symmetric x {2, 4} regions; asymmetric x {1, 3} regions; multiple seeds;
and edge cases exercising the branch structure — topics with zero observations (the `n_obs == 0`
and `n_obs <= 1` paths), documents with no coordinates, and singleton terms.

**Level 4 — downstream integration.** Rust outputs loaded through `load_gclda_model` and passed
to the three real consumers — `gclda_decode_roi`, `gclda_decode_map`, `gclda_encode` — compared
against the same functions driven by the Python model.

Levels 2 through 4 run Python live and compare against a fresh Rust run, so only the small
Level 1 fixtures are committed. Tests live in `nimare/tests/test_gclda_rust.py` and skip cleanly
when the binary is not built, so the existing suite is never broken by their absence.

## Profiling

**Time.** Matched phase-level instrumentation in both implementations — word sampling, peak
sampling (including PDF evaluation), region update, log-likelihood, I/O — because attributing
the difference is the point. Plus end-to-end wall clock at matched iteration counts; criterion
benchmarks on the Rust inner kernels; `perf stat` cache-miss and bandwidth counters as direct
evidence for or against the A1 bandwidth diagnosis; and thread scaling at 1, 2, 4, and 8 threads
for rayon against numba's `parallel=True`.

**Memory.** Python peak RSS via `memory_profiler`, with `tracemalloc` for attribution; Rust peak
RSS via `/usr/bin/time -v`, with a tracking allocator for peak heap. The specific claim under
test: Rust's peak is O(outputs), not O(n_peaks x T x R).

**Data.** Full Neurosynth v7 for headline numbers; a synthetic generator sweeping `n_topics`,
`n_peaks`, vocabulary size, and document count for scaling curves.

**Reporting caveat.** A 5000-iteration Neurosynth run in Python may take hours. The head-to-head
will therefore use a reduced iteration count with per-iteration cost extrapolated, and will be
labeled as extrapolated rather than presented as a measured full run.

Results land in `benchmarks/bench_gclda_rust.py` with a written summary. They are deliberately
kept out of `asv`, which is built for tracking Python performance over time rather than for
cross-language comparison.

## Risks

| Risk | Mitigation |
|---|---|
| A single sampling divergence cascades and destroys bit-exactness for a whole run | Level 2 per-iteration comparison at high iteration counts on small data catches the first divergence |
| Legacy `randint` masked-rejection semantics are subtle | Level 1 golden vectors at bounds that are and are not powers of 2 |
| NIfTI affine or boolean-mask handling differs from nibabel | Level 1 compares `mask_xyz` bit-for-bit against Python |
| Speedup on the word sampler proves negligible | Expected and stated; the memory and peak-sampling wins are the primary justification |
| Closed-form inverse changes shipped NiMARE numerics | Accepted deliberately: more accurate symmetry, 2.5x faster, landed as a separate reviewable commit |

## Future directions

### Approach B — SparseLDA bucket decomposition

The word-topic sampler is O(N_wtokens x T) per iteration and dominates cost at large `n_topics`.
The SparseLDA decomposition of Yao, Mimno, and McCallum (2009) splits the sampling mass into
smoothing, document, and topic buckets, exploiting the fact that most documents touch few topics
and most words appear under few topics. Cost falls from O(T) per token to roughly O(number of
topics actually occupied), typically a large asymptotic win that grows with `n_topics`.

It is excluded from this design because it **changes the order and quantity of random numbers
consumed**, which forfeits bit-exact reproducibility — the property chosen as this port's
correctness oracle.

It becomes attractive once the bit-exact implementation exists and is trusted, at which point
the exact implementation serves as the reference for validating an approximate-but-faster one.
The natural form is a `--fast` flag: bit-exact by default, SparseLDA opt-in, validated by
statistical equivalence — log-likelihood trajectories and per-topic distributions compared after
Hungarian topic matching — rather than by bit equality.

### Other candidates

- **PyO3 bindings**, removing the TSV round-trip. The crate is already structured as library
  plus thin binary to permit this without restructuring; the cost is maturin and per-platform
  wheel building in a repository that currently ships pure Python.
- **float32 sampling path**, halving bandwidth in the peak sampler. Excluded for the same
  bit-exactness reason, and belongs behind the same `--fast` flag.
- **Multi-chain parallelism** — independent seeds across cores, which is embarrassingly parallel
  and preserves exactness per chain, useful for convergence assessment.
