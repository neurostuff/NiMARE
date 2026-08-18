# GCLDA Rust — Block-wise Parallel PDF Evaluation and float32 Compute Path

Date: 2026-08-18
Branch: `rs-gclda`
Status: implemented (block-wise PDF evaluation); the float32 compute path described below was
**cut before implementation** (commit `0c9bec5`) and never built — see "Post-implementation
correction" at the bottom.

Follow-up to `2026-08-15-gclda-rust-port-design.md`, addressing the one regression that
benchmark measurement exposed in the completed port.

## Goal

Recover the peak-sampling performance the original port lost, without giving up bit-exact
reproducibility against the Python implementation. Secondarily, add an opt-in float32 compute
path for the spatial evaluation.

## Why: what the measurements showed

`benchmarks/gclda_rust_results.md` records the completed port at Neurosynth scale (14,371
documents, 3,228 terms, 507,891 peaks, T=100, R=2) as a **wall-clock wash (1.01x)** with an
11.5x reduction in peak memory. Per-iteration Rust cost decomposes as:

| Phase | s/iter | Share |
|---|---:|---:|
| Peak sampling | 1.348 | **50.7%** |
| Word sampling | 1.183 | **44.5%** |
| Log-likelihood (amortized, `loglikely_freq=10`) | 0.123 | 4.6% |
| Region update | 0.004 | 0.14% |

Peak sampling is **0.62x** against Python — Rust is *slower*. The cause is understood and is a
direct consequence of the original design's A1 "fuse the peak PDF into the sampler" decision:

Python's `_get_peak_probs` is `@njit(parallel=True)`. It evaluates all
`507,891 x 100 x 2 = 101.6 million` Gaussian densities per iteration across **all 28 cores**,
materializes them into an ~800 MB array, and streams that array through a sequential sampler.
The Rust port eliminated the array by computing each peak's `T x R` block on demand inside the
sampling loop — but because that loop is inherently sequential, it also made those 101.6 M
evaluations **single-threaded**.

The original design justified A1 on the premise that the phase was memory-bandwidth-bound. That
premise was wrong: at this scale the phase is *compute*-bound, and losing 28-way parallelism
costs more than the eliminated memory traffic saves.

**Fusion and parallelism are not actually in conflict.** The evaluations are order-independent;
only the sampling is sequential. Blocking recovers both.

## Decisions

| Decision | Choice |
|---|---|
| Strategy | Evaluate PDFs one block of peaks at a time, in parallel; sample each block sequentially |
| Block size | `--peak-block-size`, default 8192 |
| Bit-exactness | **Preserved** for the default f64 path; guarded by the existing Level 2/3 suite |
| float32 scope **(cut — not implemented)** | PDF evaluation and sampling weights; region parameters stay f64 |
| float32 flag **(cut — not implemented)** | `--compute-dtype {f64,f32}`, default `f64`, distinct from the existing `--output-dtype` |
| float32 validation **(cut — not implemented)** | Short-run tolerance against f64, documented experimental |
| Sequencing | Measure the PDF/sampling split **before** implementing |

## Design

### 1. Block-wise PDF evaluation

A new primitive alongside the existing `Model::peak_probs_for`:

```rust
fn peak_probs_block(&self, start: usize, len: usize, out: &mut [f64])
```

fills `out` (length `len * n_topics * n_regions`) with `p(x_i | topic, region)` for peaks
`[start, start + len)`. Parallelized with rayon via `par_chunks_mut(n_topics * n_regions)`
zipped against peak indices: writes are disjoint per peak, and there is no cross-peak
reduction.

`update_peak_assignments` becomes a two-level loop:

```
for block_start in (0..n_ptokens).step_by(block_size):
    len = min(block_size, n_ptokens - block_start)
    peak_probs_block(block_start, len, &mut block_buf)     // parallel, &self
    for i in 0..len:                                       // sequential, &mut self
        peak_probs = &block_buf[i * T * R .. (i + 1) * T * R]
        <existing per-peak body, unchanged>
```

**Element layout is `(local_peak * n_topics + topic) * n_regions + region`**, which keeps the
existing `topic * n_regions + region` ordering *within* each peak's block. Each peak's `T x R`
block is therefore contiguous, and the sequential body's indexing into it is unchanged from
today.

**Why precomputation ahead of the mutations is valid.** `pdf()` reads only
`corpus.ptoken_coords` and the cached region parameters (`regions_mu`, `regions_precision`,
`regions_log_norm`). None of these are mutated by `update_peak_assignments` — it mutates only
`n_peak_tokens_region_by_topic`, `n_peak_tokens_doc_by_topic`, `region_totals`,
`peak_topic_idx`, and `peak_region_idx`. Region parameters are recomputed in `update_regions`,
a separate phase. This invariant is what makes the whole design legal and MUST be stated in a
doc comment on `peak_probs_block`, so a future change that moves region updates into the sweep
is recognized as breaking it.

**Why it stays bit-exact.** Every element is produced by the same `pdf()` call with the same
inputs as today. Parallelism is across disjoint peaks with no reassociated floating-point
reduction, and the sampling order and RNG consumption are untouched.

**Borrowing.** `block_buf` is a local allocation, not a field of `Model`, so the immutable
borrow for `peak_probs_block` ends before the mutable sequential loop begins.

**Rayon threshold.** A block is filled in parallel only when it carries at least 32,768
Gaussian evaluations (`len * n_topics * n_regions`); below that it is filled sequentially, since
rayon's task overhead exceeds the work. At the default block size with T=100, R=2 every full
block clears this comfortably (1.6 M evaluations), so the threshold only takes effect on small
corpora and on the partial final block. The constant is a starting value, confirmed or revised
by the thread-scaling benchmark. The
measured 4x regression in `update_regions` at 8 threads on small corpora
(`gclda_rust_results.md`, thread-scaling table) is direct evidence that unconditional rayon is
a net loss when per-task work is small.

**Edge cases.** `n_ptokens == 0` produces no blocks. `n_ptokens < block_size` produces one
partial block. A partial final block is the norm, not an exception.

**Memory.** `block_size * T * R * 8` bytes — 13.1 MB at the default with T=100, R=2, against
the ~800 MB Python allocates. The formula is documented so users tuning large `n_topics` can
reason about it.

### 2. Log-likelihood

`compute_log_likelihood` consumes the same per-peak PDF helper and gains the same treatment.
At 4.6% of runtime this is worth roughly 3% overall — included because the primitive already
exists, not because it is significant on its own.

### 3. float32 compute path

**CUT before implementation (commit `0c9bec5`) — never built.** This section is kept in
place, unedited, as the record of what was designed and descoped rather than deleted; see
"Post-implementation correction" at the bottom for why. Nothing described under this heading
exists in the codebase.

`--compute-dtype {f64,f32}` (default `f64`). This is **separate from `--output-dtype`**, which
affects only serialization of the two large `V x T` matrices and never the computation.

Under `f32`, the Gaussian evaluation and the sampling weight vector are computed in `f32`.
**Region parameters remain `f64`** and are converted into a small `f32` mirror
(`T x R x 13` values) once per iteration. This is not a stylistic choice: `regions_sigma` comes
from `cross_matrix - outer(sum, sum) / n_obs`, where at Neurosynth scale the cross terms reach
~1.2e8 while the centered result is ~2.5e3. `f32`'s ~7 significant digits would destroy that
subtraction.

Implementation is **generic over a small private `Float` trait** covering the operations used
(multiply, add, divide, `exp`, `ln_1p`, `NEG_INFINITY`), rather than a duplicated `f32` sampler.
The sequential sampler body is subtle enough that two copies drifting apart would be a worse
defect than any performance this flag delivers.

**Documented limit.** `f32` represents integers exactly only below 2^24. The counts involved
stay well below it (per-document topic counts ~10^2; `region_totals` ~5x10^5 against
16.7 million), but the ceiling is real and is documented alongside the flag.

**Honest expectation.** After block-wise parallelism lands, the PDF is roughly 0.15 s of a
~1.76 s iteration, so halving its cost buys about **4%** total. This flag is materially more
valuable before block-wise than after, and must not be presented as a headline feature.

## Sequencing: measure first

Before implementing, add a sub-phase timer splitting peak sampling into PDF evaluation and
sampling. The ~1.5x projection assumes the PDF is 75-80% of that phase, a figure derived from
an `exp()` cost model rather than measured.

- If PDF is **>= 70%** of the phase, proceed as specified.
- If PDF is **< 50%**, the projected gain falls to roughly 1.3x; stop and report before
  continuing, so scope can be reconsidered.

The timer is a few lines and doubles as the before/after evidence for
`benchmarks/gclda_rust_results.md`.

**Coupling:** `test_both_implementations_report_matching_phase_keys` asserts both
implementations expose the same phase-key set, so any new sub-phase key added to Rust must be
added to `nimare/annotate/gclda.py` as well.

## Expected results

| Metric | Current | Projected |
|---|---:|---:|
| Peak-sampling phase | 1.348 s/iter | ~0.45 s/iter |
| Total per iteration | 2.657 s | ~1.76 s |
| Speed vs Python | 1.01x | **~1.5x** |
| Peak RSS | 253.5 MiB | ~267 MiB |
| Bit-exact vs Python | Yes | **Yes** |

Projections are replaced by measurements in the results document. If a measured figure lands
below its projection, the measured figure is what gets reported.

## Testing

**The existing bit-exact suite is the primary guard.** Level 2 (per-iteration state equality,
all four region configurations) and Level 3 (end-to-end outputs and edge cases) must pass
unchanged with block-wise evaluation enabled. This also catches any codegen perturbation
introduced by the generic `Float` refactor on the `f64` path.

**New — block-size invariance.** Identical seed and inputs across `--peak-block-size` values of
`1`, `7`, `8192`, and a value exceeding `n_peaks` must produce bit-identical outputs. This is
the key new test: it targets precisely where this change can break — off-by-one errors at block
boundaries and mishandling of the partial final block. The value `7` is deliberately chosen as
a non-power-of-2 that does not divide the peak count evenly.

**New — float32 tolerance. (CUT — not implemented; the float32 path was cut before
implementation, see "Post-implementation correction" at the bottom, so this test was never
written. Kept in place as the record of what was planned.)** `f32` versus `f64` outputs compared with a relative tolerance over
a small number of iterations, before sampling divergence can accumulate, plus an assertion that
the `f64` default remains bit-exact against Python. The flag is documented as experimental; this
test establishes that it runs and produces plausible output, and deliberately makes no claim of
scientific equivalence.

**Benchmark.** Rerun `benchmarks/bench_gclda_rust.py` at `--scale neurosynth`. The driver's
equality gate must pass before any new timing is recorded.

## Risks

| Risk | Mitigation |
|---|---|
| Block boundary or partial-final-block bug | Block-size invariance test across 1, 7, 8192, and > `n_peaks` |
| Generic refactor perturbs `f64` codegen and breaks bit-exactness | Existing Level 2/3 suite runs unchanged and is the acceptance gate |
| Rayon overhead dominates on small corpora | Size threshold below which the block fills sequentially; regression already measured in `update_regions` |
| A future change mutates region parameters inside the sweep | Invariant documented on `peak_probs_block`; Level 2 would fail immediately |
| `f32` counts exceed 2^24 | Documented limit; values at realistic scale are two orders of magnitude below it |
| Measured PDF share is lower than projected | Measured first; explicit stop-and-report threshold at < 50% |

## Files touched

Below, items marked **(cut — not implemented)** describe the float32 compute path, which was
cut before implementation (commit `0c9bec5`); see "Post-implementation correction" at the
bottom. Everything else in this list was implemented as described.

- `rust/gclda/src/sampler/peaks.rs` — block loop, `peak_probs_block`; generic float **(cut — not implemented)**
- `rust/gclda/src/gaussian.rs` — generic / `f32` PDF evaluation **(cut — not implemented; file never touched)**
- `rust/gclda/src/loglik.rs` — block-wise PDF consumption
- `rust/gclda/src/model.rs` — sub-phase timing; `f32` region-parameter mirror **(cut — not implemented)**
- `rust/gclda/src/bin/gclda-train.rs` — `--peak-block-size`; `--compute-dtype` **(cut — not implemented)**
- `nimare/annotate/gclda.py` — matching sub-phase timing keys
- `nimare/annotate/gclda_rs.py` — pass new parameters through `train_gclda_rust`
- `nimare/tests/test_gclda_rust.py` — block-size invariance; `f32` tolerance **(cut — not implemented)**
- `benchmarks/bench_gclda_rust.py` — expose the new flags
- `benchmarks/gclda_rust_results.md` — updated measurements

## Out of scope

- **SparseLDA bucket decomposition.** The largest remaining win (~2x on the word sampler, 44.5%
  of runtime), but it changes the order and quantity of random numbers consumed, so Rust and
  Python produce different results even when both are correct. That removes exact comparison as
  a test and replaces it with tolerance-based statistical comparison. It is a separate decision
  on its own merits, taken after this work lands and is trusted.
- **GPU offload.** *(Figures corrected twice after measurement — see "Post-implementation
  correction" below.)* The final measured parallel-safe fraction is ~21.3% of pre-change Rust
  runtime, with ~78.7% inherently sequential collapsed-Gibbs work (word sampling plus the
  per-peak sampling body) that no accelerator touches regardless of hardware speed. A naive
  Amdahl derivation from that split gives a ceiling of ~1.271x over pre-change Rust, but
  `benchmarks/gclda_rust_results.md` records a block-size-sweep measurement — 1.319x internal
  speedup at `--peak-block-size 65536` — that exceeds this "ceiling," because blocking also
  improves cache locality in the sequential phase, an effect the pure-parallelization Amdahl
  model does not capture. So 1.271x should not be treated as a hard bound. The conclusion
  nonetheless stands on the sequential-fraction argument alone: GPU offload can only address
  the ~21.3% that block-wise CPU parallelism already captures with 28 cores, so the
  incremental value of a CUDA/wgpu dependency, a hardware requirement, and near-certain loss
  of bit-exactness is low. Not recommended.
- **PyO3 bindings.** A usability improvement, not a performance one, and it would make peak
  memory *worse* by keeping the dense count DataFrame resident in Python alongside Rust's
  working set.
- **Threshold-gating rayon in `update_regions`.** A real measured regression, but that phase is
  0.14% of runtime at Neurosynth scale. Worth doing opportunistically; not worth a task here.

---

## Post-implementation correction (2026-08-18)

Three things in this spec were written before the gate measurement, or before later
decisions, and needed correction after the fact. They are corrected here rather than
silently edited away, because the error is the point:

- **The projected ~1.5x total rested on PDF evaluation being 75-80% of the peak-sampling
  phase.** Task 1 measured it at **35.5-37.9%**, which revised the projection to ~1.2x before
  any optimization code was written. The measured outcome was **1.27x**.

- **The GPU "Out of scope" note went through three stages, not one correction.** The
  original draft (as first written) estimated a 44% parallel-safe fraction and a 1.79x Amdahl
  ceiling. An intermediate correction — made after Task 1's PDF-share measurement but from an
  accounting that counted only the peak-sampler's own PDF evaluation and dropped
  log-likelihood's PDF pass and region update — revised this to ~22.6% / 1.29x. The "Out of
  scope" bullet above has since been updated in place to the final figures below, rather than
  left at that intermediate stage. The final measured derivation, built from all three
  parallel-safe components and recorded in `benchmarks/gclda_rust_results.md`,
  corrects this again to **~21.3% / 1.271x / 78.7% sequential**. That document also shows that
  even the final 1.271x figure is not a hard ceiling on total achievable speedup: a
  block-size-sweep measurement at `--peak-block-size 65536` reached 1.319x internal speedup,
  3.7% above it, because blocking improves cache locality in the sequential sampling body in a
  way the pure-parallelization Amdahl model does not account for. The case against GPU offload
  still holds — it rests on the ~78.7% sequential fraction, not on the exact ceiling number —
  but is *stronger* than this spec first argued (44%/1.79x), not weaker.

- **The float32 compute path (Decisions table, Section 3, and the corresponding testing and
  Files-touched entries above) was cut before implementation, commit `0c9bec5`, and never
  built.** Only block-wise f64 PDF evaluation shipped. The float32 material above is left in
  place, marked cut-and-not-implemented, as the record of what was designed and descoped
  rather than deleted.

~78.7% of Rust runtime is inherently sequential collapsed-Gibbs work (word sampling plus the
peak-sampling body). That is the best current estimate of the parallel-safe/sequential split
for this algorithm, but — per the GPU bullet above — it should not be treated as a hard
ceiling on total achievable speedup.

Measured results: `benchmarks/gclda_rust_results.md`.
