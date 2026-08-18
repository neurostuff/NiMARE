# GCLDA Python Back-port — Design

Date: 2026-08-18
Branch: `rs-gclda`
Status: approved, ready for implementation planning

Back-ports the structural memory and speed wins discovered during the Rust port into
`nimare/annotate/gclda.py`, so users of the Python implementation get them without a Rust
toolchain.

## Goal

Cut `GCLDAModel`'s peak memory roughly in half and make the log-likelihood phase several times
faster, **without changing the fitted model** and **without breaking the Rust port's bit-exact
regression suite**.

## Context

Two of the Rust port's improvements already landed in Python as pre-port commits:

- `6672389` — fixed the `compute_log_likelihood` off-by-one that made every reported
  log-likelihood wrong (indices were already 0-based; the code subtracted 1).
- `540280b` — replaced `np.linalg.inv`/`slogdet` with a closed-form 3x3 adjugate: 2.5x faster,
  exactly symmetric, and reproducible bit-for-bit.

What follows is the remainder, ordered by measured value. All figures are from
`benchmarks/gclda_rust_results.md` at Neurosynth v7 scale (14,371 documents, 3,228 terms,
507,891 peaks, `n_topics=100`, `n_regions=2`), where Python's measured peak RSS is **2.85 GiB**.

## Measured memory breakdown

| Component | MiB | Addressed by |
|---|---:|---|
| Interpreter + NumPy/numba/nibabel/nilearn/NiMARE imports | ~1,040 | Nothing — measured floor |
| `peak_probs` array (`n_peaks x n_topics x n_regions`) | 775 | **Change 1** |
| Dense `spatial_dists` + two `V x T` outputs | 523 | **Change 3** (partial) |
| Dense `D x W` log-likelihood matmul | 354 | **Change 2** |
| Dense `count_df` (caller-supplied) | 354 | Out of scope — API change |
| Token index arrays (int64) | 148 | Out of scope |

The ~1,040 MiB floor is measured directly: at the `tiny` benchmark scale (30 documents, 200
peaks), where GCLDA's own data is negligible, Python's peak RSS is still 1.04 GiB.

## Global constraint: the bit-exactness contract

`nimare/tests/test_gclda_rust.py` compares the Rust trainer against this Python implementation
**bit-for-bit** — per-iteration state equality across four region configurations (Level 2) and
end-to-end outputs (Level 3). Any change here that alters the sampling path's arithmetic, its
operation order, or its random-number consumption breaks that suite.

Every change below is therefore either arithmetic-order-preserving, or confined to the
log-likelihood, which never feeds back into sampling and is already compared with a `1e-10`
relative tolerance rather than bit-exactly (see the header comment in
`rust/gclda/src/loglik.rs`: BLAS `ddot` ordering is not reproducible across builds).

**Acceptance gate for this work: the full existing Rust regression suite must still pass,
unchanged.** If it does not, the change is wrong, not the test.

## Change 1 — Block-wise peak PDF evaluation (~775 MiB)

`_jit_get_peak_probs` materializes an `n_peaks x n_topics x n_regions` float64 array every
iteration — 775 MiB at Neurosynth scale — which `_jit_update_peak_assignments` then streams
through exactly once, reading only the current peak's `T x R` block.

Replace with block processing: for each block of peaks, fill that block's PDFs with a
`@njit(parallel=True)` kernel over `prange`, then run the existing sequential sampler over just
that block. Memory drops from `O(n_peaks x T x R)` to `O(block x T x R)` — 12 MiB at a block
size of 8192.

**Do not simply fuse the evaluation into the sampling loop.** That is what the Rust port did,
and it is measurably *slower*: the sampling loop is sequential, so fusing serializes the
Gaussian evaluation that `parallel=True` currently spreads across every core. Blocking keeps
both properties.

### Structural requirements

These are the details that make it bit-exact, and each is a way to get it wrong:

- **Drive the block loop from Python, not from inside an `@njit` function.** numba applies
  `prange` parallelism at the top level of a `parallel=True` function; nesting it under another
  jitted driver does not reliably parallelize. Two jitted calls per block over ~62 blocks per
  iteration is negligible overhead.
- **Seed the RNG exactly once, before the block loop** — not per block. numba's random state is
  global per thread and persists across jitted calls, so seeding once preserves the current
  stream exactly. Re-seeding per block would change every draw after the first block.
- **Hoist `region_totals` out of the sampler** and pass it across block calls. It is currently
  computed once at the top of `_jit_update_peak_assignments` and mutated per token; recomputing
  it per block would change its float accumulation.
- Count matrices are NumPy arrays mutated in place, so they carry across block calls unchanged.
- Per-peak arithmetic, sampling order, and the number of random draws are all untouched.

Expected effect: memory −775 MiB; time neutral to slightly better (same parallelism, better
cache locality). This is a memory change, not a speed change.

## Change 2 — Sparse log-likelihood (~354 MiB, 3-10x on that phase)

`compute_log_likelihood` builds `p_wtoken_g_doc = np.dot(docprobs_z, wordprobs.T)`, a dense
`14,371 x 3,228` float64 matrix (354 MiB, ~4.6 GFLOP), then reads only the entries at observed
`(doc, word)` pairs. There are 1,049,300 such pairs, so the useful work is ~105 MFLOP — about
**44x less arithmetic than the matmul performs**.

Replace with a `@njit` loop over the unique observed pairs, computing each dot product directly:
`O(nnz x T)` time and `O(nnz)` memory. Python currently spends 3.73 s per log-likelihood call
against Rust's 1.23 s, and this is the bulk of that gap.

**This changes reported log-likelihood values slightly**, because a sequential per-pair
accumulation rounds differently from BLAS's `ddot`. That is acceptable and arguably an
improvement:

- Log-likelihood never feeds back into sampling, so the fitted model is bit-identical.
- The Rust suite already compares these values at `1e-10` relative tolerance for exactly this
  reason, so the comparison does not tighten or loosen.
- Rust already computes them sparsely, so Python and Rust would agree *more* closely, not less.

Precedent: `540280b` already changed shipped NiMARE numerics deliberately, for the same class of
reason, as its own reviewable commit.

## Change 3 — In-place probability distributions (~175 MiB)

`get_probability_distributions` holds `spatial_dists` plus both `V x T` outputs live
simultaneously: 3 x 228,483 x 100 x 8 B = 523 MiB.

Compute the column sums and row sums first, allocate `p_voxel_g_topic` from `spatial_dists`, then
normalize `spatial_dists` **in place** to become `p_topic_g_voxel`. One array instead of three.

Element-wise arithmetic is unchanged, so results are bit-identical. Care is needed only with the
two `np.nan_to_num` calls, which must apply to the same values as before.

## Out of scope

- **Sparse `count_df` ingest.** Rust wins here by streaming a TSV, but Python's public API
  *accepts* a dense DataFrame — 354 MiB is already materialized in the caller before
  `GCLDAModel` is constructed. Fixing this means changing the public signature to accept sparse
  input, which is a separate API discussion.
- **Narrowing token index arrays to int32** (148 -> 74 MiB). Plausible, but it touches the
  dtypes the jitted samplers are compiled against and risks the bit-exactness contract for a
  modest gain.
- **Parallelizing `_jit_accumulate_region_stats`.** It is 0.14% of runtime, and parallelizing a
  float accumulation would break bit-exactness outright.

## Expected results

| Metric | Current | Projected |
|---|---:|---:|
| Peak RSS (Neurosynth, T=100) | 2.85 GiB | **~1.5 GiB** |
| Log-likelihood phase | 3.73 s/call | **~0.4-1.2 s/call** |
| Total wall clock | 2.690 s/iter | ~2.5 s/iter (log-likelihood only) |
| Fitted model | — | **Bit-identical** |

Projections are replaced by measurements. `benchmarks/bench_gclda_rust.py` already reports
Python peak RSS and per-phase timings, so the same driver measures the before and after.

## Consequence for the Rust port, stated deliberately

This work removes most of the Rust port's headline advantage: peak RSS 11.5x becomes roughly
6.1x, and essentially all of the remainder is structural to the runtime rather than the
algorithm — ~1.04 GiB of interpreter and library baseline that Rust does not carry, `u32`
instead of int64 token indices, streamed TSV ingest, and streamed `.npy` output.

That is the correct outcome: most NiMARE users run the Python path and should not need a Rust
toolchain to get a 2x memory reduction. But it means the Rust port's remaining justification is
~1.2x speed plus the absence of an interpreter baseline, and whether that justifies maintaining
a Rust crate is a decision worth taking deliberately rather than by default.

## Testing

- **The existing Rust regression suite is the primary gate**, run unchanged: Level 2
  per-iteration equality across all four region configurations, and Level 3 end-to-end outputs.
  Changes 1 and 3 must leave it bit-identical; Change 2 must stay inside the existing `1e-10`
  log-likelihood tolerance.
- **New — block-size invariance in Python**, mirroring the Rust test: identical outputs across
  several block sizes including 1, a non-power-of-2, and a value exceeding the peak count.
- **New — log-likelihood equivalence**: the sparse implementation compared against the dense one
  on a small corpus, within a documented tolerance, with the dense version retained in the test
  only as the reference.
- `nimare/tests/test_annotate_gclda.py` must continue to pass unchanged.
- Memory measured with the existing benchmark driver, not estimated.

## Risks

| Risk | Mitigation |
|---|---|
| Re-seeding per block silently changes every draw | Seed once before the block loop; Level 2 equality fails immediately if violated |
| `region_totals` recomputed per block changes accumulation | Hoisted and passed across calls; covered by Level 2 |
| numba does not parallelize a nested `prange` | Block loop driven from Python; verify with thread scaling, not by assumption |
| Sparse log-likelihood drifts outside the `1e-10` tolerance | Compared against the dense implementation directly in a new test |
| In-place normalization aliases a still-needed array | Compute both sums before any mutation; Level 3 covers the outputs |
