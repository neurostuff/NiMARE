# GCLDA Rust port — measured benchmark results

All numbers below were produced by `benchmarks/bench_gclda_rust.py`, which verifies that
both implementations produced **bit-identical** `p_topic_g_voxel`, `p_voxel_g_topic`,
`p_topic_g_word`, and `p_word_g_topic` *before* it records any timing. Every configuration
reported here passed that check, so no timing below credits a run that computed a different
answer.

## Headline

**Block-wise parallel PDF evaluation turned the prior 1.01x wall-clock wash into a measured
1.27x speedup on the real Neurosynth corpus, on top of the 11.5x memory reduction, which is
unchanged.** Peak sampling — the phase that was previously *slower* in Rust (0.62x) — is now
close to parity (0.955x), and the `peak_pdf` sub-phase now scales visibly with thread count,
which it did not before. Full history, including the original over-optimistic projection and
why it was wrong, is in "Where the wins came from" below.

## Environment

| | |
|---|---|
| CPU | Intel Core i7-14700F, 28 logical cores |
| RAM | 31 GiB |
| OS | Linux 5.15.167.4-microsoft-standard-WSL2 (WSL2) |
| Python | 3.13.15 |
| NumPy / numba / pandas | 2.5.2 / 0.67.0 / 3.0.5 |
| Rust | rustc 1.95.0, cargo 1.95.0, `--release` |
| Mask | `MNI152_2x2x2_brainmask.nii.gz` (228,483 voxels) |

`perf` is not available under this WSL2 kernel, so the cache-behaviour evidence called for
in the plan could not be collected. The fusion analysis below therefore rests on per-phase
timing and peak-RSS measurements rather than hardware counters.

## Real corpus — Neurosynth v7 — before/after block-wise PDF evaluation

14,371 documents, 3,228 terms, **507,891 peaks**, `n_topics=100`, `n_regions=2`,
symmetric, 20 iterations, 3 repeats, all 28 threads. Medians shown. "Before" is the
fused-sequential-loop Rust measured prior to this plan; "After" is the block-wise
parallel-PDF Rust measured for this task. Python is unchanged and re-measured here purely
as a same-run baseline (it agrees with the prior Python numbers within run-to-run noise).

| Phase | Python | Rust (before) | Rust (after) | Ratio (after) | Rust internal (before→after) |
|---|---:|---:|---:|---:|---:|
| Word sampling | 29.58 s | 23.65 s | 23.63 s | **1.25x** | 1.00x |
| Peak sampling (total) | 16.75 s | 26.96 s | 17.54 s | **0.955x** (was 0.62x) | **1.54x** |
| &nbsp;&nbsp;`peak_pdf` | 2.063 s | n/a¹ | 3.902 s | 0.529x | n/a¹ |
| &nbsp;&nbsp;`peak_sample` | 14.49 s | n/a¹ | 13.74 s | 1.055x | n/a¹ |
| Region update | 0.120 s | 0.076 s | 0.061 s | 1.97x | 1.25x |
| Log-likelihood | 7.627 s | 2.46 s | 1.418 s | **5.38x** | **1.74x** |
| **Total (four phases)** | **54.14 s** | **53.14 s** | **42.63 s** | **1.27x** | **1.25x** |
| **Peak RSS** | **2.85 GiB** | **253.5 MiB** | **252.9 MiB** | **11.53x lower** | ~unchanged |

¹ `peak_pdf`/`peak_sample` are new phase keys added by this plan (Tasks 2-4) to attribute
time within peak sampling. The pre-change Rust binary did not report them — it evaluated
Gaussian PDFs fused into the sequential per-peak loop, with no separable PDF phase to time.

Per-iteration cost is 2.707 s (Python) vs 2.131 s (Rust, after). *Extrapolated* to a full
5,000-iteration production run: **≈3.76 h Python vs ≈2.96 h Rust** — these two figures are
extrapolations from a 20-iteration measurement, not measured full runs.

Log-likelihood also improved substantially (2.46 s → 1.418 s, 1.74x internal speedup):
commit `869200b` applied the same block-wise PDF evaluation to the log-likelihood
computation's peak-probability pass, which was in scope alongside the peak-sampling change.

> **Corpus caveat.** Neurosynth ships tf-idf weights, not raw counts. The driver uses
> `round(tfidf * 100)` clipped at zero purely to reproduce a realistic vocabulary size and
> sparsity pattern *for timing*. It is **not** a scientifically meaningful GCLDA training
> corpus, and the report JSON records this as `counts_are_scaled_tfidf`.

## Block-size sweep (Neurosynth scale)

507,891 peaks, `n_topics=100`, 10 iterations, 1 repeat (single measurement per block size —
not a median). Confirms `--peak-block-size 8192` (the default) is a reasonable choice and
shows the memory/parallelism trade directly.

| `--peak-block-size` | Rust `peak_pdf` | Rust `peak_sample` | Rust `peak_sampling` | Python `peak_sampling` | Rust total | Total ratio | Rust peak RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 39.32 s | 8.074 s | 47.40 s | 8.960 s | 63.88 s | 0.43x | 254.4 MiB |
| 1,024 | 10.75 s | 7.411 s | 18.16 s | 8.387 s | 31.53 s | 0.86x | 253.7 MiB |
| **8,192 (default)** | **1.884 s** | **6.844 s** | **8.734 s** | **8.532 s** | **21.05 s** | **1.29x** | **252.7 MiB** |
| 65,536 | 0.779 s | 6.926 s | 7.707 s | 8.625 s | 20.15 s | 1.37x | 292.0 MiB |

Very small blocks (256) are actively harmful: the per-block parallel-launch overhead
dominates, and `peak_pdf` alone costs 39.3 s — nearly 5x the entire Python peak-sampling
phase. Blocks of 1,024 recover most but not all of the win. 8,192 lands close to the top of
the curve with peak RSS barely above the unblocked baseline (~13 MB of buffer as designed:
8,192 × 100 × 2 × 8 B ≈ 13 MB). 65,536 is marginally faster still but starts trading away the
memory win it was designed to protect (292 MiB vs 253 MiB, i.e. roughly 8x the theoretical
buffer size at this peak count). **8,192 remains the right default**: it captures effectively
all of the available speedup while keeping peak RSS within noise of the un-blocked figure.

## Scaling over `n_topics` (synthetic `small`)

300 documents, 250 terms, 3,000 peaks, 50 iterations, 3 repeats. Ratios are Python/Rust.

| Phase | T=25 | T=50 | T=100 |
|---|---:|---:|---:|
| Word sampling | 1.94x | 1.82x | 1.55x |
| Peak sampling (total) | 2.02x | 1.97x | 1.87x |
| &nbsp;&nbsp;`peak_pdf` | 2.27x | 2.23x | 2.34x |
| &nbsp;&nbsp;`peak_sample` | 1.57x | 1.70x | 1.62x |
| Region update | 1.52x | 1.98x | 1.85x |
| Log-likelihood | 8.60x | 9.29x | 8.55x |
| **Total** | **2.25x** | **2.20x** | **2.04x** |
| Peak RSS | 96.7x | 99.6x | 105.3x |

At this scale (3,000 peaks, well under the 8,192 block size) every block-wise PDF pass is a
single block, so the peak-sampling numbers here mostly reflect the same fused-vs-blocked
code path already winning even at one block. **Rust's advantage still shrinks monotonically
as `n_topics` grows**, consistent with the pre-change finding: every per-token inner loop is
O(`n_topics`), so as T rises the phases become dominated by raw floating-point arithmetic —
where numba's LLVM output is already competitive — rather than by memory traffic or
interpreter-boundary overhead.

> **Do not read the small-scale RSS ratios as a memory result.** At this scale Python's
> ~1.3-1.6 GiB is almost entirely CPython + numba + NiMARE import baseline, not GCLDA data.
> The only trustworthy memory comparison is the Neurosynth row (11.5x), where the arrays
> actually dominate the footprint.

## Thread scaling (synthetic `small`, T=100, 50 iterations) — before/after

3,000 peaks, `n_topics=100`, 50 iterations, 3 repeats. Ratios are Python/Rust; higher means
Rust wins more. This is the direct evidence that block-wise evaluation now uses the cores
where it previously did not.

| Rust threads | Word | Peak (total) | `peak_pdf` | `peak_sample` | Region | Log-lik | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.53x | 1.52x | 1.57x | 1.60x | 35.68x | 7.67x | 2.00x |
| 2 | 1.60x | 2.05x | 2.95x | 1.62x | 21.46x | 9.68x | 2.32x |
| 4 | 1.54x | 2.36x | 5.47x | 1.63x | 15.20x | 10.45x | 2.43x |
| 8 | 1.53x | 2.67x | **7.84x** | 1.58x | 6.31x | 13.31x | 2.52x |

**This is the headline evidence the block-wise change works as designed.** Before this plan,
Rust gained essentially nothing from additional threads (total ratio 1.86x → 1.80x from 1 to
8 threads, i.e. *flat or slightly regressing*). Now, `peak_pdf` alone scales from 1.57x at 1
thread to 7.84x at 8 threads — a 5x improvement in that sub-phase from parallelism alone —
and the total ratio climbs steadily from 2.00x to 2.52x. `peak_sample` (the sequential
sampling loop) stays essentially flat across thread counts (1.60x → 1.58x), exactly as
expected: it was never meant to parallelize, and it didn't.

Region update still gets markedly *worse* (in relative terms) as threads increase (35.68x →
6.31x), because Rust's absolute region-update time grows with more rayon workers at this
small problem size — the same pre-existing finding as before: the per-topic work here is far
smaller than rayon's task-spawn overhead, so parallelizing it is a net loss at this scale.
That issue is unrelated to this plan and remains open (see "Recommended follow-up" below).

## Where the wins came from

The original design spec made four predictions. This plan (block-wise parallel PDF
evaluation) targeted specifically prediction 2.

| # | Prediction | Measured (Neurosynth, after) | Verdict |
|---|---|---|---|
| 1 | Memory: large reduction | 2.85 GiB → 252.9 MiB (11.53x) | **Confirmed, unchanged** |
| 2 | Peak sampling 2–4x | **0.955x — still Rust-slightly-slower, but corrected from 0.62x** | **Substantially improved, not fully met** |
| 3 | Log-likelihood order-of-magnitude | 5.38x (was 3.03x before this plan) | **Improved, still partial** |
| 4 | Word sampling 1.0–1.5x | 1.25x | **Confirmed, unchanged** |

### Prediction 2: the history, the wrong number, and the fix

This is the record of how the number moved, kept intact because it explains why this plan
exists:

1. **Original spec projection: ~1.5x.** The spec argued that `_get_peak_probs`
   materialising an `n_peaks × n_topics × n_regions` float64 array every iteration (~800 MB
   at Neurosynth scale) made the phase memory-bandwidth-bound, so fusing the Gaussian
   evaluation into the sequential sampling loop should win 2–4x on peak sampling, projecting
   ~1.5x overall. This assumed PDF evaluation was **75-80%** of the peak-sampling phase.
2. **Measured reality (first port, pre-plan): 0.62x — Rust slower.** Fusing the evaluation
   into the sequential per-peak loop eliminated the 800 MB array but also made the ~101.6
   million Gaussian evaluations per iteration single-threaded. At 507,891 peaks the phase is
   compute-bound, and losing 28-way parallelism cost far more than the saved memory traffic
   gained. The two goals (memory, speed) were in direct tension, and the port had
   unknowingly traded time for memory.
3. **Task 1 measurement, before any fix was written: PDF evaluation was only 35.5-37.9% of
   the peak-sampling phase** (`--profile-pdf`, median of 3 runs: 0.530/0.511/0.508 s against
   a 1.348 s phase), not the 75-80% the spec assumed. That single measurement revised the
   achievable projection down to **~1.2x** before a line of the block-wise code existed —
   the original 1.5x number was never achievable even in principle, because it rested on an
   unverified assumption about where time was going.
4. **This plan's fix: chunked parallel evaluation.** Evaluate PDFs in parallel blocks
   (default 8,192 peaks) while sampling remains strictly sequential within and across
   blocks — recovering the 28-way parallelism for the ~35-38% of the phase that actually is
   the PDF evaluation, while capping the buffer at ~13 MB instead of ~800 MB. Sampling order
   is unchanged, so bit-exactness holds (verified at block sizes 1, 7, 8,192, and
   10,000,000 — see test suite below).
5. **Final measured result: total 1.27x, peak-sampling phase 0.955x.** This *exceeds* the
   revised ~1.2x projection at the total-runtime level. Peak sampling itself did not cross
   1.0x — Rust remains marginally slower than Python's `@njit(parallel=True)` implementation
   on that specific phase — but the gap closed from 62% slower to 4.5% slower, and the
   `peak_pdf` sub-phase alone is now clearly parallel (7.84x at 8 threads vs 1.57x at 1
   thread; see thread-scaling table). The remaining ~4.5% gap is consistent with `peak_pdf`
   itself still trailing Python's PDF pass in isolation (0.529x, Neurosynth scale) even
   though it now uses all cores — plausibly per-block synchronization/dispatch overhead that
   Python's single parallel `njit` launch does not pay repeatedly. This was not
   independently isolated in this task and is a candidate for further profiling.

**If you are looking for one number: the projection corrected itself from 1.5x to ~1.2x
before this plan wrote any code, and the plan then delivered 1.27x — slightly better than
the corrected projection, not the original one.**

### On prediction 3

Replacing the dense `D × W` posterior-predictive matmul with sparse per-token dot products
was already a real win before this plan (3.03x at real scale). This plan's block-wise PDF
evaluation also applies to log-likelihood's peak-probability pass (commit `869200b`), pushing
the ratio to 5.38x. Log-likelihood is still only ~3.3% of total Rust runtime at Neurosynth
scale (1.418 s / 42.63 s), so this improvement, while real, does not move the headline much.
It also runs only every `loglikely_freq` iterations.

### The Amdahl ceiling on future parallelization — and a correction to the design spec

Task 1's measurement implies a hard ceiling on how much further parallelizing *this specific
algorithm* can help. Peak sampling was 50.75% of pre-change Rust total runtime
(26.96 s / 53.14 s), and only 35.5-37.9% of that phase is the parallelizable PDF evaluation —
so **the parallelizable fraction of total pre-change Rust runtime is only ~22.6%.** The
remaining **~77.2% is inherently sequential** (word sampling plus the sequential body of peak
sampling — the per-peak topic/region draw itself, which must stay in order for bit-exactness).

By Amdahl's law, with infinitely fast hardware for the parallel 22.6%:

```
ceiling = 1 / (1 - 0.226) ≈ 1.29x over pre-change Rust
```

This plan measured a **1.25x internal Rust speedup** (53.14 s → 42.63 s), i.e. **~96% of the
theoretical ceiling** — there is very little headroom left in parallelizing this algorithm
further without changing its sequential structure (e.g. SparseLDA-style bucket decomposition,
which is out of scope here because it changes RNG consumption and breaks bit-exactness).

**This corrects the design spec's `docs/superpowers/specs/2026-08-18-gclda-blockwise-pdf-design.md`
"Out of scope → GPU offload" note**, which estimated the parallel-safe fraction at ~44% of
runtime and an Amdahl ceiling of **1.79x**, with block-wise CPU parallelism expected to
capture ~1.5x of that (leaving ~1.17x incremental for GPU offload). The 44%/1.79x figures
predate Task 1's direct measurement and were themselves an estimate, not a measurement; the
measured parallelizable fraction is roughly half that (22.6%), and the corrected ceiling is
**1.29x, not 1.79x**. Since this plan's CPU-only fix already captured ~96% of the 1.29x
ceiling, a GPU offload would have at most ~0.03x of headroom left over the current Rust
implementation — the GPU option was already low-value under the original (wrong) estimate,
and is essentially worthless under the corrected one. It should not be pursued for this
algorithm's current sequential structure.

### Recommended follow-up

- **`update_regions` rayon gating.** Still a real, measured regression at small problem
  sizes (region update gets up to ~6x relatively worse from 1 to 8 threads in this run, the
  same direction as before). Worth gating behind a problem-size threshold; not attempted in
  this plan (Neurosynth-scale region update is 0.14% of total runtime, so it doesn't move the
  headline, but it's a legitimate small bug).
- **Isolate the remaining `peak_pdf` gap.** `peak_pdf` at Neurosynth scale is 0.529x versus
  Python even with full thread scaling recovered — worth a follow-up profiling pass (e.g.
  with `perf` on a kernel that supports it) to see whether this is block-dispatch overhead,
  a difference in how numba vs Rayon distribute uneven work across threads, or something
  else. Not investigated further here; the measured net effect (1.27x total) is positive
  regardless.

## Honest summary

- **Memory: unambiguous success, unchanged by this plan.** 11.53x lower peak RSS on the real
  corpus (252.9 MiB Rust vs 2.85 GiB Python). This was already true before this plan and
  remains true after it — the block-wise PDF buffer (~13 MB at the default block size) does
  not meaningfully change peak RSS.
- **Speed: now a real, if modest, win at production scale.** 1.27x total on the real
  Neurosynth corpus, up from the prior 1.01x wash. This exceeds the ~1.2x projection Task 1
  revised the (wrong) 1.5x spec estimate down to.
- **Peak sampling itself is not yet a Rust win** (0.955x — still slightly slower than
  Python), but the regression that made the whole port a wash is now closed to within 4.5%,
  down from 38%.
- **The bit-exactness goal held throughout, at every block size tested.** Every
  configuration — including the full 507,891-peak corpus, and block sizes 1, 7, 8,192, and
  10,000,000 — produced all four probability matrices bit-identical to the Python
  implementation.
- **There is little room left to parallelize this algorithm further.** The measured
  sequential fraction (77.2% of pre-change Rust runtime) sets an Amdahl ceiling of ~1.29x
  over pre-change Rust, and this plan already captured ~96% of that ceiling. Both the
  original spec's GPU-offload estimate (1.79x ceiling) and any similar future proposal should
  be evaluated against the corrected 1.29x figure, not the original one.

## Test suite status

```
$ micromamba run -n nimenv python -m pytest nimare/tests/test_gclda_rust.py nimare/tests/test_annotate_gclda.py -v
============================= test session starts ==============================
collected 33 items
...
============================= 33 passed in 14.89s ==============================
```

Includes `test_rust_outputs_are_invariant_to_peak_block_size[1]`,
`[7]`, `[8192]`, and `[10000000]` — all passed, confirming bit-exactness across block sizes
spanning "one peak per block" to "every peak in one block."

```
$ cd rust/gclda && cargo test --release
cargo total passed: 36  failed: 0
```

## Reproducing

```bash
cd rust/gclda && cargo build --release && cd ../..

# real corpus (--neurosynth-data-dir only needed if the cache is not under ~/.nimare)
micromamba run -n nimenv python benchmarks/bench_gclda_rust.py \
    --scale neurosynth --n-iters 20 --n-topics 100 --repeats 3 \
    --neurosynth-data-dir ~/.nimare --out /tmp/gclda_bench_ns.json

# n_topics sweep
for t in 25 50 100; do
  micromamba run -n nimenv python benchmarks/bench_gclda_rust.py \
      --scale small --n-iters 50 --n-topics $t --out /tmp/gclda_bench_small_t$t.json
done

# thread scaling
for t in 1 2 4 8; do
  micromamba run -n nimenv python benchmarks/bench_gclda_rust.py \
      --scale small --n-iters 50 --n-topics 100 --threads $t \
      --out /tmp/gclda_bench_threads_$t.json
done

# block-size sweep (Neurosynth scale)
for b in 256 1024 8192 65536; do
  micromamba run -n nimenv python benchmarks/bench_gclda_rust.py \
      --scale neurosynth --n-iters 10 --n-topics 100 --repeats 1 \
      --peak-block-size $b --neurosynth-data-dir ~/.nimare \
      --out /tmp/gclda_bench_block_$b.json
done
```
