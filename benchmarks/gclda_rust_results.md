# GCLDA Rust port — measured benchmark results

All numbers below were produced by `benchmarks/bench_gclda_rust.py`, which verifies that
both implementations produced **bit-identical** `p_topic_g_voxel`, `p_voxel_g_topic`,
`p_topic_g_word`, and `p_word_g_topic` *before* it records any timing. Every configuration
reported here passed that check, so no timing below credits a run that computed a different
answer.

## Headline

**On the real Neurosynth corpus the Rust port is a wall-clock wash (1.01x) and an
11.5x reduction in peak memory.** The memory goal was met decisively. The speed goal
was not: the phase the port was specifically designed to accelerate — peak sampling —
is measurably *slower* in Rust at real scale. Details and cause in
"Where the wins came from" below.

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

## Real corpus — Neurosynth v7

14,371 documents, 3,228 terms, **507,891 peaks**, `n_topics=100`, `n_regions=2`,
symmetric, 20 iterations, 3 repeats, all 28 threads. Medians shown.

| Phase | Python | Rust | Ratio |
|---|---:|---:|---:|
| Word sampling | 29.54 s | 23.65 s | **1.25x** |
| Peak sampling | 16.68 s | 26.96 s | **0.62x** (Rust slower) |
| Region update | 0.122 s | 0.076 s | 1.62x |
| Log-likelihood | 7.46 s | 2.46 s | **3.03x** |
| **Total (four phases)** | **53.80 s** | **53.14 s** | **1.01x** |
| **Peak RSS** | **2.85 GiB** | **253.5 MiB** | **11.5x lower** |

Per-iteration cost is 2.690 s (Python) vs 2.657 s (Rust). *Extrapolated* to a full
5,000-iteration production run: **≈3.74 h Python vs ≈3.69 h Rust** — these two figures are
extrapolations from a 20-iteration measurement, not measured full runs.

> **Corpus caveat.** Neurosynth ships tf-idf weights, not raw counts. The driver uses
> `round(tfidf * 100)` clipped at zero purely to reproduce a realistic vocabulary size and
> sparsity pattern *for timing*. It is **not** a scientifically meaningful GCLDA training
> corpus, and the report JSON records this as `counts_are_scaled_tfidf`.

## Scaling over `n_topics` (synthetic `small`)

300 documents, 250 terms, 3,000 peaks, 50 iterations, 3 repeats. Ratios are Python/Rust.

| Phase | T=25 | T=50 | T=100 |
|---|---:|---:|---:|
| Word sampling | 1.96x | 1.83x | 1.67x |
| Peak sampling | 3.46x | 2.35x | 1.70x |
| Region update | 1.68x | 1.72x | 2.14x |
| Log-likelihood | 14.69x | 9.31x | 9.19x |
| **Total** | **3.02x** | **2.42x** | **2.08x** |
| Peak RSS | 90.2x | 103.9x | 115.0x |

**Rust's advantage shrinks monotonically as `n_topics` grows.** Every per-token inner loop
is O(`n_topics`), so as T rises the phases become dominated by raw floating-point
arithmetic — where numba's LLVM output is already competitive — rather than by the memory
traffic and interpreter-boundary overheads Rust removes.

> **Do not read the small-scale RSS ratios as a memory result.** At this scale Python's
> ~1.6 GiB is almost entirely CPython + numba + NiMARE import baseline, not GCLDA data.
> The only trustworthy memory comparison is the Neurosynth row (11.5x), where the arrays
> actually dominate the footprint.

## Thread scaling (synthetic `small`, T=100, 50 iterations)

| Rust threads | Word | Peak | Region | Log-lik | Total |
|---|---:|---:|---:|---:|---:|
| 1 | 1.45x | 1.45x | **20.89x** | 6.68x | 1.86x |
| 2 | 1.55x | 1.46x | 22.55x | 6.44x | 1.93x |
| 4 | 1.44x | 1.21x | 14.71x | 6.70x | 1.74x |
| 8 | 1.49x | 1.51x | **4.95x** | 6.11x | 1.80x |

**Rust gains essentially nothing from additional threads** (total 1.86x → 1.80x from 1 to 8),
which is expected: collapsed Gibbs sampling is sequential per token, so only region updates
and the final voxel distributions are parallelisable at all.

More importantly, **region update gets 4x *worse* going from 1 thread to 8** (20.89x → 4.95x).
At this problem size the per-topic work is far smaller than rayon's task overhead, so
parallelising it is a net loss. The rayon parallelism in `update_regions` should be gated
behind a problem-size threshold rather than applied unconditionally.

## Where the wins came from

The design spec made four predictions. Measured against the real corpus:

| # | Prediction | Measured (Neurosynth) | Verdict |
|---|---|---|---|
| 1 | Memory: large reduction | 2.85 GiB → 253.5 MiB (11.5x) | **Confirmed** |
| 2 | Peak sampling 2–4x | **0.62x — Rust slower** | **Wrong** |
| 3 | Log-likelihood order-of-magnitude | 3.03x (9–15x at small scale) | **Partial** |
| 4 | Word sampling 1.0–1.5x | 1.25x | **Confirmed** |

### Prediction 2 was wrong, and why

The spec argued that `_get_peak_probs` materialising an `n_peaks × n_topics × n_regions`
float64 array every iteration (~800 MB at Neurosynth scale) made the phase
memory-bandwidth-bound, so fusing the Gaussian evaluation into the sequential sampling loop
should win 2–4x. Measured, that fusion **costs** 62% more time than Python.

The error was in the "bandwidth-bound" premise. Python's `_get_peak_probs` is
`@njit(parallel=True)`: it computes all 507,891 × 100 × 2 ≈ **101.6 million** Gaussian
evaluations across **all 28 cores**, then streams the resulting array through a sequential
sampler. Fusing that evaluation into the per-peak sampling loop eliminated the array — but
because the sampling loop is inherently sequential, it also made those 101.6 M evaluations
**single-threaded**. At this scale the phase is *compute*-bound, and losing 28-way
parallelism costs far more than the saved memory traffic gains.

This is consistent with the `n_topics` sweep: at 3,000 peaks the array is small, Python's
parallel launch overhead is not amortised, and fusion wins 3.46x. The advantage decays as
the array grows, and at 507,891 peaks it inverts.

The memory saving from fusion is real and is most of the 11.5x RSS win. The two goals were
in direct tension here, and the port traded time for memory without that trade being
recognised at design time.

### Recommended fix: chunked parallel evaluation

Fusion and parallelism are not actually mutually exclusive. Processing peaks in blocks of a
few thousand — evaluating one block's PDFs in parallel (the evaluations are
order-independent), then sampling sequentially within the block — would recover the 28-way
parallelism while capping the buffer at roughly 13 MB (8,192 × 100 × 2 × 8 B) instead of
800 MB. Sampling order is unchanged, so **bit-exactness is preserved**.

This is a design change beyond the scope of the port as specified, so it has not been
implemented. It is the single highest-value follow-up, and it is what would turn the
current 1.01x wash into a genuine speedup.

### On prediction 3

Replacing the dense `D × W` posterior-predictive matmul with sparse per-token dot products
is a real and consistent win, but it is 3.03x at real scale rather than the 9–15x seen on
small corpora. Log-likelihood is only ~14% of total runtime at Neurosynth scale, so this
does not move the headline much. It also runs only every `loglikely_freq` iterations.

## Honest summary

- **Memory: unambiguous success.** 11.5x lower peak RSS on the real corpus, and the Rust
  trainer fits comfortably in 254 MiB where Python needs 2.85 GiB. On memory-constrained
  machines this alone can be the difference between running and not running.
- **Speed: no improvement at production scale.** 1.01x total. Two phases improved (word
  sampling 1.25x, log-likelihood 3.03x) and one regressed enough to cancel them out
  (peak sampling 0.62x).
- **The bit-exactness goal held throughout.** Every configuration — including the full
  507,891-peak corpus — produced all four probability matrices bit-identical to the
  Python implementation. That is the port's strongest result and what makes the
  regression above safely fixable.
- **Smaller corpora do benefit** (2–3x total at the `small` scale), so the port is not
  without speed value; it simply does not deliver one where it was most wanted.

## Test suite status

```
$ micromamba run -n nimenv pytest nimare/tests/test_gclda_rust.py nimare/tests/test_annotate_gclda.py -q
28 passed in 11.55s

$ cd rust/gclda && cargo test --release
cargo total passed: 33  failed: 0
```

## Reproducing

```bash
cd rust/gclda && cargo build --release && cd ../..

# n_topics sweep
for t in 25 50 100; do
  python benchmarks/bench_gclda_rust.py --scale small --n-iters 50 --n-topics $t \
      --out /tmp/gclda_bench_small_t$t.json
done

# real corpus (--neurosynth-data-dir only needed if the cache is not under ~/.nimare)
python benchmarks/bench_gclda_rust.py --scale neurosynth --n-iters 20 --n-topics 100 \
    --neurosynth-data-dir ~/.nimare --out /tmp/gclda_bench_ns.json

# thread scaling
for t in 1 2 4 8; do
  python benchmarks/bench_gclda_rust.py --scale small --n-iters 50 --n-topics 100 \
      --threads $t --out /tmp/gclda_bench_threads_$t.json
done
```
