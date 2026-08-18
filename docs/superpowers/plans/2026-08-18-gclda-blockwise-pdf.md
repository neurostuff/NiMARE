# GCLDA Block-wise Parallel PDF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the peak-sampling performance the GCLDA Rust port lost to serialization, without giving up bit-exact reproducibility against Python, and add an opt-in float32 compute path.

**Architecture:** Evaluate Gaussian PDFs one block of peaks at a time using rayon (the evaluations are order-independent), then sample each block sequentially (the sampling is not). Sampling order and RNG consumption are untouched, so f64 results stay bit-identical to Python. A `Float` trait makes the PDF and weight arithmetic generic so an `f32` path can share one implementation rather than duplicating the sampler.

**Tech Stack:** Rust (rayon, clap), Python 3.13 (numpy, numba, pandas), micromamba env `nimenv`.

**Spec:** `docs/superpowers/specs/2026-08-18-gclda-blockwise-pdf-design.md`

## Global Constraints

- **The `f64` path must remain bit-identical to Python.** The existing Level 2 (per-iteration state equality, all four region configurations) and Level 3 (end-to-end outputs) suites are the acceptance gate for every task. If they fail, the task is not done.
- **Never reassociate a floating-point reduction.** Parallelism is permitted only across disjoint peaks. `sample_from_unnormalized`'s accumulation and every per-peak arithmetic sequence stay exactly as they are.
- **Region parameters (`regions_mu`, `regions_sigma`, `regions_precision`, `regions_log_norm`) stay `f64` always**, including under `--compute-dtype f32`.
- Default `--peak-block-size` is `8192`. Default `--compute-dtype` is `f64`.
- Rayon parallel fill threshold: `PARALLEL_MIN_EVALS = 32_768` (`len * n_topics * n_regions`).
- Python commands run as `micromamba run -n nimenv <cmd>` from the repo root.
- Rust commands run from `rust/gclda`.
- Commit messages end with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- **Do not `git add -A`.** The working tree carries ~220 files of unrelated CRLF line-ending churn. Stage only the exact paths each task names.
- Nothing is pushed.

---

## File Structure

| File | Responsibility |
|---|---|
| `rust/gclda/src/gaussian.rs` | 3x3 inverse/logdet; PDF evaluation, made generic over the float type |
| `rust/gclda/src/float.rs` | **New.** Private `Float` trait abstracting the arithmetic used by PDF and weight building |
| `rust/gclda/src/sampler/peaks.rs` | `peak_probs_for`, new `peak_probs_block`, blocked `update_peak_assignments` |
| `rust/gclda/src/loglik.rs` | Sparse log-likelihood; consumes the block primitive |
| `rust/gclda/src/model.rs` | `Model`, `Params`, `PhaseTimes` (gains `peak_pdf` / `peak_sample`) |
| `rust/gclda/src/output.rs` | `model.json` writer (gains the two new phase keys) |
| `rust/gclda/src/bin/gclda-train.rs` | CLI: `--peak-block-size`, `--compute-dtype`, `--profile-pdf` |
| `nimare/annotate/gclda.py` | Matching `peak_pdf` / `peak_sample` timing keys |
| `nimare/annotate/gclda_rs.py` | Pass new parameters through `train_gclda_rust` |
| `nimare/tests/test_gclda_rust.py` | Block-size invariance, f32 tolerance, phase-key parity |
| `benchmarks/bench_gclda_rust.py` | Expose the new flags |
| `benchmarks/gclda_rust_results.md` | Updated measurements |

---

### Task 1: Measure the PDF share of the peak-sampling phase (gate)

The whole plan rests on the claim that PDF evaluation is 75-80% of the peak-sampling phase, which was derived from an `exp()` cost model, not measured. This task measures it before any optimization work happens.

A per-peak timer inside the current fused loop would add ~500k `Instant::now()` pairs per iteration (~2% perturbation), so instead this adds a one-shot diagnostic pass that times serial PDF evaluation over every peak directly.

**Files:**
- Modify: `rust/gclda/src/bin/gclda-train.rs`
- Modify: `rust/gclda/src/sampler/peaks.rs`
- Test: `rust/gclda/tests/profile_pdf.rs`

**Interfaces:**
- Consumes: `Model::peak_probs_for(&self, i_peak: usize, out: &mut [f64])` (existing)
- Produces: `Model::time_serial_pdf_pass(&self) -> f64` — evaluates `peak_probs_for` for every peak into one reusable buffer and returns elapsed seconds. Used only by the `--profile-pdf` diagnostic.

- [ ] **Step 1: Write the failing test**

Create `rust/gclda/tests/profile_pdf.rs`:

```rust
//! `time_serial_pdf_pass` must actually evaluate every peak and report a
//! positive duration, so the Task 1 gate measurement is trustworthy.

use gclda::io::{nifti::load_mask_xyz, tsv::load_corpus};
use gclda::model::{Model, Params};
use std::path::PathBuf;

mod common;
use common::load;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

/// Build a model from the committed fixtures with region Gaussians populated.
/// Mirrors the construction in `tests/sampler_peaks.rs` -- `common/mod.rs`
/// exposes only `repo_path`/`load`/`bits_to_f64`, no model builder.
fn fixture_model() -> Model {
    let mask_meta = load("mask_xyz.json");
    let mask_path = common::repo_path(mask_meta["path"].as_str().unwrap());
    let corpus = load_corpus(&fixture("counts.tsv"), &fixture("coordinates.tsv")).unwrap();
    let mask = load_mask_xyz(&mask_path).unwrap();
    let params = Params {
        n_topics: 3,
        n_regions: 2,
        symmetric: true,
        alpha: 0.1,
        beta: 0.01,
        gamma: 0.01,
        delta: 1.0,
        dobs: 25.0,
        roi_size: 50.0,
        seed_init: 1,
    };
    let mut model = Model::new(corpus, mask, params).unwrap();
    model.update_regions().unwrap();
    model
}

#[test]
fn serial_pdf_pass_reports_positive_time() {
    let model = fixture_model();
    let seconds = model.time_serial_pdf_pass();
    assert!(
        seconds > 0.0,
        "serial PDF pass reported {seconds} seconds; timer is not measuring anything"
    );
}
```

`common/mod.rs` is NOT modified by this task — it exposes only `repo_path`,
`load`, and `bits_to_f64`, and this test needs no addition to it.

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --release --test profile_pdf`
Expected: FAIL to compile — `no method named time_serial_pdf_pass`.

- [ ] **Step 3: Implement the diagnostic pass**

In `rust/gclda/src/sampler/peaks.rs`, inside `impl Model`:

```rust
/// Time one serial pass of `peak_probs_for` over every peak, into a single
/// reusable buffer. Diagnostic only: used by `--profile-pdf` to measure what
/// share of the peak-sampling phase is PDF evaluation, which is the number
/// the block-wise parallelization decision is gated on. Not called during
/// normal training.
pub fn time_serial_pdf_pass(&self) -> f64 {
    let n_topics = self.params.n_topics;
    let n_regions = self.params.n_regions;
    let mut buf = vec![0.0f64; n_topics * n_regions];
    let start = std::time::Instant::now();
    for i_peak in 0..self.corpus.ptoken_coords.len() {
        self.peak_probs_for(i_peak, &mut buf);
    }
    // Consume `buf` so the optimizer cannot eliminate the loop entirely.
    std::hint::black_box(&buf);
    start.elapsed().as_secs_f64()
}
```

- [ ] **Step 4: Add the CLI flag**

In `rust/gclda/src/bin/gclda-train.rs`, add to `struct Args`:

```rust
    /// Diagnostic: before training, time one serial pass of Gaussian PDF
    /// evaluation over every peak and print the result, then continue.
    /// Used to measure what fraction of the peak-sampling phase is PDF
    /// evaluation. Not part of normal training.
    #[arg(long, default_value_t = false)]
    profile_pdf: bool,
```

After the `Model` is constructed and before `model.fit(...)` is called:

```rust
    if args.profile_pdf {
        let seconds = model.time_serial_pdf_pass();
        println!("profile_pdf: serial_pdf_pass_seconds={seconds:.6}");
    }
```

- [ ] **Step 5: Run to verify it passes**

Run: `cargo test --release --test profile_pdf`
Expected: PASS.

- [ ] **Step 6: Take the gate measurement**

```bash
cd rust/gclda && cargo build --release && cd ../..
rust/gclda/target/release/gclda-train \
  --counts <staged neurosynth counts.tsv> \
  --coordinates <staged neurosynth coordinates.tsv> \
  --mask nimare/resources/templates/MNI152_2x2x2_brainmask.nii.gz \
  --out-dir /tmp/gclda_profile_out \
  --n-topics 100 --n-iters 1 --profile-pdf
```

Stage the Neurosynth TSVs first, if not already present, with:

```bash
micromamba run -n nimenv python benchmarks/bench_gclda_rust.py \
    --scale neurosynth --n-iters 1 --n-topics 10 --repeats 1 \
    --neurosynth-data-dir /mnt/c/Users/tsalo/.nimare \
    --stage-dir /tmp/gclda_stage --out /tmp/gclda_stage.json
```

Compute `serial_pdf_pass_seconds / 1.348` — 1.348 s/iter is the measured Rust peak-sampling cost from `benchmarks/gclda_rust_results.md` at T=100.

- [ ] **Step 7: Apply the gate**

- Ratio **>= 0.70**: proceed to Task 2.
- Ratio **0.50-0.70**: proceed, but record the revised projection in the task report — the ~1.5x estimate is optimistic.
- Ratio **< 0.50**: **STOP.** Report the measured value and the revised projection (~1.3x or less) and ask whether to continue before writing any more code.

- [ ] **Step 8: Commit**

```bash
git add rust/gclda/src/bin/gclda-train.rs rust/gclda/src/sampler/peaks.rs \
        rust/gclda/tests/profile_pdf.rs
git diff --cached --stat
git commit -m "[ENH] Add --profile-pdf diagnostic to measure PDF share of peak sampling

The block-wise parallelization decision is gated on what fraction of the
peak-sampling phase is Gaussian PDF evaluation. A per-peak timer inside
the fused loop would perturb the measurement by ~2%, so this times one
serial pass over all peaks instead.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: `peak_probs_block` primitive

**Files:**
- Modify: `rust/gclda/src/sampler/peaks.rs`
- Test: `rust/gclda/tests/peak_probs_block.rs`

**Interfaces:**
- Consumes: `Model::peak_probs_for(&self, i_peak: usize, out: &mut [f64])`
- Produces: `Model::peak_probs_block(&self, start: usize, len: usize, out: &mut [f64])` — fills `out[0 .. len * n_topics * n_regions]` with densities for peaks `[start, start + len)`. Element at `(local_peak * n_topics + topic) * n_regions + region`. Also `Model::peak_probs_block_forced(&self, start: usize, len: usize, out: &mut [f64], parallel: bool)` and `pub const PARALLEL_MIN_EVALS: usize = 32_768;`

- [ ] **Step 1: Write the failing test**

Create `rust/gclda/tests/peak_probs_block.rs`:

```rust
//! `peak_probs_block` must be an exact, bit-for-bit restatement of repeated
//! `peak_probs_for` calls, and its sequential and rayon fill paths must agree
//! with each other. Any divergence breaks bit-exactness against Python, so
//! this compares bit patterns, not approximate equality.

use gclda::io::{nifti::load_mask_xyz, tsv::load_corpus};
use gclda::model::{Model, Params};
use std::path::PathBuf;

mod common;
use common::load;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

/// Build a model from the committed fixtures, with region Gaussians populated
/// by one `update_regions` call so `peak_probs_for` has real parameters to
/// evaluate against. Mirrors the construction in `tests/sampler_peaks.rs`.
fn fixture_model() -> Model {
    let mask_meta = load("mask_xyz.json");
    let mask_path = common::repo_path(mask_meta["path"].as_str().unwrap());
    let corpus = load_corpus(&fixture("counts.tsv"), &fixture("coordinates.tsv")).unwrap();
    let mask = load_mask_xyz(&mask_path).unwrap();
    let params = Params {
        n_topics: 3,
        n_regions: 2,
        symmetric: true,
        alpha: 0.1,
        beta: 0.01,
        gamma: 0.01,
        delta: 1.0,
        dobs: 25.0,
        roi_size: 50.0,
        seed_init: 1,
    };
    let mut model = Model::new(corpus, mask, params).unwrap();
    model.update_regions().unwrap();
    model
}

fn check_block(model: &Model, start: usize, len: usize) {
    let stride = model.params.n_topics * model.params.n_regions;

    let mut block = vec![0.0f64; len * stride];
    model.peak_probs_block(start, len, &mut block);

    let mut single = vec![0.0f64; stride];
    for i in 0..len {
        model.peak_probs_for(start + i, &mut single);
        for k in 0..stride {
            let (got, want) = (block[i * stride + k], single[k]);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "peak {} element {k}: block={got:?} single={want:?}",
                start + i
            );
        }
    }
}

#[test]
fn block_matches_per_peak_evaluation() {
    let model = fixture_model();
    let n = model.corpus.ptoken_coords.len();
    assert!(n >= 4, "fixture has {n} peaks; too few to exercise partial blocks");

    check_block(&model, 0, n);      // whole corpus in one block
    check_block(&model, 0, 1);      // single peak
    check_block(&model, 1, n - 1);  // offset start
    check_block(&model, n - 1, 1);  // final peak
}

#[test]
fn sequential_and_parallel_fill_paths_agree() {
    // Crossing PARALLEL_MIN_EVALS honestly would need thousands of peaks,
    // which no committed fixture has. Drive both paths explicitly instead, so
    // the property under test -- that they produce identical bits -- is
    // checked directly rather than inferred from a corpus size.
    let model = fixture_model();
    let n = model.corpus.ptoken_coords.len();
    let stride = model.params.n_topics * model.params.n_regions;

    let mut seq = vec![0.0f64; n * stride];
    let mut par = vec![0.0f64; n * stride];
    model.peak_probs_block_forced(0, n, &mut seq, false);
    model.peak_probs_block_forced(0, n, &mut par, true);

    for k in 0..n * stride {
        assert_eq!(
            seq[k].to_bits(),
            par[k].to_bits(),
            "element {k}: sequential={:?} parallel={:?}",
            seq[k],
            par[k]
        );
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --release --test peak_probs_block`
Expected: FAIL to compile — `no method named peak_probs_block`.

- [ ] **Step 3: Implement the primitive**

In `rust/gclda/src/sampler/peaks.rs`, add the import and the constant near the top:

```rust
use rayon::prelude::*;

/// Minimum number of Gaussian evaluations (`len * n_topics * n_regions`) in a
/// block before it is worth filling with rayon. Below this, task overhead
/// exceeds the work: `update_regions` was measured 4x *slower* at 8 threads
/// than at 1 on small corpora for exactly this reason (see
/// `benchmarks/gclda_rust_results.md`, thread-scaling table).
pub const PARALLEL_MIN_EVALS: usize = 32_768;
```

Then inside `impl Model`:

```rust
/// Fill `out` with `p(x_i | topic, region)` for peaks `[start, start + len)`.
///
/// Layout is `(local_peak * n_topics + topic) * n_regions + region`, so each
/// peak's `n_topics * n_regions` block is contiguous and indexes exactly as
/// [`Model::peak_probs_for`]'s output does.
///
/// # Invariant this relies on
///
/// [`crate::gaussian::pdf`] reads only `corpus.ptoken_coords` and the cached
/// region parameters (`regions_mu`, `regions_precision`, `regions_log_norm`).
/// None of those are mutated by [`Model::update_peak_assignments`], which
/// touches only count matrices and assignment vectors; region parameters are
/// recomputed in the separate `update_regions` phase. **That is what makes it
/// legal to evaluate a whole block up front, before the sequential loop
/// mutates anything.** If a future change moves region-parameter updates
/// inside the sampling sweep, this precomputation becomes wrong and the
/// Level 2 per-iteration equality tests will fail immediately.
///
/// Parallelism is across disjoint peaks with no floating-point reduction, so
/// results are bit-identical to `len` successive `peak_probs_for` calls.
pub fn peak_probs_block(&self, start: usize, len: usize, out: &mut [f64]) {
    let stride = self.params.n_topics * self.params.n_regions;
    let out = &mut out[..len * stride];

    self.peak_probs_block_forced(start, len, out, len * stride >= PARALLEL_MIN_EVALS);
}

/// [`Model::peak_probs_block`] with the sequential/parallel choice forced
/// rather than taken from [`PARALLEL_MIN_EVALS`]. Exists so tests can prove
/// both fill paths produce identical bits without needing a fixture large
/// enough to cross the threshold naturally.
pub fn peak_probs_block_forced(&self, start: usize, len: usize, out: &mut [f64], parallel: bool) {
    let stride = self.params.n_topics * self.params.n_regions;
    let out = &mut out[..len * stride];

    if parallel {
        out.par_chunks_mut(stride).enumerate().for_each(|(i, chunk)| {
            self.peak_probs_for(start + i, chunk);
        });
    } else {
        for (i, chunk) in out.chunks_mut(stride).enumerate() {
            self.peak_probs_for(start + i, chunk);
        }
    }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --release --test peak_probs_block`
Expected: PASS, both tests.

- [ ] **Step 5: Run the full Rust suite**

Run: `cargo test --release`
Expected: all PASS (33 existing + 2 new).

- [ ] **Step 6: Commit**

```bash
git add rust/gclda/src/sampler/peaks.rs rust/gclda/tests/peak_probs_block.rs
git diff --cached --stat
git commit -m "[ENH] Add peak_probs_block for parallel per-block PDF evaluation

Evaluates a contiguous block of peaks' Gaussian densities with rayon.
Parallelism is across disjoint peaks with no floating-point reduction, so
the result is bit-identical to repeated peak_probs_for calls; the test
compares bit patterns rather than approximate equality.

Falls back to a sequential fill below PARALLEL_MIN_EVALS, since rayon
overhead exceeds the work on small blocks.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Block the peak sampler, add `--peak-block-size` and phase sub-keys

**Files:**
- Modify: `rust/gclda/src/sampler/peaks.rs`
- Modify: `rust/gclda/src/model.rs` (`Params`, `PhaseTimes`)
- Modify: `rust/gclda/src/output.rs`
- Modify: `rust/gclda/src/bin/gclda-train.rs`
- Modify: `nimare/annotate/gclda.py`
- Test: `nimare/tests/test_gclda_rust.py`

**Interfaces:**
- Consumes: `Model::peak_probs_block(&self, start: usize, len: usize, out: &mut [f64])`
- Produces: `Params.peak_block_size: usize`; `PhaseTimes.peak_pdf: f64`, `PhaseTimes.peak_sample: f64`; CLI `--peak-block-size <usize>`; Python `GCLDAModel.phase_times_` keys `"peak_pdf"` and `"peak_sample"`.

Both implementations now report seven phase keys: `word_sampling`, `peak_sampling`, `peak_pdf`, `peak_sample`, `region_update`, `loglikelihood`, `total`. `peak_pdf + peak_sample` approximately equals `peak_sampling`; `peak_sampling` keeps its existing meaning so previously recorded results stay comparable.

- [ ] **Step 1: Write the failing test**

Append to `nimare/tests/test_gclda_rust.py`:

```python
@requires_rust
@pytest.mark.parametrize("block_size", [1, 7, 8192, 10_000_000])
def test_rust_outputs_are_invariant_to_peak_block_size(
    small_corpus, mni_mask, tmp_path, block_size
):
    """Block size must not change a single bit of any output.

    Blocking the peak sampler is only legal because Gaussian evaluation is
    order-independent while sampling is not. The failure modes it introduces
    are off-by-one at block boundaries and mishandling of the partial final
    block, neither of which shows up as a crash -- only as different numbers.
    Block size 7 is deliberately a non-power-of-2 that does not divide the
    peak count evenly; 10_000_000 exceeds the corpus so the whole run is one
    partial block; 1 makes every block partial.
    """
    counts, coords = small_corpus
    mask_path = str(tmp_path / "mask.nii.gz")
    mni_mask.to_filename(mask_path)

    out_dir = str(tmp_path / f"out_{block_size}")
    annotate.gclda_rs.train_gclda_rust(
        counts, coords, mask=mask_path, out_dir=out_dir, binary=BINARY,
        n_topics=4, n_regions=2, symmetric=True, n_iters=5, loglikely_freq=5,
        peak_block_size=block_size,
    )

    reference_dir = str(tmp_path / "out_reference")
    annotate.gclda_rs.train_gclda_rust(
        counts, coords, mask=mask_path, out_dir=reference_dir, binary=BINARY,
        n_topics=4, n_regions=2, symmetric=True, n_iters=5, loglikely_freq=5,
        peak_block_size=8192,
    )

    for name in (
        "p_topic_g_voxel", "p_voxel_g_topic", "p_topic_g_word", "p_word_g_topic",
        "peak_topic_idx", "peak_region_idx", "n_peak_tokens_region_by_topic",
    ):
        got = np.load(os.path.join(out_dir, f"{name}.npy"))
        want = np.load(os.path.join(reference_dir, f"{name}.npy"))
        np.testing.assert_array_equal(
            got, want, err_msg=f"{name} changed with peak_block_size={block_size}"
        )
```

Also update the existing phase-key test in the same file, replacing its `expected` set and sub-phase list:

```python
    expected = {
        "word_sampling", "peak_sampling", "peak_pdf", "peak_sample",
        "region_update", "loglikelihood", "total",
    }
    assert set(model.phase_times_) == expected
    assert set(rust_meta["phase_times"]) == expected

    sub_phases = [
        "word_sampling", "peak_sampling", "peak_pdf", "peak_sample",
        "region_update", "loglikelihood",
    ]
```

- [ ] **Step 2: Run to verify it fails**

Run: `micromamba run -n nimenv python -m pytest nimare/tests/test_gclda_rust.py -k "block_size or phase_keys" -v`
Expected: FAIL — `train_gclda_rust() got an unexpected keyword argument 'peak_block_size'`, and the phase-key test fails on the missing keys.

- [ ] **Step 3: Add the parameter and phase fields in Rust**

In `rust/gclda/src/model.rs`, add to `Params`:

```rust
    /// Number of peaks whose Gaussian densities are evaluated per parallel
    /// block in `update_peak_assignments`. Buffer cost is
    /// `peak_block_size * n_topics * n_regions * 8` bytes.
    pub peak_block_size: usize,
```

**This breaks every `Params { ... }` struct literal.** `Params` has no
`Default` impl, so all eight existing construction sites stop compiling until
`peak_block_size: 8192,` is added to each:

`rust/gclda/src/bin/gclda-train.rs`, `rust/gclda/src/model.rs`,
`rust/gclda/tests/init_golden.rs` (**two** sites),
`rust/gclda/tests/loglik.rs`, `rust/gclda/tests/outputs.rs`,
`rust/gclda/tests/pairwise_sum_wiring.rs`,
`rust/gclda/tests/sampler_peaks.rs`, `rust/gclda/tests/sampler_regions.rs`,
`rust/gclda/tests/sampler_words.rs`, plus the two test files created earlier in
this plan: `rust/gclda/tests/profile_pdf.rs` (Task 1) and
`rust/gclda/tests/peak_probs_block.rs` (Task 2).

Add those last two to this task's `git add` list as well.

Expect a wall of compile errors on first build; that is this change, not a
mistake. `cargo build` lists every site.

and to `PhaseTimes`:

```rust
    /// Time spent evaluating peak Gaussian densities. Subset of
    /// `peak_sampling`.
    pub peak_pdf: f64,
    /// Time spent in the sequential per-peak sampling body. Subset of
    /// `peak_sampling`.
    pub peak_sample: f64,
```

In `rust/gclda/src/output.rs`, add both keys to the `"phase_times"` object:

```rust
            "peak_pdf": model.phase_times.peak_pdf,
            "peak_sample": model.phase_times.peak_sample,
```

- [ ] **Step 4: Block the sampler loop**

In `rust/gclda/src/sampler/peaks.rs`, replace the single `for i_ptoken in 0..n_ptokens` loop in `update_peak_assignments` with a blocked pair of loops. Delete the old per-peak `self.peak_probs_for(i_ptoken, &mut peak_probs);` call and the `peak_probs` buffer it filled; everything else in the per-peak body is unchanged.

```rust
        let stride = n_topics * n_regions;
        let block_size = self.params.peak_block_size.max(1);
        let mut block_buf = vec![0.0f64; block_size.min(n_ptokens.max(1)) * stride];

        let mut block_start = 0usize;
        while block_start < n_ptokens {
            let len = block_size.min(n_ptokens - block_start);

            let t_pdf = std::time::Instant::now();
            self.peak_probs_block(block_start, len, &mut block_buf);
            self.phase_times.peak_pdf += t_pdf.elapsed().as_secs_f64();

            let t_sample = std::time::Instant::now();
            for i in 0..len {
                let i_ptoken = block_start + i;
                let peak_probs = &block_buf[i * stride..(i + 1) * stride];

                // The rest of this loop body is the CURRENT contents of the
                // existing `for i_ptoken in 0..n_ptokens` loop, moved here
                // unchanged: the three count decrements, the max_logp /
                // log1p / exp stabilization over topics, the `probs_pdf`
                // build, `rng.sample_from_unnormalized(&probs_pdf)?`, the
                // `region = idx / n_topics` / `topic = idx % n_topics`
                // decode, and the three re-increments plus the two
                // assignment writes. Do not retype it from memory -- move
                // the existing lines. The ONLY edits are:
                //   1. delete the `self.peak_probs_for(i_ptoken, &mut peak_probs);`
                //      call, since `peak_probs` is now the block slice above;
                //   2. delete the now-unused `let mut peak_probs = vec![...]`
                //      declaration further up the function.
                // `peak_probs` is indexed exactly as before
                // (`Model::at(i_topic, j_region, n_regions)`).
            }
            self.phase_times.peak_sample += t_sample.elapsed().as_secs_f64();

            block_start += len;
        }
```

Borrowing note: `block_buf` is a local `Vec`, so the `&self` borrow taken by `peak_probs_block` ends before the mutable per-peak loop begins. If the borrow checker objects, the cause is `block_buf` having been made a field of `Model` — keep it local.

- [ ] **Step 5: Add the CLI flag**

In `rust/gclda/src/bin/gclda-train.rs`, add to `struct Args`:

```rust
    /// Peaks per parallel PDF-evaluation block. Larger blocks expose more
    /// parallelism; buffer cost is
    /// `peak_block_size * n_topics * n_regions * 8` bytes.
    #[arg(long, default_value_t = 8192)]
    peak_block_size: usize,
```

and pass it through where `Params` is constructed.

- [ ] **Step 6: Add matching Python timing keys**

In `nimare/annotate/gclda.py`, add `"peak_pdf": 0.0,` and `"peak_sample": 0.0,` to the `self.phase_times_` dict literal.

In `_update_peak_assignments`, wrap the two existing sub-steps:

```python
        t_pdf = time.perf_counter()
        peak_probs = self._get_peak_probs(self)
        self.phase_times_["peak_pdf"] += time.perf_counter() - t_pdf
```

and

```python
        t_sample = time.perf_counter()
        _jit_update_peak_assignments(
            ...  # existing arguments, unchanged
        )
        self.phase_times_["peak_sample"] += time.perf_counter() - t_sample
```

- [ ] **Step 7: Add the Python parameter passthrough**

In `nimare/annotate/gclda_rs.py`, `train_gclda_rust` already forwards `**params` to CLI flags. Confirm `peak_block_size` maps to `--peak-block-size` via the existing name-to-flag conversion; if the mapping is an explicit allowlist, add `peak_block_size` to it.

- [ ] **Step 8: Run to verify it passes**

```bash
cd rust/gclda && cargo build --release && cargo test --release && cd ../..
micromamba run -n nimenv python -m pytest nimare/tests/test_gclda_rust.py nimare/tests/test_annotate_gclda.py -v
```

Expected: all PASS, including all four Level 2 configurations. **The Level 2 and Level 3 tests passing is the real acceptance criterion for this task** — they are what prove blocking did not change a single bit.

- [ ] **Step 9: Commit**

```bash
git add rust/gclda/src/sampler/peaks.rs rust/gclda/src/model.rs \
        rust/gclda/src/output.rs rust/gclda/src/bin/gclda-train.rs \
        rust/gclda/tests/profile_pdf.rs rust/gclda/tests/peak_probs_block.rs \
        rust/gclda/tests/loglik.rs rust/gclda/tests/outputs.rs \
        rust/gclda/tests/init_golden.rs rust/gclda/tests/pairwise_sum_wiring.rs \
        rust/gclda/tests/sampler_peaks.rs rust/gclda/tests/sampler_regions.rs \
        rust/gclda/tests/sampler_words.rs \
        nimare/annotate/gclda.py nimare/annotate/gclda_rs.py \
        nimare/tests/test_gclda_rust.py
git diff --cached --stat
git commit -m "[ENH] Evaluate peak PDFs in parallel blocks

Restores the parallelism the original fusion gave up. Gaussian evaluation
for a block of peaks runs across all cores; the block is then sampled
sequentially, so sampling order and RNG consumption are unchanged and
outputs stay bit-identical to Python.

Adds --peak-block-size and peak_pdf/peak_sample phase timing to both
implementations, so the split can be attributed rather than inferred.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Block-wise log-likelihood

**Files:**
- Modify: `rust/gclda/src/loglik.rs`
- Test: covered by existing Level 3 (`loglikelihood.tsv` bit-identity)

**Interfaces:**
- Consumes: `Model::peak_probs_block`, `Params.peak_block_size`
- Produces: no new public interface.

- [ ] **Step 1: Confirm the existing test covers this**

Run: `micromamba run -n nimenv python -m pytest nimare/tests/test_gclda_rust.py -k "probability_matrices or every_iteration" -v`
Expected: PASS. These compare log-likelihood output against Python; they must still pass after the change, which is what makes this task safe without a new test.

- [ ] **Step 2: Replace per-peak evaluation with block evaluation**

In `rust/gclda/src/loglik.rs`, find the loop that calls `peak_probs_for` once per peak. Restructure it exactly as Task 3 restructured the sampler: an outer `while block_start < n_ptokens` loop calling `peak_probs_block` into a local buffer, and an inner loop reading each peak's contiguous `stride`-length slice.

The per-peak accumulation body is unchanged. The accumulation into the running log-likelihood total stays **sequential and in peak order** — it is a floating-point reduction, and reassociating it would change the result.

- [ ] **Step 3: Verify bit-identity**

```bash
cd rust/gclda && cargo build --release && cargo test --release && cd ../..
micromamba run -n nimenv python -m pytest nimare/tests/test_gclda_rust.py -v
```

Expected: all PASS. Any difference in `loglikelihood.tsv` means the accumulation order changed.

- [ ] **Step 4: Commit**

```bash
git add rust/gclda/src/loglik.rs
git diff --cached --stat
git commit -m "[ENH] Evaluate log-likelihood peak PDFs in parallel blocks

Same treatment as the sampler: block PDF evaluation is parallel, the
accumulation into the running total stays sequential and in peak order,
since reassociating that reduction would change the result.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: `Float` trait and the `--compute-dtype f32` path

**Files:**
- Create: `rust/gclda/src/float.rs`
- Modify: `rust/gclda/src/gaussian.rs`, `rust/gclda/src/sampler/peaks.rs`, `rust/gclda/src/model.rs`, `rust/gclda/src/lib.rs`, `rust/gclda/src/bin/gclda-train.rs`
- Modify: `rust/gclda/src/rng.rs`
- Test: `nimare/tests/test_gclda_rust.py`

**Interfaces:**
- Produces: `crate::float::Float` trait; `gaussian::pdf_generic<F: Float>`; `Params.compute_dtype: ComputeDtype` (`enum ComputeDtype { F64, F32 }`); CLI `--compute-dtype {f64,f32}`; `Mt19937::sample_from_unnormalized_f32(&mut self, weights: &[f32]) -> Result<usize, GcldaError>`.

**This flag buys roughly 4% once Task 3 has landed.** It is a correctness-sensitive change for a small gain; if any step here threatens the bit-exactness of the `f64` path, stop and report rather than working around it.

- [ ] **Step 1: Write the failing test**

Append to `nimare/tests/test_gclda_rust.py`:

```python
@requires_rust
def test_f32_compute_path_is_close_to_f64_and_leaves_f64_exact(
    small_corpus, mni_mask, tmp_path
):
    """--compute-dtype f32 must run and land near f64 over a short run.

    f32 perturbs the sampling weights, so a single flipped categorical draw
    sends the two chains apart permanently. This therefore runs few enough
    iterations that divergence is unlikely, and asserts only closeness -- it
    deliberately makes no claim of scientific equivalence. The flag is
    documented as experimental. The load-bearing assertion is the last one:
    that the f64 default is untouched.
    """
    counts, coords = small_corpus
    mask_path = str(tmp_path / "mask.nii.gz")
    mni_mask.to_filename(mask_path)

    common = dict(
        mask=mask_path, binary=BINARY, n_topics=4, n_regions=2,
        symmetric=True, n_iters=2, loglikely_freq=2,
    )
    f64_dir = str(tmp_path / "f64")
    f32_dir = str(tmp_path / "f32")
    annotate.gclda_rs.train_gclda_rust(
        counts, coords, out_dir=f64_dir, compute_dtype="f64", **common
    )
    annotate.gclda_rs.train_gclda_rust(
        counts, coords, out_dir=f32_dir, compute_dtype="f32", **common
    )

    for name in ("p_topic_g_voxel", "p_word_g_topic"):
        a = np.load(os.path.join(f64_dir, f"{name}.npy"))
        b = np.load(os.path.join(f32_dir, f"{name}.npy"))
        assert a.shape == b.shape
        assert np.isfinite(b).all(), f"{name} contains non-finite values under f32"
        np.testing.assert_allclose(b, a, rtol=1e-3, atol=1e-6)

    # The f64 default must be bit-identical to Python, exactly as before.
    py_model = annotate.gclda.GCLDAModel(
        counts, coords, mask=mask_path, n_topics=4, n_regions=2, symmetric=True
    )
    py_model.fit(n_iters=2, loglikely_freq=2)
    np.testing.assert_array_equal(
        np.load(os.path.join(f64_dir, "p_topic_g_voxel.npy")),
        py_model.p_topic_g_voxel_,
    )
```

- [ ] **Step 2: Run to verify it fails**

Run: `micromamba run -n nimenv python -m pytest nimare/tests/test_gclda_rust.py -k f32_compute -v`
Expected: FAIL — unexpected keyword argument `compute_dtype`.

- [ ] **Step 3: Create the `Float` trait**

Create `rust/gclda/src/float.rs`:

```rust
//! Minimal float abstraction so the Gaussian PDF and the peak sampling weight
//! arithmetic can be written once and instantiated at `f64` (the bit-exact
//! default) and `f32` (the opt-in `--compute-dtype f32` path).
//!
//! This exists specifically to avoid a duplicated `f32` copy of the peak
//! sampler: that body is subtle, and two copies silently drifting apart would
//! be a worse defect than any performance this buys.

pub trait Float:
    Copy
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + PartialOrd
{
    const ZERO: Self;
    const NEG_INFINITY: Self;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn exp(self) -> Self;
    fn ln_1p(self) -> Self;
}

impl Float for f64 {
    const ZERO: Self = 0.0;
    const NEG_INFINITY: Self = f64::NEG_INFINITY;
    #[inline] fn from_f64(v: f64) -> Self { v }
    #[inline] fn to_f64(self) -> f64 { self }
    #[inline] fn exp(self) -> Self { f64::exp(self) }
    #[inline] fn ln_1p(self) -> Self { f64::ln_1p(self) }
}

impl Float for f32 {
    const ZERO: Self = 0.0;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;
    #[inline] fn from_f64(v: f64) -> Self { v as f32 }
    #[inline] fn to_f64(self) -> f64 { self as f64 }
    #[inline] fn exp(self) -> Self { f32::exp(self) }
    #[inline] fn ln_1p(self) -> Self { f32::ln_1p(self) }
}
```

Register it in `rust/gclda/src/lib.rs` with `pub mod float;`.

- [ ] **Step 4: Make the PDF generic**

In `rust/gclda/src/gaussian.rs`, add a generic evaluator beside the existing `pdf`. **Keep `pdf` as-is** so the `f64` call sites are byte-for-byte unchanged:

```rust
/// Generic form of [`pdf`], used by the `--compute-dtype f32` path. The `f64`
/// instantiation must produce bit-identical results to [`pdf`]; the
/// bit-exactness suite is what proves it.
#[inline]
pub fn pdf_generic<F: crate::float::Float>(
    point: &[F; 3],
    mean: &[F; 3],
    precision: &[[F; 3]; 3],
    log_norm: F,
) -> F {
    let mut quad = F::ZERO;
    for i in 0..3 {
        let centered_i = point[i] - mean[i];
        let mut inner = F::ZERO;
        for j in 0..3 {
            inner = inner + precision[i][j] * (point[j] - mean[j]);
        }
        quad = quad + centered_i * inner;
    }
    (log_norm - F::from_f64(0.5) * quad).exp()
}
```

- [ ] **Step 5: Add the f32 region-parameter mirror**

In `rust/gclda/src/model.rs`, add to `Model`:

```rust
    /// `f32` mirror of `regions_mu` / `regions_precision` / `regions_log_norm`,
    /// rebuilt once per iteration and used only under
    /// `--compute-dtype f32`. The `f64` originals remain authoritative:
    /// `regions_sigma` is computed as
    /// `cross_matrix - outer(sum, sum) / n_obs`, where at Neurosynth scale the
    /// cross terms reach ~1.2e8 while the centered result is ~2.5e3. `f32`'s
    /// ~7 significant digits cannot survive that subtraction, so the
    /// covariance path is never narrowed.
    pub regions_f32: Option<RegionsF32>,
```

with a `pub struct RegionsF32 { pub mu: Vec<[f32; 3]>, pub precision: Vec<[[f32; 3]; 3]>, pub log_norm: Vec<f32> }`, rebuilt at the end of `update_regions` when `params.compute_dtype == ComputeDtype::F32`.

Also add to `Params`:

```rust
    /// Precision used for Gaussian evaluation and sampling weights. Region
    /// parameters are always computed in `f64` regardless.
    pub compute_dtype: ComputeDtype,
```

and `#[derive(Clone, Copy, PartialEq, Eq)] pub enum ComputeDtype { F64, F32 }`.

As in Task 3, this breaks every `Params { ... }` literal. Add
`compute_dtype: ComputeDtype::F64,` to all nine sites listed there (the eight
originals plus `rust/gclda/tests/peak_probs_block.rs` added in Task 2).

- [ ] **Step 6: Make the sampler generic and add the f32 RNG draw**

In `rust/gclda/src/rng.rs`, add `sample_from_unnormalized_f32` mirroring the existing `f64` version exactly — same sequential accumulation, same comparison structure, same error on a non-positive total — operating on `&[f32]` and accumulating in `f32`.

In `rust/gclda/src/sampler/peaks.rs`:

1. **Make the block primitive generic.** `peak_probs_block` and
   `peak_probs_block_forced` currently take `out: &mut [f64]`. Change both to
   `out: &mut [F]` with `F: Float`, evaluating through `pdf_generic`. Without
   this the `f32` path has no way to produce `f32` densities and the flag
   buys nothing — the whole point is that the Gaussian evaluation itself runs
   narrow. Under `F = f32` the block reads the `regions_f32` mirror; under
   `F = f64` it reads the `f64` originals exactly as today.

2. **Make the per-peak body generic.** Extract it into a private function
   parameterized on `F: Float`, and have `update_peak_assignments` dispatch
   once on `params.compute_dtype` to the `f64` or `f32` instantiation.
   Dispatch **once per sweep**, not per peak.

The `f64` instantiation must remain bit-identical to the pre-Task-5 code.
`tests/peak_probs_block.rs` from Task 2 keeps proving block-vs-per-peak
bit-identity at `f64`, and the Level 2/3 suites prove end-to-end identity; if
either fails after this refactor, stop and report rather than adjusting them.

- [ ] **Step 7: Add the CLI flag**

In `rust/gclda/src/bin/gclda-train.rs`:

```rust
/// Precision for Gaussian evaluation and sampling weights. `f64` is the
/// default and is bit-exact against the Python implementation. `f32` is
/// EXPERIMENTAL: it perturbs sampling weights, so results diverge from `f64`
/// and are not validated for scientific use. Region parameters are computed
/// in `f64` regardless of this setting.
#[derive(Clone, Copy, clap::ValueEnum)]
enum ComputeDtypeArg { F64, F32 }
```

with `#[arg(long, value_enum, default_value = "f64")] compute_dtype: ComputeDtypeArg,` on `Args`, mapped into `Params`.

- [ ] **Step 8: Add the Python passthrough**

In `nimare/annotate/gclda_rs.py`, ensure `compute_dtype` forwards to `--compute-dtype` (add to the allowlist if the mapping is explicit).

- [ ] **Step 9: Run to verify it passes**

```bash
cd rust/gclda && cargo build --release && cargo test --release && cd ../..
micromamba run -n nimenv python -m pytest nimare/tests/test_gclda_rust.py nimare/tests/test_annotate_gclda.py -v
```

Expected: all PASS. **If any Level 2 or Level 3 test fails, the generic refactor perturbed `f64` codegen — stop and report; do not adjust tolerances.**

- [ ] **Step 10: Commit**

```bash
git add rust/gclda/src/float.rs rust/gclda/src/lib.rs rust/gclda/src/gaussian.rs \
        rust/gclda/src/sampler/peaks.rs rust/gclda/src/model.rs \
        rust/gclda/src/rng.rs rust/gclda/src/bin/gclda-train.rs \
        nimare/annotate/gclda_rs.py nimare/tests/test_gclda_rust.py
git diff --cached --stat
git commit -m "[ENH] Add experimental --compute-dtype f32 path

Gaussian evaluation and sampling weights can run in f32 via a small Float
trait, so the subtle sequential sampler body exists in one copy rather
than two. Region parameters stay f64 unconditionally: the covariance
subtraction cannot survive f32 at realistic scale.

f32 perturbs sampling weights and so forfeits bit-exactness against
Python. It is validated only by a short-run tolerance check and is
documented experimental. The f64 default remains bit-exact.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Expose the new flags in the benchmark driver

**Files:**
- Modify: `benchmarks/bench_gclda_rust.py`
- Test: `nimare/tests/test_gclda_rust.py`

**Interfaces:**
- Consumes: CLI `--peak-block-size`, `--compute-dtype`
- Produces: driver arguments `--peak-block-size`, `--compute-dtype`, both recorded in the report JSON's `params`.

- [ ] **Step 1: Write the failing test**

Append to `nimare/tests/test_gclda_rust.py`:

```python
def test_benchmark_driver_exposes_blockwise_flags():
    """The benchmark must be able to sweep the new knobs, and record them.

    A benchmark that cannot vary block size cannot produce the thread- and
    block-scaling evidence this optimization is justified by, and one that
    does not record the setting produces numbers nobody can reproduce.
    """
    import sys

    sys.path.insert(0, os.path.join(REPO_ROOT, "benchmarks"))
    import bench_gclda_rust

    argv = [
        "--scale", "tiny", "--peak-block-size", "512", "--compute-dtype", "f32",
    ]
    old = sys.argv
    try:
        sys.argv = ["bench_gclda_rust.py", *argv]
        args = bench_gclda_rust.parse_args()
    finally:
        sys.argv = old

    assert args.peak_block_size == 512
    assert args.compute_dtype == "f32"
```

- [ ] **Step 2: Run to verify it fails**

Run: `micromamba run -n nimenv python -m pytest nimare/tests/test_gclda_rust.py -k benchmark_driver_exposes -v`
Expected: FAIL — `unrecognized arguments: --peak-block-size`.

- [ ] **Step 3: Add the arguments**

In `benchmarks/bench_gclda_rust.py`, in `parse_args()`:

```python
    parser.add_argument(
        "--peak-block-size",
        type=int,
        default=8192,
        help="Peaks per parallel PDF-evaluation block in the Rust trainer.",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=["f64", "f32"],
        default="f64",
        help="Rust Gaussian/weight precision. f64 is bit-exact; f32 is experimental.",
    )
```

Thread them into the params dict the driver builds for `run_rust_once`, and into the recorded `params` in the report JSON. The Python child ignores both — Python has no equivalent knob — so when `--compute-dtype f32` is set the driver must **skip the equality check and label the run as not comparable**, rather than reporting a failure. Add a caveat string saying so.

- [ ] **Step 4: Run to verify it passes**

Run: `micromamba run -n nimenv python -m pytest nimare/tests/test_gclda_rust.py -k benchmark_driver_exposes -v`
Expected: PASS.

- [ ] **Step 5: Smoke-test**

Run:
```bash
micromamba run -n nimenv python benchmarks/bench_gclda_rust.py \
    --scale tiny --n-iters 5 --peak-block-size 512 \
    --out /tmp/gclda_bench_block.json
```
Expected: completes, equality check PASSES (f64 default), prints a table.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/bench_gclda_rust.py nimare/tests/test_gclda_rust.py
git diff --cached --stat
git commit -m "[ENH] Expose --peak-block-size and --compute-dtype in the benchmark driver

Records both in the report JSON so results are reproducible. Under
--compute-dtype f32 the equality check is skipped and the run labelled
not comparable, since Python has no equivalent knob.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Re-measure and update the results document

**Files:**
- Modify: `benchmarks/gclda_rust_results.md`

- [ ] **Step 1: Rebuild**

```bash
cd rust/gclda && cargo build --release && cd ../..
```

- [ ] **Step 2: Re-run the real-corpus benchmark**

```bash
micromamba run -n nimenv python benchmarks/bench_gclda_rust.py \
    --scale neurosynth --n-iters 20 --n-topics 100 --repeats 3 \
    --neurosynth-data-dir /mnt/c/Users/tsalo/.nimare \
    --out /tmp/gclda_bench_ns_after.json
```

The equality check must PASS. If it does not, blocking changed results and the work is not done.

- [ ] **Step 3: Re-run the n_topics sweep and thread scaling**

```bash
for t in 25 50 100; do
  micromamba run -n nimenv python benchmarks/bench_gclda_rust.py \
      --scale small --n-iters 50 --n-topics $t --out /tmp/gclda_after_t$t.json
done
for t in 1 2 4 8; do
  micromamba run -n nimenv python benchmarks/bench_gclda_rust.py \
      --scale small --n-iters 50 --n-topics 100 --threads $t \
      --out /tmp/gclda_after_threads_$t.json
done
```

Thread scaling matters more than before: it is the direct evidence that block-wise evaluation actually uses the cores.

- [ ] **Step 4: Sweep block size**

```bash
for b in 256 1024 8192 65536; do
  micromamba run -n nimenv python benchmarks/bench_gclda_rust.py \
      --scale neurosynth --n-iters 10 --n-topics 100 --repeats 1 \
      --peak-block-size $b --neurosynth-data-dir /mnt/c/Users/tsalo/.nimare \
      --out /tmp/gclda_after_block_$b.json
done
```

Confirms 8192 is a reasonable default and shows the memory/parallelism trade.

- [ ] **Step 5: Run the full test suite**

```bash
micromamba run -n nimenv python -m pytest nimare/tests/test_gclda_rust.py nimare/tests/test_annotate_gclda.py -v
cd rust/gclda && cargo test --release && cd ../..
```

Expected: all PASS. Paste the actual summary lines into the results document.

- [ ] **Step 6: Rewrite the results document**

Update `benchmarks/gclda_rust_results.md`:

- Add a **before/after** per-phase table for Neurosynth, using the `peak_pdf` / `peak_sample` split to show where the change landed.
- Update the headline. The prior headline reads "wall-clock wash (1.01x)"; replace it with the measured figure.
- Add the block-size sweep and the updated thread-scaling table.
- In "Where the wins came from", update the prediction-2 verdict: it was measured wrong, diagnosed, and fixed. State the new measured ratio. **Keep the record of the original wrong prediction and its cause** — that history is why the fix exists.
- Note the `--compute-dtype f32` flag, its measured effect, and its experimental status.
- **Report what was measured.** If the result is 1.2x rather than the projected 1.5x, write 1.2x and say the projection was optimistic.

- [ ] **Step 7: Commit**

```bash
git add benchmarks/gclda_rust_results.md
git diff --cached --stat
git commit -m "[DOC] Update GCLDA benchmark results after block-wise PDF evaluation

Measured before/after per-phase timings using the new peak_pdf and
peak_sample split, plus block-size and thread-scaling sweeps.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Completion Checklist

- [ ] `cargo test --release` passes in `rust/gclda`
- [ ] `micromamba run -n nimenv pytest nimare/tests/test_gclda_rust.py nimare/tests/test_annotate_gclda.py -v` passes
- [ ] Level 2 per-iteration equality passes for all four configurations, with blocking enabled
- [ ] Outputs are bit-identical across `--peak-block-size` of 1, 7, 8192, and a value exceeding the peak count
- [ ] The benchmark driver's equality gate passes at Neurosynth scale under the f64 default
- [ ] `benchmarks/gclda_rust_results.md` reports measured before/after figures, with any shortfall against the ~1.5x projection stated plainly
- [ ] `git status` shows no unintended files staged (the tree carries unrelated CRLF churn)
- [ ] **Nothing has been pushed**
