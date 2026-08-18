# GCLDA Block-wise Parallel PDF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the peak-sampling performance the GCLDA Rust port lost to serialization, without giving up bit-exact reproducibility against Python.

**Architecture:** Evaluate Gaussian PDFs one block of peaks at a time using rayon (the evaluations are order-independent), then sample each block sequentially (the sampling is not). Sampling order and RNG consumption are untouched, so f64 results stay bit-identical to Python.

**Tech Stack:** Rust (rayon, clap), Python 3.13 (numpy, numba, pandas), micromamba env `nimenv`.

**Spec:** `docs/superpowers/specs/2026-08-18-gclda-blockwise-pdf-design.md`

## Global Constraints

- **The `f64` path must remain bit-identical to Python.** The existing Level 2 (per-iteration state equality, all four region configurations) and Level 3 (end-to-end outputs) suites are the acceptance gate for every task. If they fail, the task is not done.
- **Never reassociate a floating-point reduction.** Parallelism is permitted only across disjoint peaks. `sample_from_unnormalized`'s accumulation and every per-peak arithmetic sequence stay exactly as they are.
- **All computation stays `f64`.** The float32 compute path was cut from this plan (see Task 5).
- Default `--peak-block-size` is `8192`.
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
| `rust/gclda/src/sampler/peaks.rs` | `peak_probs_for`, new `peak_probs_block`, blocked `update_peak_assignments` |
| `rust/gclda/src/loglik.rs` | Sparse log-likelihood; consumes the block primitive |
| `rust/gclda/src/model.rs` | `Model`, `Params`, `PhaseTimes` (gains `peak_pdf` / `peak_sample`) |
| `rust/gclda/src/output.rs` | `model.json` writer (gains the two new phase keys) |
| `rust/gclda/src/bin/gclda-train.rs` | CLI: `--peak-block-size`, `--profile-pdf` |
| `nimare/annotate/gclda.py` | Matching `peak_pdf` / `peak_sample` timing keys |
| `nimare/annotate/gclda_rs.py` | Pass new parameters through `train_gclda_rust` |
| `nimare/tests/test_gclda_rust.py` | Block-size invariance, phase-key parity |
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

### Task 5: REMOVED — float32 compute path cut from scope

**Do not implement this task.** It is retained as a numbered placeholder only so
that Task 6 and Task 7 keep their numbering.

Task 1 measured PDF evaluation at **35.5%** of the peak-sampling phase, not the
75-80% the design projected. The float32 path was justified at roughly a 4% total
gain on the old figure; on the measured one it is closer to 2%, in exchange for a
generic refactor threaded through the sampler and the loss of exact comparison
against Python. That is a bad trade, and the scope was cut on 2026-08-18.

If float32 is revisited later it needs its own spec: the measured sequential
fraction (77.2% of runtime) caps what any narrowing of the spatial path can buy.

---

### Task 6: Expose the new flags in the benchmark driver

**Files:**
- Modify: `benchmarks/bench_gclda_rust.py`
- Test: `nimare/tests/test_gclda_rust.py`

**Interfaces:**
- Consumes: CLI `--peak-block-size`
- Produces: driver argument `--peak-block-size`, recorded in the report JSON's `params`.

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

    argv = ["--scale", "tiny", "--peak-block-size", "512"]
    old = sys.argv
    try:
        sys.argv = ["bench_gclda_rust.py", *argv]
        args = bench_gclda_rust.parse_args()
    finally:
        sys.argv = old

    assert args.peak_block_size == 512
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
```

Thread it into the params dict the driver builds for `run_rust_once`, and into the recorded `params` in the report JSON. Block size does not affect results, so the equality check runs normally and must still pass.

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
git commit -m "[ENH] Expose --peak-block-size in the benchmark driver

Records it in the report JSON so block-size sweeps are reproducible.

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
- **Report what was measured.** The spec projected ~1.5x on the assumption that PDF
  evaluation was 75-80% of the peak-sampling phase. Task 1 measured 35.5%, which revised the
  projection to **~1.2x** before any code was written. Record both the original projection,
  the measurement that corrected it, and the final measured result. If the outcome is below
  ~1.2x, write what it is.
- Record the measured sequential fraction (Task 1 put it at 77.2% of runtime) as the ceiling
  on any future parallelization of this algorithm, and correct the GPU note in the "Out of
  scope" discussion: the Amdahl ceiling is 1.29x over current Rust, not the 1.79x estimated
  before measurement.

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
- [ ] The benchmark driver's equality gate passes at Neurosynth scale
- [ ] `benchmarks/gclda_rust_results.md` reports measured before/after figures against the **revised ~1.2x** projection (the spec's original ~1.5x rested on an assumption Task 1 measured and corrected), with any shortfall stated plainly
- [ ] `git status` shows no unintended files staged (the tree carries unrelated CRLF churn)
- [ ] **Nothing has been pushed**
