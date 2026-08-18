# GCLDA Rust Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port NiMARE's GCLDA topic model to a standalone Rust training binary that reproduces the Python implementation bit-for-bit while using far less memory.

**Architecture:** Two Python correctness/reproducibility fixes land first, establishing a stable reference. A Rust crate (library + thin CLI binary) then reimplements the sampler with identical arithmetic and identical RNG consumption, but fuses away the large per-iteration allocations. A thin Python loader exposes the Rust outputs to NiMARE's existing decoders unchanged.

**Tech Stack:** Rust 1.95 (rayon, clap, flate2), Python 3.10+ (NumPy, numba, nibabel, pytest), micromamba env `nimenv`.

**Spec:** `docs/superpowers/specs/2026-08-15-gclda-rust-port-design.md`

## Global Constraints

- **Environment:** every Python command runs as `micromamba run -n nimenv <command>`. Never use conda/venv/pip-venv. Never install into `base`.
- **Working directory:** `/mnt/c/Users/tsalo/Documents/neurostuff/NiMARE`, branch `rs-gclda`. Work on this branch directly.
- **Commits are allowed. Pushing is forbidden.** Never run `git push`.
- **The working tree has CRLF line-ending churn across 222 unrelated files.** NEVER run `git add -A`, `git add .`, or `git commit -a`. Stage only explicit paths, always. Verify with `git diff --cached --stat` before every commit.
- **Bit-exactness requires preserving arithmetic operation order.** Do not replace a division with a multiply-by-reciprocal. Do not reassociate sums. Do not enable fast-math. Do not use FMA where the Python does two separate operations. Rust does not contract to FMA by default — keep it that way.
- **All model floats are `f64`.** `f32` appears only in optional output serialization, never in training.
- **Coordinates are always 3-dimensional** (`x`, `y`, `z`). Matrix code may hard-code 3x3.
- **Parallelism (rayon) is permitted ONLY** in region-statistic accumulation, per-topic region parameter updates, and the final voxel probability sweep. The word sampler and peak sampler MUST remain sequential — this is required for correctness of collapsed Gibbs sampling, not merely for bit-exactness.
- **Rust crate location:** `rust/gclda/`. **Binary name:** `gclda-train`.
- **Reference constants:** MNI152 2mm brain mask is 91x109x91, 228483 nonzero voxels.

> **Line numbers in this plan are advisory — locate functions by NAME.** Tasks 1 and 2 add
> ~54 lines near the top of `nimare/annotate/gclda.py`, which shifts every reference below
> them. The citations here were remapped once after those tasks landed, but any further edit
> to that file re-staleness them. A stale citation is dangerous rather than merely unhelpful:
> `gclda.py:200-270` used to be the peak sampler and now lands inside the *word* sampler, so
> following it literally means porting the wrong function. Always confirm by function name.

**Convention for port tasks.** This is a port, so the specification for each sampler already
exists as working code. Where a task says "implement per the reference at `gclda.py:NNN-MMM`",
those lines ARE the specification — read them and translate them, preserving operation order.
The golden fixture in the same task is the acceptance criterion. Do not invent behavior that the
referenced Python does not have, and do not "improve" arithmetic along the way; the only
permitted deviations from the Python are the structural ones named explicitly in the task.

---

## Verified Facts (do not re-derive; these were measured before planning)

These were confirmed empirically. Trust them; the tasks below encode them.

1. **numba and NumPy share an identical MT19937 stream** for scalar seeds. `np.random.seed(s)` + `np.random.random()` gives identical f64 bits inside and outside `@njit`. One Rust RNG covers both.
2. **NumPy legacy `randint(bound, size=n)` uses 32-bit masked rejection sampling.** Confirmed against bounds 2, 3, 7, 64, 100, 1000. NOT 64-bit draws, NOT Lemire.
3. **`random()` is built from two 32-bit draws:** `a = u32() >> 5`, `b = u32() >> 6`, result `(a * 67108864.0 + b) / 9007199254740992.0`.
4. **MT19937 seeding is `init_genrand`** (Knuth multiplier 1812433253), not `init_by_array`, for scalar seeds.
5. **LAPACK's 3x3 inverse is not portably reproducible** — naive scalar LU matched `np.linalg.inv` in only 104/3000 trials. Hence Task 2.
6. **The bundled MNI mask** (`nimare/resources/templates/MNI152_2x2x2_brainmask.nii.gz`) has
   `sform_code=4`, `qform_code=4`, `datatype=2` (uint8), `scl_slope=nan`. nibabel's `img.affine`
   equals the sform. A `nan` scl_slope means "no scaling" — treat as slope 1.0, inter 0.0.

   **CORRECTED during Task 7 — the original entry here was wrong on two counts:**

   - **The file is BIG-ENDIAN.** `sizeof_hdr` reads 348 only when byte-swapped; interpreted
     little-endian it reads 1543569408.
   - **The real `vox_offset` is 448, not 0.** The file carries one AFNI extension (extension
     flag bytes at `[348:352]` are `01 00 00 00`), and `448 + 91*109*91 = 903077` is exactly the
     decompressed file size. The original "vox_offset=0" came from reading
     `img.header['vox_offset']` through nibabel, which **normalizes that field to 0** on load
     while tracking the true value in `img.dataobj.offset`. Read the raw header bytes, not
     nibabel's normalized view.

   `nib.Nifti1Image.to_filename()` **preserves** both properties, so every mask this project
   writes is big-endian with `vox_offset=448`. Consequently the byte-swap path and the
   nonzero-`vox_offset` path are exercised by every mask test, and the genuinely untested
   branches are **little-endian** files, the qform/pixdim affine fallback, and non-uint8
   datatypes. The `vox_offset == 0` → offset-352 fallback is covered by a synthetic-buffer
   unit test, since no real file in this project reaches it.

---

## File Structure

**Python — modified:**
- `nimare/annotate/gclda.py` — two reference fixes (Tasks 1, 2), phase timing (Task 19)

**Python — created:**
- `nimare/annotate/gclda_rs.py` — TSV export, output loading, `GCLDAResult` (Task 15)
- `nimare/tests/test_gclda_rust.py` — Levels 2-4 regression harness (Tasks 16-18)
- `nimare/tests/generate_gclda_fixtures.py` — golden fixture generator (Task 3, extended by 4, 6, 7)
- `benchmarks/gclda_synthetic.py` — synthetic dataset generator (Task 20)
- `benchmarks/bench_gclda_rust.py` — benchmark driver (Task 20)
- `benchmarks/gclda_rust_results.md` — measured results (Task 21)

**Rust — created, all under `rust/gclda/`:**

| File | Responsibility |
|---|---|
| `Cargo.toml` | crate manifest |
| `src/lib.rs` | public API re-exports |
| `src/rng.rs` | MT19937: `seed`, `random()`, `randint(bound)` |
| `src/gaussian.rs` | closed-form 3x3 inverse, logdet, PDF evaluation |
| `src/io/tsv.rs` | streaming count + coordinate TSV readers |
| `src/io/nifti.rs` | minimal NIfTI-1 reader -> bool volume + affine |
| `src/io/npy.rs` | `.npy` writer (f64/f32/i64) |
| `src/model.rs` | params, data, count matrices, assignments, init |
| `src/sampler/words.rs` | z-assignment sampler (sequential) |
| `src/sampler/peaks.rs` | y,r-assignment sampler, PDF fused (sequential) |
| `src/sampler/regions.rs` | spatial parameter update (rayon) |
| `src/loglik.rs` | sparse log-likelihood |
| `src/output.rs` | probability distributions + output directory writer |
| `src/bin/gclda-train.rs` | CLI |
| `tests/fixtures/` | golden fixtures generated from Python |

---

# Phase 0 — Python reference fixes

These must land before any Rust, so the port targets a stable, correct reference.

### Task 1: Fix the `compute_log_likelihood` off-by-one bug

`compute_log_likelihood` subtracts 1 from indices that are already 0-indexed, mapping document 0 to `-1` (which wraps to the last document). Every reported log-likelihood is computed against shifted indices. The fitted model is unaffected — log-likelihood never feeds back into sampling — but every logged value is wrong.

**Files:**
- Modify: `nimare/annotate/gclda.py` — the three offset lines in `compute_log_likelihood` (locate by content, not line number)
- Test: `nimare/tests/test_annotate_gclda.py`

**Interfaces:**
- Consumes: nothing
- Produces: corrected `GCLDAModel.compute_log_likelihood()` returning `(x_loglikely, w_loglikely, tot_loglikely)`; all Rust log-likelihood output is compared against this

- [ ] **Step 1: Write the failing test**

Add to `nimare/tests/test_annotate_gclda.py`:

```python
def test_gclda_loglikelihood_uses_zero_indexed_tokens(testdata_laird):
    """Log-likelihood must index documents/words directly, not offset by one.

    Regression test: the indices produced by docidx_mapper are already
    0-indexed, so subtracting 1 wrapped document 0 to the final document.
    """
    counts_df = annotate.text.generate_counts(
        testdata_laird.texts, text_column="abstract", tfidf=False, min_df=1, max_df=1.0
    )
    model = annotate.gclda.GCLDAModel(
        counts_df,
        testdata_laird.coordinates,
        mask=testdata_laird.masker.mask_img,
        n_topics=5,
        n_regions=2,
        symmetric=True,
    )
    model._update_regions()

    # Recompute the word log-likelihood independently, with correct indexing.
    alpha, beta, gamma = model.params["alpha"], model.params["beta"], model.params["gamma"]
    doccounts = model.topics["n_peak_tokens_doc_by_topic"] + gamma
    docprobs_z = doccounts / np.sum(doccounts, axis=1)[:, None]
    wordcounts = model.topics["n_word_tokens_word_by_topic"] + beta
    wordprobs = wordcounts / np.sum(wordcounts, axis=0)[None, :]
    p_w_g_d = np.dot(docprobs_z, wordprobs.T)

    expected_w = 0.0
    for i in range(len(model.data["wtoken_word_idx"])):
        w = model.data["wtoken_word_idx"][i]
        d = model.data["wtoken_doc_idx"][i]
        expected_w += np.log(p_w_g_d[d, w])

    _, w_loglikely, _ = model.compute_log_likelihood(update_vectors=False)
    assert np.isclose(w_loglikely, expected_w)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `micromamba run -n nimenv pytest nimare/tests/test_annotate_gclda.py::test_gclda_loglikelihood_uses_zero_indexed_tokens -v`

Expected: FAIL — the assertion compares a log-likelihood computed with shifted indices against one computed correctly.

- [ ] **Step 3: Apply the fix**

In `nimare/annotate/gclda.py`, change line 984 from:

```python
            doc = self.data["ptoken_doc_idx"][i_ptoken] - 1  # convert didx from 1-idx to 0-idx
```

to:

```python
            doc = self.data["ptoken_doc_idx"][i_ptoken]
```

Change lines 1019-1022 from:

```python
            # convert wtoken_word_idx from 1-idx to 0-idx
            word_token = self.data["wtoken_word_idx"][i_wtoken] - 1
            # convert wtoken_doc_idx from 1-idx to 0-idx
            doc = self.data["wtoken_doc_idx"][i_wtoken] - 1
```

to:

```python
            word_token = self.data["wtoken_word_idx"][i_wtoken]
            doc = self.data["wtoken_doc_idx"][i_wtoken]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `micromamba run -n nimenv pytest nimare/tests/test_annotate_gclda.py -v`

Expected: PASS, including the two pre-existing smoke tests.

- [ ] **Step 5: Commit**

```bash
git add nimare/annotate/gclda.py nimare/tests/test_annotate_gclda.py
git diff --cached --stat   # MUST show exactly 2 files
git commit -m "[FIX] Correct off-by-one indexing in GCLDA log-likelihood

compute_log_likelihood subtracted 1 from token document and word indices,
described as converting from 1-indexed to 0-indexed. Those indices come from
docidx_mapper, built with enumerate(), so they were already 0-indexed.
Document 0 became index -1 and wrapped to the last document.

The fitted model is unaffected, since log-likelihood never feeds back into
sampling, but every reported likelihood value was computed against shifted
indices.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Replace LAPACK 3x3 inverse with closed-form adjugate

`np.linalg.inv`/`slogdet` dispatch to OpenBLAS, whose results depend on build configuration, CPU features, and threading, and are therefore not reproducible from Rust. Since these precision matrices feed sampling probabilities directly, a 1-ulp difference can flip a sampled index and cascade. A closed-form 3x3 adjugate inverse with fixed operation order is reproducible, exactly symmetric, and 2.5x faster.

**Files:**
- Modify: `nimare/annotate/gclda.py` — `_cache_region_pdf_params`
- Test: `nimare/tests/test_annotate_gclda.py`

**Interfaces:**
- Consumes: nothing
- Produces: `_inv3_logdet(sigma) -> (inv, logdet)` module-level function in `nimare/annotate/gclda.py`; Rust's `gaussian::inv3_logdet` must reproduce it bit-for-bit

- [ ] **Step 1: Write the failing test**

```python
def test_gclda_inv3_logdet_matches_lapack_and_is_symmetric():
    """Closed-form 3x3 inverse must agree with LAPACK and be exactly symmetric."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        m = rng.normal(size=(3, 3)) * rng.uniform(1, 60)
        sigma = m @ m.T + 50.0 * np.eye(3) * rng.uniform(0.1, 3)

        inv, logdet = annotate.gclda._inv3_logdet(sigma)

        assert np.allclose(inv, np.linalg.inv(sigma), rtol=1e-10)
        _, ref_logdet = np.linalg.slogdet(sigma)
        assert np.isclose(logdet, ref_logdet, rtol=1e-12)
        # Inverse of a symmetric matrix must itself be exactly symmetric.
        assert np.array_equal(inv, inv.T)
        # Deterministic: identical inputs give identical bits.
        inv2, logdet2 = annotate.gclda._inv3_logdet(sigma.copy())
        assert np.array_equal(inv, inv2) and logdet == logdet2


def test_gclda_inv3_logdet_rejects_nonpositive_definite():
    """A non-positive-definite covariance must raise, as the LAPACK path did."""
    singular = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(np.linalg.LinAlgError):
        annotate.gclda._inv3_logdet(singular)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `micromamba run -n nimenv pytest nimare/tests/test_annotate_gclda.py -k inv3 -v`

Expected: FAIL with `AttributeError: module 'nimare.annotate.gclda' has no attribute '_inv3_logdet'`

- [ ] **Step 3: Implement**

Add to `nimare/annotate/gclda.py`, immediately after the `_sample_from_unnormalized` function:

```python
def _inv3_logdet(sigma):
    """Invert a 3x3 matrix in closed form and return its log-determinant.

    Uses the adjugate formula with a fixed operation order. Unlike LAPACK,
    this is bit-for-bit reproducible across platforms and BLAS builds, which
    allows the Rust implementation to match Python exactly. It is also
    exactly symmetric for symmetric input, and about 2.5x faster than the
    ``inv`` + ``slogdet`` pair it replaces.

    Parameters
    ----------
    sigma : (3, 3) :obj:`numpy.ndarray`
        A symmetric positive-definite matrix.

    Returns
    -------
    inv : (3, 3) :obj:`numpy.ndarray`
        The matrix inverse.
    logdet : :obj:`float`
        The natural log of the determinant.
    """
    a00, a01, a02 = sigma[0, 0], sigma[0, 1], sigma[0, 2]
    a10, a11, a12 = sigma[1, 0], sigma[1, 1], sigma[1, 2]
    a20, a21, a22 = sigma[2, 0], sigma[2, 1], sigma[2, 2]

    c00 = a11 * a22 - a12 * a21
    c01 = a02 * a21 - a01 * a22
    c02 = a01 * a12 - a02 * a11
    c10 = a12 * a20 - a10 * a22
    c11 = a00 * a22 - a02 * a20
    c12 = a02 * a10 - a00 * a12
    c20 = a10 * a21 - a11 * a20
    c21 = a01 * a20 - a00 * a21
    c22 = a00 * a11 - a01 * a10

    det = a00 * c00 + a01 * c10 + a02 * c20
    if not det > 0.0:
        raise np.linalg.LinAlgError("Region covariance must be positive definite.")

    inv = np.empty((3, 3), dtype=np.float64)
    inv[0, 0] = c00 / det
    inv[0, 1] = c01 / det
    inv[0, 2] = c02 / det
    inv[1, 0] = c10 / det
    inv[1, 1] = c11 / det
    inv[1, 2] = c12 / det
    inv[2, 0] = c20 / det
    inv[2, 1] = c21 / det
    inv[2, 2] = c22 / det

    return inv, np.log(det)
```

Then replace the body of `_cache_region_pdf_params` (currently at `gclda.py:746-753`) with:

```python
    def _cache_region_pdf_params(self, topic_idx, region_idx, sigma):
        """Cache Gaussian parameters used repeatedly during sampling and decoding."""
        inv, logdet = _inv3_logdet(sigma)
        self.topics["regions_precision"][topic_idx, region_idx, ...] = inv
        self.topics["regions_log_norm"][topic_idx, region_idx] = -0.5 * (
            sigma.shape[0] * np.log(2 * np.pi) + logdet
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `micromamba run -n nimenv pytest nimare/tests/test_annotate_gclda.py -v`

Expected: PASS. The two smoke tests still pass — this changes results only at the 1e-13 level.

- [ ] **Step 5: Commit**

```bash
git add nimare/annotate/gclda.py nimare/tests/test_annotate_gclda.py
git diff --cached --stat
git commit -m "[REF] Use closed-form 3x3 inverse for GCLDA region parameters

Region covariances are always 3x3, since peak coordinates are always
(x, y, z). Replacing the np.linalg.inv/slogdet pair with a closed-form
adjugate inverse is 2.5x faster (0.064s vs 0.158s over 40000 calls),
removing 200 LAPACK calls per sampling iteration, and yields an exactly
symmetric inverse where LAPACK's result was not.

It is also reproducible. LAPACK dispatches to OpenBLAS, whose result
depends on build configuration, CPU features, and threading; a naive
scalar LU reproduced np.linalg.inv bit-for-bit in only 104 of 3000
trials. Since these precision matrices feed sampling probabilities
directly, that non-determinism blocks an exactly-reproducing port.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

# Phase 1 — Rust foundations

### Task 3: Crate scaffold and MT19937 RNG

The single highest-risk component: every sampled index depends on reproducing NumPy's legacy RNG exactly.

**Files:**
- Create: `rust/gclda/Cargo.toml`, `rust/gclda/src/lib.rs`, `rust/gclda/src/rng.rs`
- Create: `nimare/tests/generate_gclda_fixtures.py`
- Create: `rust/gclda/tests/fixtures/rng_random.json`, `rust/gclda/tests/fixtures/rng_randint.json`
- Modify: `.gitignore` (add `rust/gclda/target/`)

**Interfaces:**
- Consumes: nothing
- Produces:
  - `gclda::rng::Mt19937` with `Mt19937::new(seed: u32) -> Self`, `fn random(&mut self) -> f64`, `fn randint(&mut self, bound: u64) -> u64`, `fn sample_from_unnormalized(&mut self, weights: &[f64]) -> Result<usize, GcldaError>`
  - `gclda::GcldaError` enum

- [ ] **Step 1: Write the fixture generator**

Create `nimare/tests/generate_gclda_fixtures.py`:

```python
"""Generate golden fixtures pinning Python behavior for the Rust GCLDA port.

Run with:
    micromamba run -n nimenv python nimare/tests/generate_gclda_fixtures.py

Writes JSON fixtures into rust/gclda/tests/fixtures/. These pin the exact
numeric behavior the Rust implementation must reproduce bit-for-bit. Floats
are serialized as hex bit patterns so JSON round-tripping cannot lose
precision.
"""

import json
import os
import struct

import numpy as np

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
FIXTURE_DIR = os.path.join(REPO_ROOT, "rust", "gclda", "tests", "fixtures")


def f64_bits(x):
    """Serialize a float64 as a hex bit pattern, losslessly."""
    return struct.pack("<d", float(x)).hex()


def write(name, obj):
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    path = os.path.join(FIXTURE_DIR, name)
    with open(path, "w") as fo:
        json.dump(obj, fo, indent=2)
    print(f"wrote {path}")


def gen_rng_random():
    """np.random.random() streams for several seeds."""
    cases = []
    for seed in (0, 1, 42, 12345, 2**31 - 1):
        np.random.seed(seed)
        draws = [f64_bits(np.random.random()) for _ in range(64)]
        cases.append({"seed": int(seed), "draws": draws})
    write("rng_random.json", cases)


def gen_rng_randint():
    """np.random.randint(bound, size=n) for bounds that do and do not straddle
    a power of two, exercising the masked-rejection path."""
    cases = []
    for seed in (1, 42):
        for bound in (2, 3, 7, 8, 64, 100, 1000, 1024, 65537):
            np.random.seed(seed)
            values = np.random.randint(bound, size=64).tolist()
            cases.append({"seed": int(seed), "bound": int(bound), "values": values})
    write("rng_randint.json", cases)


if __name__ == "__main__":
    gen_rng_random()
    gen_rng_randint()
```

- [ ] **Step 2: Generate the fixtures**

```bash
micromamba run -n nimenv python nimare/tests/generate_gclda_fixtures.py
```

Expected: writes `rng_random.json` and `rng_randint.json` under `rust/gclda/tests/fixtures/`.

- [ ] **Step 3: Create the crate manifest and lib root**

`rust/gclda/Cargo.toml`:

```toml
[package]
name = "gclda"
version = "0.1.0"
edition = "2021"
description = "Generalized Correspondence LDA trainer, bit-compatible with NiMARE's Python implementation"

[dependencies]
clap = { version = "4", features = ["derive"] }
rayon = "1"
flate2 = "1"
serde_json = "1"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
debug = true          # keep symbols for perf profiling
```

> **Two manifest details, both verified.** There is deliberately **no `[[bin]]` section**:
> cargo auto-discovers `src/bin/*.rs` when Task 14 creates it. Declaring the bin target here
> makes `cargo test` refuse to compile for Tasks 3-13 with "can't find bin `gclda-train`".
> And `serde_json` is a **regular dependency**, not a dev-dependency, because Task 13 writes
> `model.json` from `src/output.rs`.

`rust/gclda/src/lib.rs`:

```rust
//! Generalized Correspondence LDA, bit-compatible with NiMARE's Python implementation.
//!
//! Bit-exactness is a hard requirement, not a nicety: it is the correctness
//! oracle for this port. Do not reorder floating-point operations, do not
//! replace division with reciprocal multiplication, and do not enable
//! fast-math anywhere in this crate.

pub mod rng;

#[derive(Debug)]
pub enum GcldaError {
    NonPositiveWeights,
    NotPositiveDefinite,
    Parse(String),
    Io(std::io::Error),
}

impl std::fmt::Display for GcldaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GcldaError::NonPositiveWeights => {
                write!(f, "Sampling weights must sum to a positive value.")
            }
            GcldaError::NotPositiveDefinite => {
                write!(f, "Region covariance must be positive definite.")
            }
            GcldaError::Parse(m) => write!(f, "parse error: {m}"),
            GcldaError::Io(e) => write!(f, "io error: {e}"),
        }
    }
}

impl std::error::Error for GcldaError {}

impl From<std::io::Error> for GcldaError {
    fn from(e: std::io::Error) -> Self {
        GcldaError::Io(e)
    }
}
```

Append to `.gitignore`:

```
rust/gclda/target/
```

- [ ] **Step 4: Write the failing RNG tests**

`rust/gclda/tests/rng_golden.rs`:

```rust
use gclda::rng::Mt19937;

fn load(name: &str) -> serde_json::Value {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing fixture {path}: {e}. Run generate_gclda_fixtures.py"));
    serde_json::from_str(&text).unwrap()
}

fn bits_to_f64(hex: &str) -> f64 {
    let raw = (0..8)
        .map(|i| u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap())
        .collect::<Vec<u8>>();
    f64::from_le_bytes(raw.try_into().unwrap())
}

#[test]
fn random_stream_matches_numpy_bit_for_bit() {
    for case in load("rng_random.json").as_array().unwrap() {
        let seed = case["seed"].as_u64().unwrap() as u32;
        let mut rng = Mt19937::new(seed);
        for (i, expected) in case["draws"].as_array().unwrap().iter().enumerate() {
            let want = bits_to_f64(expected.as_str().unwrap());
            let got = rng.random();
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "seed {seed} draw {i}: got {got:?} want {want:?}"
            );
        }
    }
}

#[test]
fn randint_matches_numpy_legacy_masked_rejection() {
    for case in load("rng_randint.json").as_array().unwrap() {
        let seed = case["seed"].as_u64().unwrap() as u32;
        let bound = case["bound"].as_u64().unwrap();
        let mut rng = Mt19937::new(seed);
        for (i, expected) in case["values"].as_array().unwrap().iter().enumerate() {
            let want = expected.as_u64().unwrap();
            let got = rng.randint(bound);
            assert_eq!(got, want, "seed {seed} bound {bound} draw {i}");
        }
    }
}
```

- [ ] **Step 5: Run to verify it fails**

Run: `cd rust/gclda && cargo test --test rng_golden`

Expected: FAIL to compile — `rng::Mt19937` does not exist yet.

- [ ] **Step 6: Implement the RNG**

`rust/gclda/src/rng.rs`:

```rust
//! MT19937, reproducing NumPy's legacy `RandomState` bit-for-bit.
//!
//! Verified facts this encodes:
//!   * scalar seeding uses `init_genrand` (Knuth multiplier 1812433253)
//!   * `random()` consumes two u32 draws: (a >> 5) * 2^26 + (b >> 6), over 2^53
//!   * `randint(bound)` uses 32-bit masked rejection sampling
//!   * numba's in-njit RNG produces the identical stream, so this single
//!     implementation covers both model initialization and all sampling

use crate::GcldaError;

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

pub struct Mt19937 {
    state: [u32; N],
    index: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; N];
        state[0] = seed;
        for i in 1..N {
            let prev = state[i - 1];
            state[i] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { state, index: N }
    }

    /// Re-seed in place, matching `np.random.seed(seed)`.
    pub fn reseed(&mut self, seed: u32) {
        *self = Mt19937::new(seed);
    }

    fn generate(&mut self) {
        for i in 0..N {
            let y = (self.state[i] & UPPER_MASK) | (self.state[(i + 1) % N] & LOWER_MASK);
            let mut next = self.state[(i + M) % N] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= MATRIX_A;
            }
            self.state[i] = next;
        }
        self.index = 0;
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.generate();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// Equivalent to `np.random.random()`.
    #[inline]
    pub fn random(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67108864.0 + b) / 9007199254740992.0
    }

    /// Equivalent to `np.random.randint(bound)`, i.e. uniform on [0, bound).
    ///
    /// NumPy's legacy path is 32-bit masked rejection sampling. This was
    /// verified against bounds 2, 3, 7, 64, 100, and 1000.
    pub fn randint(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0);
        let rng_range = bound - 1;
        if rng_range == 0 {
            return 0;
        }
        let mut mask = rng_range;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        loop {
            let value = (self.next_u32() as u64) & mask;
            if value <= rng_range {
                return value;
            }
        }
    }

    /// Port of `_sample_from_unnormalized` in nimare/annotate/gclda.py.
    ///
    /// The accumulation order here is load-bearing: it must match the Python
    /// loop exactly, or sampled indices can differ.
    pub fn sample_from_unnormalized(&mut self, weights: &[f64]) -> Result<usize, GcldaError> {
        let mut total = 0.0f64;
        for &w in weights {
            total += w;
        }
        if total <= 0.0 {
            return Err(GcldaError::NonPositiveWeights);
        }
        let threshold = self.random() * total;
        let mut cumulative = 0.0f64;
        for (i, &w) in weights.iter().enumerate() {
            cumulative += w;
            if threshold < cumulative {
                return Ok(i);
            }
        }
        Ok(weights.len() - 1)
    }
}
```

Add `pub mod rng;` is already in `lib.rs` from Step 3.

- [ ] **Step 7: Run to verify it passes**

Run: `cd rust/gclda && cargo test --test rng_golden`

Expected: PASS — both tests, covering 5 seeds x 64 draws and 18 (seed, bound) combinations.

- [ ] **Step 8: Commit**

```bash
git add rust/gclda/Cargo.toml rust/gclda/Cargo.lock rust/gclda/src/lib.rs \
        rust/gclda/src/rng.rs rust/gclda/tests/rng_golden.rs \
        rust/gclda/tests/fixtures/rng_random.json \
        rust/gclda/tests/fixtures/rng_randint.json \
        nimare/tests/generate_gclda_fixtures.py .gitignore
git diff --cached --stat
git commit -m "[ENH] Add Rust GCLDA crate scaffold with NumPy-compatible MT19937

Reproduces NumPy's legacy RandomState bit-for-bit: init_genrand scalar
seeding, the two-draw float64 construction, and 32-bit masked rejection
sampling for bounded integers. Verified against golden fixtures covering
5 seeds and 9 bounds, including non-powers-of-two that exercise the
rejection branch.

numba's in-njit RNG produces an identical stream, so this single
implementation covers both model initialization and all sampling.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Gaussian module

**Files:**
- Create: `rust/gclda/src/gaussian.rs`
- Modify: `rust/gclda/src/lib.rs` (add `pub mod gaussian;`)
- Modify: `nimare/tests/generate_gclda_fixtures.py` (add `gen_gaussian`)
- Create: `rust/gclda/tests/fixtures/gaussian.json`

**Interfaces:**
- Consumes: `GcldaError`
- Produces:
  - `gclda::gaussian::inv3_logdet(sigma: &[[f64; 3]; 3]) -> Result<([[f64; 3]; 3], f64), GcldaError>`
  - `gclda::gaussian::log_norm(logdet: f64) -> f64`
  - `gclda::gaussian::pdf(point: &[f64; 3], mean: &[f64; 3], precision: &[[f64; 3]; 3], log_norm: f64) -> f64`

- [ ] **Step 1: Add the fixture generator**

Append to `nimare/tests/generate_gclda_fixtures.py`, and add `gen_gaussian()` to `__main__`:

```python
def gen_gaussian():
    """Closed-form 3x3 inverse/logdet and the Gaussian PDF, on fixed matrices."""
    from nimare.annotate.gclda import _inv3_logdet

    rng = np.random.default_rng(0)
    cases = []
    for _ in range(50):
        m = rng.normal(size=(3, 3)) * rng.uniform(1, 60)
        sigma = m @ m.T + 50.0 * np.eye(3) * rng.uniform(0.1, 3)
        inv, logdet = _inv3_logdet(sigma)
        log_norm = -0.5 * (3 * np.log(2 * np.pi) + logdet)
        mean = rng.normal(size=3) * 30.0
        points = rng.normal(size=(4, 3)) * 40.0
        pdfs = []
        for p in points:
            centered = p - mean
            quad = 0.0
            for i in range(3):
                inner = 0.0
                for j in range(3):
                    inner += inv[i, j] * (p[j] - mean[j])
                quad += centered[i] * inner
            pdfs.append(f64_bits(np.exp(log_norm - 0.5 * quad)))
        cases.append(
            {
                "sigma": [[f64_bits(v) for v in row] for row in sigma],
                "inv": [[f64_bits(v) for v in row] for row in inv],
                "logdet": f64_bits(logdet),
                "log_norm": f64_bits(log_norm),
                "mean": [f64_bits(v) for v in mean],
                "points": [[f64_bits(v) for v in p] for p in points],
                "pdfs": pdfs,
            }
        )
    write("gaussian.json", cases)
```

Regenerate: `micromamba run -n nimenv python nimare/tests/generate_gclda_fixtures.py`

- [ ] **Step 2: Write the failing test**

`rust/gclda/tests/gaussian_golden.rs`:

```rust
use gclda::gaussian::{inv3_logdet, log_norm, pdf};

mod common;
use common::{bits_to_f64, load};

#[test]
fn inverse_logdet_and_pdf_match_python_bit_for_bit() {
    for (c, case) in load("gaussian.json").as_array().unwrap().iter().enumerate() {
        let sigma = mat3(&case["sigma"]);
        let (inv, logdet) = inv3_logdet(&sigma).expect("positive definite");

        let want_inv = mat3(&case["inv"]);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(inv[i][j].to_bits(), want_inv[i][j].to_bits(), "case {c} inv[{i}][{j}]");
            }
        }
        let want_logdet = bits_to_f64(case["logdet"].as_str().unwrap());
        assert_eq!(logdet.to_bits(), want_logdet.to_bits(), "case {c} logdet");

        let ln = log_norm(logdet);
        assert_eq!(
            ln.to_bits(),
            bits_to_f64(case["log_norm"].as_str().unwrap()).to_bits(),
            "case {c} log_norm"
        );

        let mean = vec3(&case["mean"]);
        for (p, point_json) in case["points"].as_array().unwrap().iter().enumerate() {
            let point = vec3(point_json);
            let got = pdf(&point, &mean, &inv, ln);
            let want = bits_to_f64(case["pdfs"].as_array().unwrap()[p].as_str().unwrap());
            assert_eq!(got.to_bits(), want.to_bits(), "case {c} pdf {p}");
        }
    }
}

#[test]
fn singular_matrix_is_rejected() {
    let singular = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
    assert!(inv3_logdet(&singular).is_err());
}

fn vec3(v: &serde_json::Value) -> [f64; 3] {
    let a = v.as_array().unwrap();
    [
        bits_to_f64(a[0].as_str().unwrap()),
        bits_to_f64(a[1].as_str().unwrap()),
        bits_to_f64(a[2].as_str().unwrap()),
    ]
}

fn mat3(v: &serde_json::Value) -> [[f64; 3]; 3] {
    let a = v.as_array().unwrap();
    [vec3(&a[0]), vec3(&a[1]), vec3(&a[2])]
}
```

Create the shared helper `rust/gclda/tests/common/mod.rs`:

```rust
use std::path::PathBuf;

/// Resolve a repo-relative fixture path (as stored in the JSON fixtures) against
/// the repository root. Fixtures MUST NOT store absolute paths — a test that
/// hard-codes the generating machine's paths passes only on that machine.
pub fn repo_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

pub fn load(name: &str) -> serde_json::Value {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing fixture {path}: {e}. Run generate_gclda_fixtures.py"));
    serde_json::from_str(&text).unwrap()
}

pub fn bits_to_f64(hex: &str) -> f64 {
    let raw: Vec<u8> = (0..8)
        .map(|i| u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap())
        .collect();
    f64::from_le_bytes(raw.try_into().unwrap())
}
```

Refactor `rust/gclda/tests/rng_golden.rs` to `mod common; use common::{bits_to_f64, load};` and delete its local copies of those two functions.

- [ ] **Step 3: Run to verify it fails**

Run: `cd rust/gclda && cargo test --test gaussian_golden`

Expected: FAIL to compile — `gaussian` module does not exist.

- [ ] **Step 4: Implement**

`rust/gclda/src/gaussian.rs`:

```rust
//! 3x3 Gaussian helpers.
//!
//! Peak coordinates are always (x, y, z), so every covariance is 3x3 and the
//! closed-form adjugate inverse applies. Operation order here MUST match
//! `_inv3_logdet` in nimare/annotate/gclda.py exactly.

use crate::GcldaError;

const LOG_2PI: f64 = 1.837_877_066_409_345_6; // ln(2*pi), matches np.log(2 * np.pi)

pub fn inv3_logdet(sigma: &[[f64; 3]; 3]) -> Result<([[f64; 3]; 3], f64), GcldaError> {
    let (a00, a01, a02) = (sigma[0][0], sigma[0][1], sigma[0][2]);
    let (a10, a11, a12) = (sigma[1][0], sigma[1][1], sigma[1][2]);
    let (a20, a21, a22) = (sigma[2][0], sigma[2][1], sigma[2][2]);

    let c00 = a11 * a22 - a12 * a21;
    let c01 = a02 * a21 - a01 * a22;
    let c02 = a01 * a12 - a02 * a11;
    let c10 = a12 * a20 - a10 * a22;
    let c11 = a00 * a22 - a02 * a20;
    let c12 = a02 * a10 - a00 * a12;
    let c20 = a10 * a21 - a11 * a20;
    let c21 = a01 * a20 - a00 * a21;
    let c22 = a00 * a11 - a01 * a10;

    let det = a00 * c00 + a01 * c10 + a02 * c20;
    if !(det > 0.0) {
        return Err(GcldaError::NotPositiveDefinite);
    }

    let inv = [
        [c00 / det, c01 / det, c02 / det],
        [c10 / det, c11 / det, c12 / det],
        [c20 / det, c21 / det, c22 / det],
    ];
    Ok((inv, det.ln()))
}

#[inline]
pub fn log_norm(logdet: f64) -> f64 {
    -0.5 * (3.0 * LOG_2PI + logdet)
}

/// Evaluate a Gaussian PDF. Mirrors the loop structure of `_jit_spatial_pdf`;
/// the nested accumulation order is load-bearing.
#[inline]
pub fn pdf(
    point: &[f64; 3],
    mean: &[f64; 3],
    precision: &[[f64; 3]; 3],
    log_norm: f64,
) -> f64 {
    let mut quad = 0.0f64;
    for i in 0..3 {
        let centered_i = point[i] - mean[i];
        let mut inner = 0.0f64;
        for j in 0..3 {
            inner += precision[i][j] * (point[j] - mean[j]);
        }
        quad += centered_i * inner;
    }
    (log_norm - 0.5 * quad).exp()
}
```

Add `pub mod gaussian;` to `lib.rs`.

> **Note on `LOG_2PI`:** if the golden test fails only on `log_norm`, the constant does not match `np.log(2 * np.pi)`. Replace it with `(2.0 * std::f64::consts::PI).ln()` and re-run. Do not "fix" this by loosening the assertion.

- [ ] **Step 5: Run to verify it passes**

Run: `cd rust/gclda && cargo test`

Expected: PASS — all RNG and Gaussian tests.

- [ ] **Step 6: Commit**

```bash
git add rust/gclda/src/gaussian.rs rust/gclda/src/lib.rs \
        rust/gclda/tests/gaussian_golden.rs rust/gclda/tests/common/mod.rs \
        rust/gclda/tests/rng_golden.rs \
        rust/gclda/tests/fixtures/gaussian.json \
        nimare/tests/generate_gclda_fixtures.py
git diff --cached --stat
git commit -m "[ENH] Add bit-exact 3x3 Gaussian helpers to Rust GCLDA crate

Closed-form adjugate inverse, log-determinant, and PDF evaluation,
matching nimare.annotate.gclda._inv3_logdet and _jit_spatial_pdf
bit-for-bit across 50 golden matrices and 200 PDF evaluations.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: `.npy` writer

**Files:**
- Create: `rust/gclda/src/io/mod.rs`, `rust/gclda/src/io/npy.rs`
- Modify: `rust/gclda/src/lib.rs` (add `pub mod io;`)

**Interfaces:**
- Consumes: `GcldaError`
- Produces:
  - `gclda::io::npy::write_f64(path: &Path, shape: &[usize], data: &[f64]) -> Result<(), GcldaError>`
  - `gclda::io::npy::write_f32_from_f64(path: &Path, shape: &[usize], data: &[f64]) -> Result<(), GcldaError>`
  - `gclda::io::npy::write_i64(path: &Path, shape: &[usize], data: &[i64]) -> Result<(), GcldaError>`
  - `gclda::io::npy::NpyWriter::create(path, shape, dtype)` / `.write_row(&[f64])` / `.finish()` for streamed writes

- [ ] **Step 1: Write the failing test**

`rust/gclda/tests/npy_roundtrip.rs`:

```rust
use gclda::io::npy;
use std::process::Command;

/// Write a .npy from Rust, then have NumPy read it back and report what it saw.
fn numpy_describe(path: &str) -> String {
    let script = format!(
        "import numpy as np; a = np.load(r'{path}'); \
         print(a.dtype, a.shape, float(a.sum()), float(a.ravel()[0]), float(a.ravel()[-1]))"
    );
    let out = Command::new("micromamba")
        .args(["run", "-n", "nimenv", "python", "-c", &script])
        .output()
        .expect("failed to run micromamba");
    assert!(
        out.status.success(),
        "numpy failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

#[test]
fn f64_matrix_roundtrips_through_numpy() {
    let dir = std::env::temp_dir().join("gclda_npy_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.npy");

    let data: Vec<f64> = (0..12).map(|i| i as f64 * 0.5).collect();
    npy::write_f64(&path, &[3, 4], &data).unwrap();

    let desc = numpy_describe(path.to_str().unwrap());
    assert_eq!(desc, "float64 (3, 4) 33.0 0.0 5.5");
}

#[test]
fn i64_matrix_roundtrips_through_numpy() {
    let dir = std::env::temp_dir().join("gclda_npy_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("i.npy");

    let data: Vec<i64> = (0..6).collect();
    npy::write_i64(&path, &[2, 3], &data).unwrap();

    let desc = numpy_describe(path.to_str().unwrap());
    assert_eq!(desc, "int64 (2, 3) 15.0 0.0 5.0");
}

#[test]
fn streamed_rows_match_a_single_shot_write() {
    let dir = std::env::temp_dir().join("gclda_npy_test");
    std::fs::create_dir_all(&dir).unwrap();

    let data: Vec<f64> = (0..20).map(|i| (i as f64).sin()).collect();
    let one_shot = dir.join("one.npy");
    npy::write_f64(&one_shot, &[4, 5], &data).unwrap();

    let streamed = dir.join("stream.npy");
    let mut w = npy::NpyWriter::create(&streamed, &[4, 5], npy::Dtype::F64).unwrap();
    for row in data.chunks(5) {
        w.write_row(row).unwrap();
    }
    w.finish().unwrap();

    assert_eq!(
        std::fs::read(&one_shot).unwrap(),
        std::fs::read(&streamed).unwrap()
    );
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd rust/gclda && cargo test --test npy_roundtrip`

Expected: FAIL to compile — `io::npy` does not exist.

- [ ] **Step 3: Implement**

`rust/gclda/src/io/mod.rs`:

```rust
pub mod npy;
```

`rust/gclda/src/io/npy.rs`:

```rust
//! Minimal NPY v1.0 writer.
//!
//! The format is a short ASCII header followed by raw little-endian data,
//! which lets Python open large outputs with np.load(..., mmap_mode="r")
//! without ever making them resident.

use crate::GcldaError;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F64,
    F32,
    I64,
}

impl Dtype {
    fn descr(self) -> &'static str {
        match self {
            Dtype::F64 => "<f8",
            Dtype::F32 => "<f4",
            Dtype::I64 => "<i8",
        }
    }
}

fn header_bytes(shape: &[usize], dtype: Dtype) -> Vec<u8> {
    // NumPy writes a trailing comma for 1-D shapes: (5,) not (5)
    let shape_repr = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        format!("({})", parts.join(", "))
    };
    let dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        dtype.descr(),
        shape_repr
    );

    // Magic (6) + version (2) + header length (2) + dict must be a multiple of 64.
    let mut padded = dict.into_bytes();
    let prefix = 10;
    let unpadded = prefix + padded.len() + 1; // +1 for the trailing newline
    let pad = (64 - (unpadded % 64)) % 64;
    padded.extend(std::iter::repeat(b' ').take(pad));
    padded.push(b'\n');

    let mut out = Vec::with_capacity(prefix + padded.len());
    out.extend_from_slice(b"\x93NUMPY");
    out.push(1); // major
    out.push(0); // minor
    out.extend_from_slice(&(padded.len() as u16).to_le_bytes());
    out.extend_from_slice(&padded);
    out
}

pub struct NpyWriter {
    inner: BufWriter<File>,
    dtype: Dtype,
    expected: usize,
    written: usize,
}

impl NpyWriter {
    pub fn create(path: &Path, shape: &[usize], dtype: Dtype) -> Result<Self, GcldaError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut inner = BufWriter::new(File::create(path)?);
        inner.write_all(&header_bytes(shape, dtype))?;
        Ok(NpyWriter {
            inner,
            dtype,
            expected: shape.iter().product(),
            written: 0,
        })
    }

    pub fn write_row(&mut self, row: &[f64]) -> Result<(), GcldaError> {
        for &v in row {
            match self.dtype {
                Dtype::F64 => self.inner.write_all(&v.to_le_bytes())?,
                Dtype::F32 => self.inner.write_all(&(v as f32).to_le_bytes())?,
                Dtype::I64 => self.inner.write_all(&(v as i64).to_le_bytes())?,
            }
        }
        self.written += row.len();
        Ok(())
    }

    pub fn write_row_i64(&mut self, row: &[i64]) -> Result<(), GcldaError> {
        for &v in row {
            self.inner.write_all(&v.to_le_bytes())?;
        }
        self.written += row.len();
        Ok(())
    }

    pub fn finish(mut self) -> Result<(), GcldaError> {
        assert_eq!(
            self.written, self.expected,
            "npy writer got {} values, header declared {}",
            self.written, self.expected
        );
        self.inner.flush()?;
        Ok(())
    }
}

pub fn write_f64(path: &Path, shape: &[usize], data: &[f64]) -> Result<(), GcldaError> {
    let mut w = NpyWriter::create(path, shape, Dtype::F64)?;
    w.write_row(data)?;
    w.finish()
}

pub fn write_f32_from_f64(path: &Path, shape: &[usize], data: &[f64]) -> Result<(), GcldaError> {
    let mut w = NpyWriter::create(path, shape, Dtype::F32)?;
    w.write_row(data)?;
    w.finish()
}

pub fn write_i64(path: &Path, shape: &[usize], data: &[i64]) -> Result<(), GcldaError> {
    let mut w = NpyWriter::create(path, shape, Dtype::I64)?;
    w.write_row_i64(data)?;
    w.finish()
}
```

Add `pub mod io;` to `lib.rs`.

- [ ] **Step 4: Run to verify it passes**

Run: `cd rust/gclda && cargo test --test npy_roundtrip`

Expected: PASS — all three tests, with NumPy confirming dtype, shape, and values.

- [ ] **Step 5: Commit**

```bash
git add rust/gclda/src/io/mod.rs rust/gclda/src/io/npy.rs rust/gclda/src/lib.rs \
        rust/gclda/tests/npy_roundtrip.rs
git diff --cached --stat
git commit -m "[ENH] Add NPY writer to Rust GCLDA crate

Minimal NPY v1.0 writer supporting float64, float32, and int64, with a
streaming row-wise mode so large output matrices are never fully
resident. Verified by round-tripping through NumPy.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: TSV ingest

The index-determining logic. A mistake here produces correctly-shaped but silently misaligned output.

**Files:**
- Create: `rust/gclda/src/io/tsv.rs`
- Modify: `rust/gclda/src/io/mod.rs`
- Modify: `nimare/tests/generate_gclda_fixtures.py` (add `gen_ingest`)
- Create: `rust/gclda/tests/fixtures/ingest.json`, `rust/gclda/tests/fixtures/counts.tsv`, `rust/gclda/tests/fixtures/coordinates.tsv`

**Interfaces:**
- Consumes: `GcldaError`
- Produces:
  ```rust
  pub struct Corpus {
      pub ids: Vec<String>,            // sorted intersection, defines docidx
      pub vocabulary: Vec<String>,     // defines the W axis
      pub wtoken_doc_idx: Vec<u32>,
      pub wtoken_word_idx: Vec<u32>,
      pub ptoken_doc_idx: Vec<u32>,
      pub ptoken_coords: Vec<[f64; 3]>,
  }
  pub fn load_corpus(counts: &Path, coords: &Path) -> Result<Corpus, GcldaError>
  ```

**Critical semantics — replicate exactly (from `gclda.py:418-523`):**

1. `ids` = **lexicographically sorted** intersection of count IDs and coordinate IDs, compared as strings. Rust's byte-order `String` sort agrees with Python's code-point sort here.
2. `docidx` is the position within that sorted `ids` list.
3. Term columns that are zero across **all retained documents** are dropped **in place**, preserving the order of surviving columns. `vocabulary` is the surviving column order.
4. Word tokens are expanded in `np.nonzero` row-major order over the `D x W` matrix, where **rows are in input-file order, not `docidx` order**. Iterate rows as they appear in the file; emit the mapped `docidx`. Within a row, ascending word index, each repeated by its count.
5. Coordinate rows keep input-file order, filtered to retained IDs.

- [ ] **Step 1: Add the fixture generator**

Append to `nimare/tests/generate_gclda_fixtures.py`, and add `gen_ingest()` to `__main__`:

```python
def gen_ingest():
    """Pin the constructor's index-determining behavior.

    Deliberately adversarial: document IDs are NOT in sorted order in the
    file, IDs differ between the two tables, one term is all-zero, and
    string sorting differs from numeric sorting ("10" < "9").
    """
    import pandas as pd

    from nimare.annotate.gclda import GCLDAModel
    from nimare.utils import get_template

    counts = pd.DataFrame(
        {
            "alpha": [2, 0, 1, 3],
            "beta": [0, 0, 0, 0],  # dropped: zero everywhere
            "gamma": [1, 4, 0, 0],
            "delta": [0, 2, 5, 1],
        },
        index=["9", "10", "2", "extra_count_only"],
    )
    coords = pd.DataFrame(
        {
            "id": ["2", "9", "9", "10", "10", "10", "coord_only"],
            "x": [10.0, -20.0, 30.0, -5.0, 15.0, -25.0, 0.0],
            "y": [-30.0, 40.0, -50.0, 12.0, -22.0, 32.0, 0.0],
            "z": [50.0, -60.0, 20.0, -18.0, 28.0, -38.0, 0.0],
        }
    )

    counts.to_csv(os.path.join(FIXTURE_DIR, "counts.tsv"), sep="\t", index_label="id")
    coords.to_csv(os.path.join(FIXTURE_DIR, "coordinates.tsv"), sep="\t", index=False)

    model = GCLDAModel(counts, coords, mask=get_template("mni152_2mm", mask="brain"), n_topics=3)
    write(
        "ingest.json",
        {
            "ids": list(model.ids),
            "vocabulary": list(model.vocabulary),
            "wtoken_doc_idx": model.data["wtoken_doc_idx"].tolist(),
            "wtoken_word_idx": model.data["wtoken_word_idx"].tolist(),
            "ptoken_doc_idx": model.data["ptoken_doc_idx"].tolist(),
            "ptoken_coords": [[f64_bits(v) for v in row] for row in model.data["ptoken_coords"]],
        },
    )
```

Regenerate the fixtures and **read `ingest.json`** — it is the specification for this task. Note in particular that `ids` will be `["10", "2", "9"]`, because these are sorted as strings.

- [ ] **Step 2: Write the failing test**

`rust/gclda/tests/ingest_golden.rs`:

```rust
use gclda::io::tsv::load_corpus;
use std::path::PathBuf;

mod common;
use common::{bits_to_f64, load};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

#[test]
fn ingest_matches_python_constructor() {
    let expected = load("ingest.json");
    let corpus = load_corpus(&fixture("counts.tsv"), &fixture("coordinates.tsv")).unwrap();

    let want_ids: Vec<String> = expected["ids"]
        .as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    assert_eq!(corpus.ids, want_ids, "document IDs (sorted as STRINGS)");

    let want_vocab: Vec<String> = expected["vocabulary"]
        .as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    assert_eq!(corpus.vocabulary, want_vocab, "vocabulary after dropping all-zero terms");

    let as_u32 = |k: &str| -> Vec<u32> {
        expected[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect()
    };
    assert_eq!(corpus.wtoken_doc_idx, as_u32("wtoken_doc_idx"));
    assert_eq!(corpus.wtoken_word_idx, as_u32("wtoken_word_idx"));
    assert_eq!(corpus.ptoken_doc_idx, as_u32("ptoken_doc_idx"));

    let want_coords = expected["ptoken_coords"].as_array().unwrap();
    assert_eq!(corpus.ptoken_coords.len(), want_coords.len());
    for (i, row) in want_coords.iter().enumerate() {
        let r = row.as_array().unwrap();
        for j in 0..3 {
            let want = bits_to_f64(r[j].as_str().unwrap());
            assert_eq!(corpus.ptoken_coords[i][j].to_bits(), want.to_bits(), "coord[{i}][{j}]");
        }
    }
}
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd rust/gclda && cargo test --test ingest_golden`

Expected: FAIL to compile — `io::tsv` does not exist.

- [ ] **Step 4: Implement**

Create `rust/gclda/src/io/tsv.rs` implementing `load_corpus` per the five semantics above. Structure it as:

```rust
//! Streaming TSV ingest.
//!
//! Reads counts and coordinates directly into token-level index arrays. The
//! dense D x W count matrix that the Python constructor materializes (~340 MB
//! at Neurosynth scale) is never built.
//!
//! The index semantics here are load-bearing; see the task notes. In
//! particular, IDs sort as STRINGS ("10" < "2" < "9"), and count rows are
//! traversed in FILE order, not docidx order.

use crate::GcldaError;
use std::path::Path;

pub struct Corpus {
    pub ids: Vec<String>,
    pub vocabulary: Vec<String>,
    pub wtoken_doc_idx: Vec<u32>,
    pub wtoken_word_idx: Vec<u32>,
    pub ptoken_doc_idx: Vec<u32>,
    pub ptoken_coords: Vec<[f64; 3]>,
}

pub fn load_corpus(counts: &Path, coords: &Path) -> Result<Corpus, GcldaError> {
    // Pass 1 over counts: header -> term names; first column -> count IDs, in file order.
    // Pass 1 over coordinates: collect coordinate IDs.
    // ids = sorted intersection (String sort); docidx = position in ids.
    // Pass 2 over counts: for retained rows only, accumulate per-column totals to
    //   identify all-zero terms; retain (row_docidx, col, count) triples.
    // vocabulary = term names whose total is nonzero, in original column order.
    // Expand tokens: for each retained row IN FILE ORDER, for each surviving column
    //   in ascending index with count > 0, push (docidx, new_word_idx) `count` times.
    // Pass 2 over coordinates: for retained rows in file order, push docidx and [x, y, z].
    todo!("implement per the semantics above; the golden test is the specification")
}
```

Requirements the implementation must satisfy:
- Parse `x`, `y`, `z` by **column name** from the coordinate header, not by position — the fixture has `id, x, y, z` but real files (e.g. `data-neurosynth_version-7_coordinates.tsv.gz`) have `id, table_id, table_num, peak_id, x, y, z`.
- Coordinate IDs may repeat; count IDs are unique.
- Float parsing must produce the same `f64` as Python's; Rust's `str::parse::<f64>()` and Python's `float()` are both correctly-rounded, so they agree. The golden test verifies this.
- Two passes over each file, streaming — do not read whole files into memory.

- [ ] **Step 5: Run to verify it passes**

Run: `cd rust/gclda && cargo test --test ingest_golden`

Expected: PASS. If `ids` mismatches, the likely cause is numeric rather than string sorting.

- [ ] **Step 6: Commit**

```bash
git add rust/gclda/src/io/tsv.rs rust/gclda/src/io/mod.rs \
        rust/gclda/tests/ingest_golden.rs \
        rust/gclda/tests/fixtures/ingest.json \
        rust/gclda/tests/fixtures/counts.tsv \
        rust/gclda/tests/fixtures/coordinates.tsv \
        nimare/tests/generate_gclda_fixtures.py
git diff --cached --stat
git commit -m "[ENH] Add streaming TSV ingest to Rust GCLDA crate

Reads counts and coordinates directly into token-level index arrays,
never materializing the dense document-by-term matrix.

Replicates the Python constructor's index-determining behavior exactly:
string-sorted ID intersection, in-place dropping of all-zero terms, and
token expansion in file-row order rather than docidx order. Verified
against a golden fixture built from deliberately adversarial input where
string and numeric ID ordering disagree.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: NIfTI mask reader

**Files:**
- Create: `rust/gclda/src/io/nifti.rs`
- Modify: `rust/gclda/src/io/mod.rs`
- Modify: `nimare/tests/generate_gclda_fixtures.py` (add `gen_mask`)
- Create: `rust/gclda/tests/fixtures/mask_xyz.json`

**Interfaces:**
- Consumes: `GcldaError`
- Produces: `gclda::io::nifti::load_mask_xyz(path: &Path) -> Result<MaskInfo, GcldaError>` where
  ```rust
  pub struct MaskInfo {
      pub xyz: Vec<[f64; 3]>,   // one row per nonzero voxel, C-order over (i, j, k)
      pub affine: [[f64; 4]; 4],
      pub shape: [usize; 3],
  }
  ```

**Reference behavior (`gclda.py:475-480`):**
```python
mask_ijk = np.vstack(np.where(_mask_img_to_bool(self.mask))).T
mask_xyz = nib.affines.apply_affine(self.mask.affine, mask_ijk)
```
`_mask_img_to_bool` is `np.asanyarray(dataobj).astype(bool)` — plain nonzero, **not** a threshold. `np.where` returns C-order indices (i slowest, k fastest). `apply_affine` computes `xyz = M @ ijk + t` where `M` is `affine[:3, :3]` and `t` is `affine[:3, 3]`.

- [ ] **Step 1: Add the fixture generator**

Append to `nimare/tests/generate_gclda_fixtures.py`, add `gen_mask()` to `__main__`:

```python
def gen_mask():
    """Pin mask loading: affine, nonzero rule, index order, and coordinates."""
    import nibabel as nib

    from nimare.utils import _mask_img_to_bool, get_resource_path

    path = os.path.join(
        get_resource_path(), "templates", "MNI152_2x2x2_brainmask.nii.gz"
    )
    img = nib.load(path)
    mask_ijk = np.vstack(np.where(_mask_img_to_bool(img))).T
    mask_xyz = nib.affines.apply_affine(img.affine, mask_ijk)

    # The full array is ~228k rows; pin the shape, a checksum, and a sample.
    sample_idx = list(range(0, len(mask_xyz), max(1, len(mask_xyz) // 500)))
    write(
        "mask_xyz.json",
        {
            # Repo-RELATIVE. Never store an absolute path in a committed fixture:
            # the test would then only pass on the machine that generated it.
            # Rust joins this against CARGO_MANIFEST_DIR/.. to reach the repo root.
            "path": os.path.relpath(path, REPO_ROOT),
            "shape": [int(d) for d in img.shape],
            "affine": [[f64_bits(v) for v in row] for row in img.affine],
            "n_voxels": int(len(mask_xyz)),
            "sum_bits": [f64_bits(v) for v in mask_xyz.sum(axis=0)],
            "sample_indices": sample_idx,
            "sample_xyz": [[f64_bits(v) for v in mask_xyz[i]] for i in sample_idx],
        },
    )
```

Regenerate fixtures. Expect `n_voxels == 228483` and `shape == [91, 109, 91]`.

- [ ] **Step 2: Write the failing test**

`rust/gclda/tests/mask_golden.rs`:

```rust
use gclda::io::nifti::load_mask_xyz;
use std::path::Path;

mod common;
use common::{bits_to_f64, load};

#[test]
fn mask_xyz_matches_nibabel() {
    let expected = load("mask_xyz.json");
    let info = load_mask_xyz(&repo_path(expected["path"].as_str().unwrap())).unwrap();

    let want_shape: Vec<usize> = expected["shape"]
        .as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    assert_eq!(info.shape.to_vec(), want_shape);

    for i in 0..4 {
        for j in 0..4 {
            let want = bits_to_f64(
                expected["affine"].as_array().unwrap()[i].as_array().unwrap()[j].as_str().unwrap(),
            );
            assert_eq!(info.affine[i][j].to_bits(), want.to_bits(), "affine[{i}][{j}]");
        }
    }

    assert_eq!(info.xyz.len(), expected["n_voxels"].as_u64().unwrap() as usize);

    // Column sums catch any ordering or off-by-one error across all voxels.
    let mut sums = [0.0f64; 3];
    for row in &info.xyz {
        for j in 0..3 {
            sums[j] += row[j];
        }
    }
    for j in 0..3 {
        let want = bits_to_f64(expected["sum_bits"].as_array().unwrap()[j].as_str().unwrap());
        assert!(
            (sums[j] - want).abs() <= want.abs() * 1e-12,
            "column sum {j}: got {} want {want}", sums[j]
        );
    }

    // Sampled rows verify exact ordering, not just aggregate agreement.
    let idx = expected["sample_indices"].as_array().unwrap();
    let xyz = expected["sample_xyz"].as_array().unwrap();
    for (s, i) in idx.iter().enumerate() {
        let i = i.as_u64().unwrap() as usize;
        for j in 0..3 {
            let want = bits_to_f64(xyz[s].as_array().unwrap()[j].as_str().unwrap());
            assert_eq!(info.xyz[i][j].to_bits(), want.to_bits(), "xyz[{i}][{j}]");
        }
    }
}
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd rust/gclda && cargo test --test mask_golden`

Expected: FAIL to compile.

- [ ] **Step 4: Implement**

Create `rust/gclda/src/io/nifti.rs`. Write a minimal NIfTI-1 reader rather than taking a heavy dependency — only a few header fields are needed, and matching nibabel's affine selection is the actual risk, which the golden test covers.

Required behavior:
- Detect gzip by the `1f 8b` magic; decompress with `flate2::read::GzDecoder`. `.nii` and `.nii.gz` must both work.
- Parse the 348-byte NIfTI-1 header, little-endian. Detect endianness by checking `sizeof_hdr == 348`; if it reads as `1543569408`, byte-swap. Fields needed, at these byte offsets: `sizeof_hdr` (0), `dim[8]` i16 (40), `datatype` i16 (70), `bitpix` i16 (72), `vox_offset` f32 (108), `scl_slope` f32 (112), `scl_inter` f32 (116), `qform_code` i16 (252), `sform_code` i16 (254), `quatern_b/c/d` f32 (256/260/264), `qoffset_x/y/z` f32 (268/272/276), `srow_x[4]` f32 (280), `srow_y[4]` f32 (296), `srow_z[4]` f32 (312).
- **Affine selection, matching nibabel's `get_best_affine`:** if `sform_code != 0`, use the `srow_*` rows; else if `qform_code != 0`, build from the quaternion; else fall back to a diagonal from `pixdim`. **The bundled mask has `sform_code = 4`, so the sform path is the one exercised.** Implement the quaternion path but keep it simple; if a test later needs it, the golden test will catch errors.
- Note `srow_*` are `f32` in the file and widened to `f64`. nibabel does the same, so widening reproduces its affine exactly.
- Support at minimum `datatype` 2 (uint8), 4 (int16), 8 (int32), 16 (float32), 64 (float64). The bundled mask is uint8.
- **`scl_slope` is `nan` in the bundled mask.** Treat `nan` or `0.0` slope as "no scaling" (slope 1.0, inter 0.0), matching nibabel.
- Voxel data starts at `vox_offset` read from the raw header. The bundled mask's is **448**, not
  0 (see the corrected fact 6 above — nibabel's normalized `header['vox_offset']` misleadingly
  reports 0). Only when the raw `vox_offset` is genuinely 0 does data begin at offset **352** for
  a single-file `.nii` — the 348-byte header plus the 4-byte extension field.
- Nonzero test: a voxel is in the mask iff its **raw** value is nonzero. Do not threshold. Do not apply scaling before the test — `astype(bool)` in nibabel operates on `dataobj`, which for these masks is the raw array.
- Iterate in **C order**: `i` outermost, `k` innermost, matching `np.where`.
- Compute `xyz[d] = affine[d][0]*i + affine[d][1]*j + affine[d][2]*k + affine[d][3]` for `d` in 0..3, in that operation order.

- [ ] **Step 5: Run to verify it passes**

Run: `cd rust/gclda && cargo test --test mask_golden`

Expected: PASS — 228483 voxels, exact affine, exact sampled coordinates.

- [ ] **Step 6: Commit**

```bash
git add rust/gclda/src/io/nifti.rs rust/gclda/src/io/mod.rs \
        rust/gclda/tests/mask_golden.rs rust/gclda/tests/fixtures/mask_xyz.json \
        nimare/tests/generate_gclda_fixtures.py
git diff --cached --stat
git commit -m "[ENH] Add NIfTI mask reader to Rust GCLDA crate

Minimal NIfTI-1 reader covering the fields needed for masking, with
nibabel-compatible affine selection (sform, then qform, then pixdim) and
gzip support. Reproduces nibabel's mask_xyz bit-for-bit over all 228483
voxels of the bundled MNI152 2mm brain mask.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

# Phase 2 — Model and samplers

### Task 8: Model struct and initialization

**Files:**
- Create: `rust/gclda/src/model.rs`
- Modify: `rust/gclda/src/lib.rs`
- Modify: `nimare/tests/generate_gclda_fixtures.py` (add `gen_init_state`)
- Create: `rust/gclda/tests/fixtures/init_state.json`

**Interfaces:**
- Consumes: `Corpus`, `MaskInfo`, `Mt19937`
- Produces:
  ```rust
  pub struct Params {
      pub n_topics: usize, pub n_regions: usize, pub symmetric: bool,
      pub alpha: f64, pub beta: f64, pub gamma: f64, pub delta: f64,
      pub dobs: f64, pub roi_size: f64, pub seed_init: u32,
  }
  pub struct Model { /* see below */ }
  impl Model {
      pub fn new(corpus: Corpus, mask: MaskInfo, params: Params) -> Result<Model, GcldaError>;
  }
  ```
  Fields (all row-major flat `Vec`s; helper accessors `at(row, col)`):
  `wtoken_topic_idx: Vec<u32>`, `peak_topic_idx: Vec<u32>`, `peak_region_idx: Vec<u32>`,
  `n_peak_tokens_doc_by_topic: Vec<i64>` (D x T), `n_peak_tokens_region_by_topic: Vec<i64>` (R x T),
  `n_word_tokens_word_by_topic: Vec<i64>` (W x T), `n_word_tokens_doc_by_topic: Vec<i64>` (D x T),
  `total_n_word_tokens_by_topic: Vec<i64>` (T),
  `regions_mu: Vec<[f64; 3]>` (T*R), `regions_sigma: Vec<[[f64; 3]; 3]>` (T*R),
  `regions_precision: Vec<[[f64; 3]; 3]>` (T*R), `regions_log_norm: Vec<f64>` (T*R),
  `iter: usize`, `seed: u32`

**Reference behavior (`gclda.py:439-658`) — exact RNG consumption order:**

1. Validate: `symmetric && n_regions % 2 != 0` is an error.
2. `np.random.seed(seed_init)` — one RNG stream for everything that follows in the constructor.
3. `peak_topic_idx = randint(n_topics, size=n_peaks)` — consumes `n_peaks` bounded draws.
4. If symmetric: `initial = randint(n_pairs, size=n_peaks)`, then `peak_region_idx = initial * 2 + (x > 0)`. **Note `n_pairs` may be 1, and `np.random.randint(1, size=n)` still consumes draws** — with `bound = 1`, `rng_range = 0`, so the Rust `randint` returns 0 immediately **without consuming a draw**. NumPy's behavior for `bound == 1` must be confirmed against the fixture; if the fixture shows a different subsequent stream, adjust `randint` to consume a draw when `rng_range == 0`.
   If asymmetric: `peak_region_idx = randint(n_regions, size=n_peaks)`.
5. Accumulate `n_peak_tokens_doc_by_topic` and `n_peak_tokens_region_by_topic` from those assignments.
6. Call the word-topic initializer, which does `np.random.seed(seed_init)` **again** (`gclda.py:141`) — re-seeding the same stream. For each word token in order: `probs[t] = peak_doc_by_topic[doc][t] + gamma`, sample, then increment `word_by_topic`, `total_word_by_topic`, `word_doc_by_topic`.
7. `self.iter = 0`, `self.seed = 0`.

- [ ] **Step 1: Add the fixture generator**

Append to `nimare/tests/generate_gclda_fixtures.py`, add `gen_init_state()` to `__main__`:

```python
def gen_init_state():
    """Pin the full post-constructor state for several configurations."""
    import pandas as pd

    from nimare.annotate.gclda import GCLDAModel
    from nimare.utils import get_template

    counts = pd.read_csv(os.path.join(FIXTURE_DIR, "counts.tsv"), sep="\t", index_col="id")
    counts.index = counts.index.astype(str)
    coords = pd.read_csv(os.path.join(FIXTURE_DIR, "coordinates.tsv"), sep="\t")
    coords["id"] = coords["id"].astype(str)
    mask = get_template("mni152_2mm", mask="brain")

    configs = [
        {"n_topics": 3, "n_regions": 2, "symmetric": True, "seed_init": 1},
        {"n_topics": 4, "n_regions": 4, "symmetric": True, "seed_init": 7},
        {"n_topics": 3, "n_regions": 1, "symmetric": False, "seed_init": 1},
        {"n_topics": 5, "n_regions": 3, "symmetric": False, "seed_init": 42},
    ]
    out = []
    for cfg in configs:
        model = GCLDAModel(counts, coords, mask=mask, **cfg)
        out.append(
            {
                "config": cfg,
                "wtoken_topic_idx": model.topics["wtoken_topic_idx"].tolist(),
                "peak_topic_idx": model.topics["peak_topic_idx"].tolist(),
                "peak_region_idx": model.topics["peak_region_idx"].tolist(),
                "n_peak_tokens_doc_by_topic": model.topics[
                    "n_peak_tokens_doc_by_topic"].tolist(),
                "n_peak_tokens_region_by_topic": model.topics[
                    "n_peak_tokens_region_by_topic"].tolist(),
                "n_word_tokens_word_by_topic": model.topics[
                    "n_word_tokens_word_by_topic"].tolist(),
                "n_word_tokens_doc_by_topic": model.topics[
                    "n_word_tokens_doc_by_topic"].tolist(),
                "total_n_word_tokens_by_topic": model.topics[
                    "total_n_word_tokens_by_topic"].tolist(),
            }
        )
    write("init_state.json", out)
```

- [ ] **Step 2: Write the failing test**

`rust/gclda/tests/init_golden.rs` — for each config in `init_state.json`, build a `Model` and assert every listed array matches exactly:

```rust
use gclda::io::{nifti::load_mask_xyz, tsv::load_corpus};
use gclda::model::{Model, Params};
use std::path::{Path, PathBuf};

mod common;
use common::load;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

#[test]
fn initial_state_matches_python_constructor() {
    let mask_meta = load("mask_xyz.json");
    let mask_path = common::repo_path(mask_meta["path"].as_str().unwrap());

    for (c, case) in load("init_state.json").as_array().unwrap().iter().enumerate() {
        let cfg = &case["config"];
        let params = Params {
            n_topics: cfg["n_topics"].as_u64().unwrap() as usize,
            n_regions: cfg["n_regions"].as_u64().unwrap() as usize,
            symmetric: cfg["symmetric"].as_bool().unwrap(),
            alpha: 0.1, beta: 0.01, gamma: 0.01, delta: 1.0,
            dobs: 25.0, roi_size: 50.0,
            seed_init: cfg["seed_init"].as_u64().unwrap() as u32,
        };
        let corpus = load_corpus(&fixture("counts.tsv"), &fixture("coordinates.tsv")).unwrap();
        let mask = load_mask_xyz(&mask_path).unwrap();
        let model = Model::new(corpus, mask, params).unwrap();

        let want_u32 = |k: &str| -> Vec<u32> {
            case[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect()
        };
        assert_eq!(model.peak_topic_idx, want_u32("peak_topic_idx"), "case {c} peak_topic_idx");
        assert_eq!(model.peak_region_idx, want_u32("peak_region_idx"), "case {c} peak_region_idx");
        assert_eq!(model.wtoken_topic_idx, want_u32("wtoken_topic_idx"), "case {c} wtoken_topic_idx");

        let flat = |k: &str| -> Vec<i64> {
            case[k].as_array().unwrap().iter()
                .flat_map(|row| match row.as_array() {
                    Some(r) => r.iter().map(|v| v.as_i64().unwrap()).collect::<Vec<_>>(),
                    None => vec![row.as_i64().unwrap()],
                })
                .collect()
        };
        assert_eq!(model.n_peak_tokens_doc_by_topic, flat("n_peak_tokens_doc_by_topic"), "case {c}");
        assert_eq!(model.n_peak_tokens_region_by_topic, flat("n_peak_tokens_region_by_topic"), "case {c}");
        assert_eq!(model.n_word_tokens_word_by_topic, flat("n_word_tokens_word_by_topic"), "case {c}");
        assert_eq!(model.n_word_tokens_doc_by_topic, flat("n_word_tokens_doc_by_topic"), "case {c}");
        assert_eq!(model.total_n_word_tokens_by_topic, flat("total_n_word_tokens_by_topic"), "case {c}");
    }
}

#[test]
fn symmetric_with_odd_regions_is_rejected() {
    let corpus = load_corpus(&fixture("counts.tsv"), &fixture("coordinates.tsv")).unwrap();
    let mask_meta = load("mask_xyz.json");
    let mask = load_mask_xyz(&common::repo_path(mask_meta["path"].as_str().unwrap())).unwrap();
    let params = Params {
        n_topics: 3, n_regions: 3, symmetric: true,
        alpha: 0.1, beta: 0.01, gamma: 0.01, delta: 1.0,
        dobs: 25.0, roi_size: 50.0, seed_init: 1,
    };
    assert!(Model::new(corpus, mask, params).is_err());
}
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd rust/gclda && cargo test --test init_golden`

Expected: FAIL to compile.

- [ ] **Step 4: Implement `Model::new`**

Follow the 7-step RNG consumption order given above exactly. The `n_regions = 1, symmetric = false` config in the fixture specifically exercises the `bound == 1` question flagged in step 4 of that list — if `peak_region_idx` mismatches for that config, `randint` must consume a draw when `rng_range == 0`. Resolve it against the fixture, and add a note in `rng.rs` recording which behavior was correct.

- [ ] **Step 5: Run to verify it passes**

Run: `cd rust/gclda && cargo test`

Expected: PASS — all four configurations.

- [ ] **Step 6: Commit**

```bash
git add rust/gclda/src/model.rs rust/gclda/src/lib.rs rust/gclda/src/rng.rs \
        rust/gclda/tests/init_golden.rs rust/gclda/tests/fixtures/init_state.json \
        nimare/tests/generate_gclda_fixtures.py
git diff --cached --stat
git commit -m "[ENH] Add GCLDA model state and initialization to Rust crate

Reproduces the Python constructor's random assignment and count
initialization exactly, including its RNG consumption order and the
re-seed before word-topic initialization. Verified across symmetric and
asymmetric configurations with 1 to 4 subregions.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Word-topic sampler

**Files:**
- Create: `rust/gclda/src/sampler/mod.rs`, `rust/gclda/src/sampler/words.rs`
- Modify: `rust/gclda/src/lib.rs`

**Interfaces:**
- Consumes: `Model`, `Mt19937`
- Produces: `impl Model { pub fn update_word_topic_assignments(&mut self, seed: u32) -> Result<(), GcldaError> }`

**Reference (`_jit_update_word_topic_assignments`, `gclda.py:213-250`).** Must be sequential. Re-seeds the RNG at entry. For each word token in order: decrement its three counts, compute
`probs[t] = ((word_by_topic[word][t] + beta) / (total_word_by_topic[t] + beta_vocabulary)) * (peak_doc_by_topic[doc][t] + gamma)`
with `beta_vocabulary = beta * vocabulary.len()`, sample, then re-increment. **Keep the division as a division.**

- [ ] **Step 1: Write the failing test**

`rust/gclda/tests/sampler_words.rs`. Generate its fixture by adding to `generate_gclda_fixtures.py`:

```python
def gen_word_sampler():
    """Pin one word-topic sampling sweep."""
    import pandas as pd

    from nimare.annotate.gclda import GCLDAModel
    from nimare.utils import get_template

    counts = pd.read_csv(os.path.join(FIXTURE_DIR, "counts.tsv"), sep="\t", index_col="id")
    counts.index = counts.index.astype(str)
    coords = pd.read_csv(os.path.join(FIXTURE_DIR, "coordinates.tsv"), sep="\t")
    coords["id"] = coords["id"].astype(str)

    model = GCLDAModel(
        counts, coords, mask=get_template("mni152_2mm", mask="brain"),
        n_topics=3, n_regions=2, symmetric=True, seed_init=1,
    )
    model._update_word_topic_assignments(1)
    write(
        "word_sampler.json",
        {
            "seed": 1,
            "wtoken_topic_idx": model.topics["wtoken_topic_idx"].tolist(),
            "n_word_tokens_word_by_topic": model.topics["n_word_tokens_word_by_topic"].tolist(),
            "n_word_tokens_doc_by_topic": model.topics["n_word_tokens_doc_by_topic"].tolist(),
            "total_n_word_tokens_by_topic": model.topics["total_n_word_tokens_by_topic"].tolist(),
        },
    )
```

The Rust test builds the same model (n_topics=3, n_regions=2, symmetric, seed_init=1), calls `update_word_topic_assignments(1)`, and asserts all four arrays match exactly.

- [ ] **Step 2: Run to verify it fails.** Run: `cd rust/gclda && cargo test --test sampler_words`. Expected: FAIL to compile.
- [ ] **Step 3: Implement** per the reference above.
- [ ] **Step 4: Run to verify it passes.** Run: `cd rust/gclda && cargo test --test sampler_words`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git add rust/gclda/src/sampler/mod.rs rust/gclda/src/sampler/words.rs \
        rust/gclda/src/lib.rs rust/gclda/tests/sampler_words.rs \
        rust/gclda/tests/fixtures/word_sampler.json \
        nimare/tests/generate_gclda_fixtures.py
git diff --cached --stat
git commit -m "[ENH] Add word-topic sampler to Rust GCLDA crate

Sequential collapsed Gibbs sweep over word tokens, matching the Python
implementation's arithmetic and RNG consumption exactly.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Peak sampler with fused PDF

The central optimization. Python materializes an `n_peaks x T x R` array (~800 MB at Neurosynth scale) each iteration and streams it once; Rust computes each peak's `T x R` block into a reusable buffer.

**Files:**
- Create: `rust/gclda/src/sampler/peaks.rs`
- Modify: `rust/gclda/src/sampler/mod.rs`

**Interfaces:**
- Consumes: `Model`, `Mt19937`, `gaussian::pdf`
- Produces:
  - `impl Model { pub fn update_peak_assignments(&mut self, seed: u32) -> Result<(), GcldaError> }`
  - `impl Model { pub fn peak_probs_for(&self, i_peak: usize, out: &mut [f64]) }` — fills a `T*R` buffer indexed `topic * n_regions + region`, used by both this sampler and the log-likelihood

**Reference (`_jit_update_peak_assignments`, `gclda.py:254-324`).** Sequential; re-seeds at entry. Precompute `region_totals[t] = sum_r region_by_topic[r][t]` once. Per peak: decrement, compute `peak_topic_probs` via the log1p/max-subtract/exp stabilization exactly as written, build `probs_pdf` in **region-major, topic-minor flat order** (`flat_idx` increments with `j_region` outer, `i_topic` inner — see `gclda.py:302-314`), sample, then decode `region = idx / n_topics`, `topic = idx % n_topics`, and re-increment.

**Critical:** `probs_pdf` ordering is region-outer/topic-inner, while `peak_probs_for` is indexed topic-outer/region-inner. Do not conflate them.

- [ ] **Step 1: Write the failing test.** Add `gen_peak_sampler()` to the fixture generator, mirroring `gen_word_sampler` but calling `model._update_regions()` then `model._update_peak_assignments(2)`, dumping `peak_topic_idx`, `peak_region_idx`, `n_peak_tokens_doc_by_topic`, `n_peak_tokens_region_by_topic`. Rust test asserts all four match exactly.
- [ ] **Step 2: Run to verify it fails.** Run: `cd rust/gclda && cargo test --test sampler_peaks`.
- [ ] **Step 3: Implement,** with the PDF computed inline per peak into a reusable `Vec<f64>` of length `T*R` allocated once outside the loop.
- [ ] **Step 4: Run to verify it passes.** Run: `cd rust/gclda && cargo test --test sampler_peaks`.
- [ ] **Step 5: Commit**

```bash
git add rust/gclda/src/sampler/peaks.rs rust/gclda/src/sampler/mod.rs \
        rust/gclda/tests/sampler_peaks.rs \
        rust/gclda/tests/fixtures/peak_sampler.json \
        nimare/tests/generate_gclda_fixtures.py
git diff --cached --stat
git commit -m "[ENH] Add peak sampler with fused PDF evaluation to Rust GCLDA crate

Computes each peak's topic-by-region Gaussian block into a reusable
buffer instead of materializing the full n_peaks x n_topics x n_regions
array that the Python implementation allocates every iteration
(~800 MB at Neurosynth scale). Arithmetic and RNG consumption are
unchanged, so results match bit-for-bit.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: Region update

**Files:**
- Create: `rust/gclda/src/sampler/regions.rs`
- Modify: `rust/gclda/src/sampler/mod.rs`

**Interfaces:**
- Consumes: `Model`, `gaussian::{inv3_logdet, log_norm}`, rayon
- Produces: `impl Model { pub fn update_regions(&mut self) -> Result<(), GcldaError> }`

**Reference (`_update_regions`, `gclda.py:827-961`).** Accumulate per (region, topic) sums and cross-products over all peaks (`_jit_accumulate_region_stats`), then per topic compute means and regularized covariances. Symmetric and asymmetric paths differ — implement both, following `gclda.py:847-927` and `gclda.py:928-961` respectively.

`_compute_covariance_from_stats` is `(cross - outer(sum, sum) / n_obs) / (n_obs - 1)`. Preserve that order.

**Parallelism:** rayon is permitted over topics in the parameter-computation phase, and over chunks in the accumulation phase with a per-chunk reduction. **Integer counts and sums must reduce deterministically** — floating-point summation is not associative, so a rayon reduction over peaks can change results. Accumulate per-chunk into separate buffers and combine them in a **fixed chunk order**, or keep accumulation sequential if the golden test shows any drift. Correctness first: get it passing sequentially, then parallelize and confirm the test still passes.

- [ ] **Step 1: Write the failing test.** Add `gen_region_update()` dumping `regions_mu`, `regions_sigma`, `regions_precision`, `regions_log_norm` (as hex bits) after `model._update_regions()`, for symmetric (R=2, R=4) and asymmetric (R=1, R=3) configs. Assert bit-equality.
- [ ] **Step 2: Run to verify it fails.** Run: `cd rust/gclda && cargo test --test sampler_regions`.
- [ ] **Step 3: Implement sequentially first.**
- [ ] **Step 4: Run to verify it passes.** Run: `cd rust/gclda && cargo test --test sampler_regions`.
- [ ] **Step 5: Add rayon parallelism, then re-run the same test.** If it fails, revert to sequential accumulation and note why in a comment.
- [ ] **Step 6: Commit**

```bash
git add rust/gclda/src/sampler/regions.rs rust/gclda/src/sampler/mod.rs \
        rust/gclda/tests/sampler_regions.rs \
        rust/gclda/tests/fixtures/region_update.json \
        nimare/tests/generate_gclda_fixtures.py
git diff --cached --stat
git commit -m "[ENH] Add spatial region parameter update to Rust GCLDA crate

Computes per-subregion means and regularized covariances for both the
symmetric and asymmetric cases, parallelized over topics. Verified
bit-for-bit against Python for 1 to 4 subregions.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 12: Sparse log-likelihood

Python builds a dense `D x W` matrix (~340 MB, ~4.2 GFLOP at Neurosynth scale) and reads only the observed `(doc, word)` entries. Rust computes those dot products directly.

**Files:**
- Create: `rust/gclda/src/loglik.rs`
- Modify: `rust/gclda/src/lib.rs`

**Interfaces:**
- Consumes: `Model`, `Model::peak_probs_for`
- Produces: `impl Model { pub fn compute_log_likelihood(&self) -> LogLikelihood }` where `pub struct LogLikelihood { pub x: f64, pub w: f64, pub total: f64 }`

**Reference:** `compute_log_likelihood`, `gclda.py:963-1087`, **as corrected by Task 1** (no `- 1` offsets).

For the word term, instead of `p_wtoken_g_doc = docprobs_z @ wordprobs.T`, compute per token
`p = sum_t docprobs_z[doc][t] * wordprobs[word][t]`, accumulating `t` in ascending order to match the BLAS-free reference. Since Task 1's test computes the reference with `np.dot`, expect agreement to ~1e-12 rather than bit-exact for this quantity; assert with a relative tolerance of `1e-10` and document why in a comment. **This is the one quantity in the port that is not asserted bit-exact**, because the Python reference routes it through BLAS.

Cache `docprobs_y`, `docprobs_z`, `regionprobs`, `wordprobs` once per call.

- [ ] **Step 1: Write the failing test.** Add `gen_loglik()` dumping `(x, w, total)` as hex bits after `_update_regions()`, for two configs. Rust test asserts `x` and `total` within `1e-10` relative and `w` within `1e-10` relative.
- [ ] **Step 2: Run to verify it fails.** Run: `cd rust/gclda && cargo test --test loglik`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run to verify it passes.** Run: `cd rust/gclda && cargo test --test loglik`.
- [ ] **Step 5: Commit**

```bash
git add rust/gclda/src/loglik.rs rust/gclda/src/lib.rs \
        rust/gclda/tests/loglik.rs rust/gclda/tests/fixtures/loglik.json \
        nimare/tests/generate_gclda_fixtures.py
git diff --cached --stat
git commit -m "[ENH] Add sparse log-likelihood to Rust GCLDA crate

Computes per-token posterior predictive probabilities directly instead
of materializing the dense document-by-word matrix the Python
implementation builds (~340 MB and ~4.2 GFLOP at Neurosynth scale),
reducing the cost from O(D*W*T) to O(nnz*T) with no extra allocation.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 13: Probability distributions and output writer

**Files:**
- Create: `rust/gclda/src/output.rs`
- Modify: `rust/gclda/src/lib.rs`

**Interfaces:**
- Consumes: `Model`, `io::npy`, rayon
- Produces:
  - `impl Model { pub fn fit(&mut self, n_iters: usize, loglikely_freq: usize) -> Result<(), GcldaError> }`
  - `pub fn write_outputs(model: &Model, dir: &Path, dtype: npy::Dtype) -> Result<(), GcldaError>`

**`fit` reference (`fit`/`_update`, `gclda.py:660-745`):** if `iter == 0`, call `update_regions()` then record log-likelihood. Then loop: `iter += 1`; `seed += 1`, `update_word_topic_assignments(seed)`; `seed += 1`, `update_peak_assignments(seed)`; `update_regions()`; if `iter % loglikely_freq == 0`, record log-likelihood.

**Distributions reference (`get_probability_distributions`, `gclda.py:1108-1156`):** `spatial_dists[v][t] = sum_r pdf(...)` (rayon over voxels is safe — each row is independent); then `p_topic_g_voxel = spatial_dists / rowsum`, `p_voxel_g_topic = spatial_dists / colsum`, `p_word_g_topic = counts / colsum`, `p_topic_g_word = counts / rowsum`, each followed by `nan_to_num`. Implement `nan_to_num` as: NaN -> 0.0, +inf -> `f64::MAX`, -inf -> `f64::MIN` (NumPy's default behavior).

**Output files** — exactly as the spec's Output directory section lists:
`p_topic_g_voxel.npy`, `p_voxel_g_topic.npy` (V x T); `p_topic_g_word.npy`, `p_word_g_topic.npy` (W x T); `vocabulary.txt`; `model.json`; `n_word_tokens_word_by_topic.npy` (W x T, i64); `n_peak_tokens_doc_by_topic.npy` (D x T, i64); `n_peak_tokens_region_by_topic.npy` (R x T, i64); `regions_mu.npy` (T x R x 3); `regions_sigma.npy` (T x R x 3 x 3); `loglikelihood.tsv`; `wtoken_topic_idx.npy`, `peak_topic_idx.npy`, `peak_region_idx.npy` (i64).

`model.json` contains: all `Params` fields, `n_iters`, `loglikely_freq`, `ids` (document order), `mask_path`, `mask_affine` (4x4), `mask_shape`, `n_voxels`, and per-phase timings (added in Task 19; emit zeros for now).

The two `V x T` matrices must be written with `NpyWriter` **row by row as they are computed**, never assembled in full.

- [ ] **Step 1: Write the failing test.** `rust/gclda/tests/outputs.rs`: fit 3 iterations on the fixture corpus, write to a temp dir, then shell out to NumPy to assert every file loads, has the expected shape, and that each `p_*` matrix's rows or columns sum to 1 where the corresponding normalization applies (allowing all-zero rows).
- [ ] **Step 2: Run to verify it fails.** Run: `cd rust/gclda && cargo test --test outputs`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run to verify it passes.** Run: `cd rust/gclda && cargo test --test outputs`.
- [ ] **Step 5: Commit**

```bash
git add rust/gclda/src/output.rs rust/gclda/src/lib.rs rust/gclda/tests/outputs.rs
git diff --cached --stat
git commit -m "[ENH] Add fit loop and output writer to Rust GCLDA crate

Writes the four probability matrices plus the count matrices, spatial
parameters, assignment vectors, and metadata needed for GCLDA decoding,
encoding, and resumption. Large voxel matrices stream to disk row by row
rather than being assembled in memory.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 14: CLI binary

**Files:**
- Create: `rust/gclda/src/bin/gclda-train.rs`

**Interfaces:**
- Consumes: everything above
- Produces: the `gclda-train` binary

Arguments, mirroring the Python signature: `--counts <PATH>`, `--coordinates <PATH>`, `--mask <PATH>`, `--out-dir <PATH>`, `--n-topics <INT=100>`, `--n-regions <INT=2>`, `--symmetric <BOOL=true>`, `--alpha <F64=0.1>`,
(**`--symmetric` takes an explicit value** — `--symmetric true` / `--symmetric false` — via
clap's `ArgAction::Set` with `value_parser!(bool)`, NOT a presence flag. Tasks 16 and 17 invoke
it that way, and a presence flag cannot express `false`.) `--beta <F64=0.01>`, `--gamma <F64=0.01>`, `--delta <F64=1.0>`, `--dobs <F64=25>`, `--roi-size <F64=50.0>`, `--seed-init <U32=1>`, `--n-iters <INT=5000>`, `--loglikely-freq <INT=10>`, `--output-dtype <f64|f32 = f64>`, `--threads <INT>` (0 = rayon default).

Print per-iteration progress to stderr at `loglikely_freq` intervals, in the same format as the Python `LGR.info` line, so runs can be compared side by side.

- [ ] **Step 1: Write the failing test.** `rust/gclda/tests/cli.rs`: invoke the built binary via `env!("CARGO_BIN_EXE_gclda-train")` on the fixture TSVs with `--n-iters 3`, assert exit status 0 and that all expected output files exist. Add a second test asserting a nonzero exit and a useful stderr message when `--symmetric true --n-regions 3`.
- [ ] **Step 2: Run to verify it fails.** Run: `cd rust/gclda && cargo test --test cli`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run to verify it passes.** Run: `cd rust/gclda && cargo test --release --test cli`.
- [ ] **Step 5: Commit**

```bash
git add rust/gclda/src/bin/gclda-train.rs rust/gclda/tests/cli.rs
git diff --cached --stat
git commit -m "[ENH] Add gclda-train CLI to Rust GCLDA crate

Mirrors the Python GCLDAModel constructor and fit signatures, reading
counts and coordinates from TSV and a mask from NIfTI.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

# Phase 3 — Regression harness

### Task 15: Python loader

**Files:**
- Create: `nimare/annotate/gclda_rs.py`
- Modify: `nimare/annotate/__init__.py`
- Test: `nimare/tests/test_gclda_rust.py`

**Interfaces:**
- Consumes: the Rust output directory
- Produces:
  - `export_gclda_tsvs(count_df, coordinates_df, out_dir) -> (counts_path, coords_path)`
  - `load_gclda_model(model_dir, mask=None, mmap=True) -> GCLDAResult`
  - `train_gclda_rust(count_df, coordinates_df, mask, out_dir, binary=None, **params) -> GCLDAResult`
  - `class GCLDAResult` exposing `.mask`, `.vocabulary`, `.ids`, `.params`, `.p_topic_g_voxel_`, `.p_voxel_g_topic_`, `.p_topic_g_word_`, `.p_word_g_topic_`, `.n_word_tokens_word_by_topic`, `.n_peak_tokens_doc_by_topic`, `.regions_mu`, `.regions_sigma`, `.loglikelihood`

`GCLDAResult` must expose exactly the attribute names the existing decoders use, so `nimare/decode/` needs no changes: `gclda_decode_map` uses `.mask`, `.p_topic_g_voxel_`, `.p_word_g_topic_`, `.vocabulary`; `gclda_decode_roi` uses `.mask`, `.p_topic_g_voxel_`, `.p_word_g_topic_`, `.vocabulary`; `gclda_encode` uses `.vocabulary`, `.p_topic_g_word_`, `.p_voxel_g_topic_`, `.mask`.

`export_gclda_tsvs` must write counts with `index_label="id"` and coordinates with an explicit `id` column, matching the ingest format from Task 6.

- [ ] **Step 1: Write the failing test**

Create `nimare/tests/test_gclda_rust.py`:

```python
"""Regression tests for the Rust GCLDA trainer against the Python implementation.

These tests are skipped unless the Rust binary has been built:

    cd rust/gclda && cargo build --release
"""

import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from nimare import annotate

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BINARY = os.path.join(REPO_ROOT, "rust", "gclda", "target", "release", "gclda-train")

requires_rust = pytest.mark.skipif(
    not os.path.exists(BINARY),
    reason="gclda-train not built; run `cd rust/gclda && cargo build --release`",
)


@pytest.fixture(scope="module")
def small_corpus():
    """A small, deterministic corpus shared by the regression tests."""
    rng = np.random.default_rng(0)
    n_docs, n_terms, n_peaks = 12, 8, 60
    ids = [f"study-{i:03d}" for i in range(n_docs)]
    counts = pd.DataFrame(
        rng.integers(0, 4, size=(n_docs, n_terms)),
        index=ids,
        columns=[f"term_{j}" for j in range(n_terms)],
    )
    counts.iloc[:, 0] = 0  # one all-zero term, to exercise column dropping
    doc_for_peak = rng.integers(0, n_docs, size=n_peaks)
    coords = pd.DataFrame(
        {
            "id": [ids[d] for d in doc_for_peak],
            "x": rng.uniform(-60, 60, n_peaks).round(1),
            "y": rng.uniform(-90, 60, n_peaks).round(1),
            "z": rng.uniform(-50, 70, n_peaks).round(1),
        }
    )
    return counts, coords


@requires_rust
def test_rust_loader_exposes_decoder_interface(small_corpus, mni_mask, tmp_path):
    """The loaded result must expose exactly what nimare.decode consumes."""
    counts, coords = small_corpus
    mask_path = str(tmp_path / "mask.nii.gz")
    mni_mask.to_filename(mask_path)

    result = annotate.gclda_rs.train_gclda_rust(
        counts, coords, mask=mask_path, out_dir=str(tmp_path / "out"),
        binary=BINARY, n_topics=4, n_regions=2, symmetric=True,
        seed_init=1, n_iters=3, loglikely_freq=1,
    )

    n_vox = int(np.asanyarray(mni_mask.dataobj).astype(bool).sum())
    assert result.p_topic_g_voxel_.shape == (n_vox, 4)
    assert result.p_voxel_g_topic_.shape == (n_vox, 4)
    assert result.p_topic_g_word_.shape == (len(result.vocabulary), 4)
    assert result.p_word_g_topic_.shape == (len(result.vocabulary), 4)
    assert result.mask is not None
    # The all-zero term must have been dropped.
    assert "term_0" not in result.vocabulary
    assert len(result.vocabulary) == 7
```

- [ ] **Step 2: Run to verify it fails.** Run: `micromamba run -n nimenv pytest nimare/tests/test_gclda_rust.py -v`. Expected: FAIL (or skip if the binary is absent — build it first with `cd rust/gclda && cargo build --release`).
- [ ] **Step 3: Implement `nimare/annotate/gclda_rs.py`.** Use `np.load(..., mmap_mode="r")` when `mmap=True`. Add `gclda_rs` to `nimare/annotate/__init__.py`'s imports and `__all__`, following the pattern already used there for `gclda`.
- [ ] **Step 4: Run to verify it passes.** Run: `micromamba run -n nimenv pytest nimare/tests/test_gclda_rust.py -v`.
- [ ] **Step 5: Commit**

```bash
git add nimare/annotate/gclda_rs.py nimare/annotate/__init__.py \
        nimare/tests/test_gclda_rust.py
git diff --cached --stat
git commit -m "[ENH] Add Python loader for Rust GCLDA outputs

GCLDAResult exposes the attributes nimare.decode consumes, so the
existing GCLDA decoding and encoding functions work against a
Rust-trained model with no changes. Large matrices are memory-mapped.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 16: Level 2 — per-iteration state equality

The core harness. Without it, a divergence introduced at iteration 3 is invisible until the endpoint, where it cannot be traced.

**Files:**
- Modify: `nimare/annotate/gclda.py` (add `_dump_state`)
- Modify: `rust/gclda/src/output.rs` (add `--dump-state-dir`)
- Modify: `rust/gclda/src/bin/gclda-train.rs`
- Modify: `nimare/tests/test_gclda_rust.py`

**Interfaces:**
- Consumes: `Model`, `GCLDAModel`
- Produces:
  - Python: `GCLDAModel._dump_state(out_dir, iteration)` writing `iter_{n:05d}.npz`
  - Rust: `--dump-state-dir <PATH>` writing `iter_{n:05d}/<name>.npy` after each iteration

**Dump point:** both implementations dump at the **end** of an iteration, after the region
update — Python at the end of `_update`, Rust at the end of its fit-loop body — so `regions_*`
reflect the post-update state for that iteration. Both must agree on this or Level 2 reports a
spurious `regions_*` mismatch.

Both dump, after each iteration: `wtoken_topic_idx`, `peak_topic_idx`, `peak_region_idx`, `n_peak_tokens_doc_by_topic`, `n_peak_tokens_region_by_topic`, `n_word_tokens_word_by_topic`, `n_word_tokens_doc_by_topic`, `total_n_word_tokens_by_topic`, `regions_mu`, `regions_sigma`, `regions_precision`, `regions_log_norm`.

- [ ] **Step 1: Write the failing test**

Add to `nimare/tests/test_gclda_rust.py`:

```python
INTEGER_ARRAYS = (
    "wtoken_topic_idx", "peak_topic_idx", "peak_region_idx",
    "n_peak_tokens_doc_by_topic", "n_peak_tokens_region_by_topic",
    "n_word_tokens_word_by_topic", "n_word_tokens_doc_by_topic",
    "total_n_word_tokens_by_topic",
)
FLOAT_ARRAYS = ("regions_mu", "regions_sigma", "regions_precision", "regions_log_norm")


@requires_rust
@pytest.mark.parametrize(
    "n_regions,symmetric", [(2, True), (4, True), (1, False), (3, False)]
)
def test_rust_matches_python_every_iteration(
    small_corpus, mni_mask, tmp_path, n_regions, symmetric
):
    """Rust and Python state must be identical after EVERY iteration.

    Comparing only endpoints would leave a divergence introduced at
    iteration 3 undiagnosable. This reports the first differing iteration
    and the first differing element.
    """
    counts, coords = small_corpus
    n_iters = 12
    mask_path = str(tmp_path / "mask.nii.gz")
    mni_mask.to_filename(mask_path)

    py_dir = tmp_path / "py_state"
    py_dir.mkdir()
    model = annotate.gclda.GCLDAModel(
        counts, coords, mask=mask_path, n_topics=4, n_regions=n_regions,
        symmetric=symmetric, seed_init=1,
    )
    model.fit(n_iters=n_iters, loglikely_freq=n_iters, dump_state_dir=str(py_dir))

    rs_dir = tmp_path / "rs_state"
    counts_path, coords_path = annotate.gclda_rs.export_gclda_tsvs(
        counts, coords, str(tmp_path / "inputs")
    )
    subprocess.run(
        [
            BINARY, "--counts", counts_path, "--coordinates", coords_path,
            "--mask", mask_path, "--out-dir", str(tmp_path / "rs_out"),
            "--n-topics", "4", "--n-regions", str(n_regions),
            "--symmetric", "true" if symmetric else "false",
            "--seed-init", "1", "--n-iters", str(n_iters),
            "--loglikely-freq", str(n_iters),
            "--dump-state-dir", str(rs_dir),
        ],
        check=True,
    )

    for it in range(1, n_iters + 1):
        py = np.load(py_dir / f"iter_{it:05d}.npz")
        for name in INTEGER_ARRAYS:
            rs = np.load(rs_dir / f"iter_{it:05d}" / f"{name}.npy")
            expected = py[name]
            if not np.array_equal(rs.ravel(), expected.ravel()):
                bad = np.flatnonzero(rs.ravel() != expected.ravel())[0]
                pytest.fail(
                    f"{name} diverged at iteration {it}, first at flat index {bad}: "
                    f"rust={rs.ravel()[bad]} python={expected.ravel()[bad]}"
                )
        for name in FLOAT_ARRAYS:
            rs = np.load(rs_dir / f"iter_{it:05d}" / f"{name}.npy")
            expected = py[name]
            # Shapes may differ harmlessly: Python stores regions_mu as
            # (T, R, 1, 3) while Rust writes (T, R, 3). Compare raveled values.
            # ascontiguousarray is required before .view() -- viewing a
            # non-contiguous array raises.
            rb = np.ascontiguousarray(rs.ravel(), dtype=np.float64).view(np.uint64)
            eb = np.ascontiguousarray(expected.ravel(), dtype=np.float64).view(np.uint64)
            assert rb.size == eb.size, f"{name} size mismatch at iteration {it}"
            if not np.array_equal(rb, eb):
                bad = np.flatnonzero(rb != eb)[0]
                pytest.fail(
                    f"{name} diverged (bitwise) at iteration {it}, flat index {bad}: "
                    f"rust={rs.ravel()[bad]!r} python={expected.ravel()[bad]!r}"
                )
```

- [ ] **Step 2: Run to verify it fails.** Run: `micromamba run -n nimenv pytest nimare/tests/test_gclda_rust.py -k every_iteration -v`. Expected: FAIL — `fit()` has no `dump_state_dir` parameter.
- [ ] **Step 3: Add `dump_state_dir` to Python.** Add an optional `dump_state_dir=None` keyword to `GCLDAModel.fit`, and a `_dump_state` method writing `np.savez` with the twelve arrays above. When `dump_state_dir` is None, do nothing — this must not affect normal operation.
- [ ] **Step 4: Add `--dump-state-dir` to Rust,** writing the same twelve arrays as `.npy` files per iteration.
- [ ] **Step 5: Run to verify it passes.** Run: `micromamba run -n nimenv pytest nimare/tests/test_gclda_rust.py -k every_iteration -v`. Expected: PASS for all four configurations.

> **If this fails, do not loosen the assertion.** Use the reported iteration and index to find the divergence. Most likely causes, in order: (a) `probs_pdf` region/topic ordering flipped in Task 10; (b) an arithmetic reassociation somewhere; (c) rayon nondeterminism in Task 11 — test with `--threads 1` to isolate.

- [ ] **Step 6: Commit**

```bash
git add nimare/annotate/gclda.py rust/gclda/src/output.rs \
        rust/gclda/src/bin/gclda-train.rs nimare/tests/test_gclda_rust.py
git diff --cached --stat
git commit -m "[TST] Add per-iteration state equality harness for Rust GCLDA port

Both implementations dump full sampler state after every iteration, and
the harness asserts bit-identity at each step across symmetric and
asymmetric configurations, reporting the first differing iteration and
element.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 17: Level 3 — end-to-end outputs and edge cases

**Files:**
- Modify: `nimare/tests/test_gclda_rust.py`

- [ ] **Step 1: Write the failing tests**

```python
@requires_rust
@pytest.mark.parametrize("n_regions,symmetric", [(2, True), (4, True), (1, False), (3, False)])
@pytest.mark.parametrize("seed_init", [1, 99])
def test_rust_probability_matrices_match_python(
    small_corpus, mni_mask, tmp_path, n_regions, symmetric, seed_init
):
    """All four probability matrices must be bit-identical after a full fit."""
    counts, coords = small_corpus
    mask_path = str(tmp_path / "mask.nii.gz")
    mni_mask.to_filename(mask_path)

    model = annotate.gclda.GCLDAModel(
        counts, coords, mask=mask_path, n_topics=4, n_regions=n_regions,
        symmetric=symmetric, seed_init=seed_init,
    )
    model.fit(n_iters=8, loglikely_freq=8)

    result = annotate.gclda_rs.train_gclda_rust(
        counts, coords, mask=mask_path, out_dir=str(tmp_path / "out"),
        binary=BINARY, n_topics=4, n_regions=n_regions, symmetric=symmetric,
        seed_init=seed_init, n_iters=8, loglikely_freq=8,
    )

    for name in (
        "p_topic_g_voxel_", "p_voxel_g_topic_", "p_topic_g_word_", "p_word_g_topic_"
    ):
        py = getattr(model, name)
        rs = np.asarray(getattr(result, name))
        assert rs.shape == py.shape, name
        assert np.array_equal(rs.view(np.uint64), py.view(np.uint64)), f"{name} not bit-identical"

    assert result.vocabulary == list(model.vocabulary)
    assert result.ids == list(model.ids)


@requires_rust
def test_rust_handles_topics_with_no_observations(mni_mask, tmp_path):
    """More topics than peaks forces empty subregions, exercising the
    n_obs == 0 and n_obs <= 1 branches of the region update."""
    ids = [f"s{i}" for i in range(4)]
    counts = pd.DataFrame(
        [[1, 2], [0, 3], [4, 0], [1, 1]], index=ids, columns=["a", "b"]
    )
    coords = pd.DataFrame(
        {"id": ids, "x": [1.0, -1.0, 2.0, -2.0],
         "y": [0.0, 0.0, 0.0, 0.0], "z": [0.0, 0.0, 0.0, 0.0]}
    )
    mask_path = str(tmp_path / "mask.nii.gz")
    mni_mask.to_filename(mask_path)

    model = annotate.gclda.GCLDAModel(
        counts, coords, mask=mask_path, n_topics=20, n_regions=2,
        symmetric=True, seed_init=1,
    )
    model.fit(n_iters=4, loglikely_freq=4)

    result = annotate.gclda_rs.train_gclda_rust(
        counts, coords, mask=mask_path, out_dir=str(tmp_path / "out"),
        binary=BINARY, n_topics=20, n_regions=2, symmetric=True,
        seed_init=1, n_iters=4, loglikely_freq=4,
    )
    assert np.array_equal(
        np.asarray(result.p_voxel_g_topic_).view(np.uint64),
        model.p_voxel_g_topic_.view(np.uint64),
    )


@requires_rust
def test_rust_handles_document_with_no_coordinates(mni_mask, tmp_path):
    """A document present in counts but absent from coordinates must be
    dropped identically by both implementations."""
    counts = pd.DataFrame(
        [[2, 1], [0, 3], [1, 1]], index=["a", "b", "no_coords"], columns=["w1", "w2"]
    )
    coords = pd.DataFrame(
        {"id": ["a", "a", "b"], "x": [5.0, -5.0, 10.0],
         "y": [1.0, 2.0, 3.0], "z": [4.0, 5.0, 6.0]}
    )
    mask_path = str(tmp_path / "mask.nii.gz")
    mni_mask.to_filename(mask_path)

    model = annotate.gclda.GCLDAModel(
        counts, coords, mask=mask_path, n_topics=3, n_regions=2,
        symmetric=True, seed_init=1,
    )
    model.fit(n_iters=3, loglikely_freq=3)

    result = annotate.gclda_rs.train_gclda_rust(
        counts, coords, mask=mask_path, out_dir=str(tmp_path / "out"),
        binary=BINARY, n_topics=3, n_regions=2, symmetric=True,
        seed_init=1, n_iters=3, loglikely_freq=3,
    )
    assert result.ids == list(model.ids) == ["a", "b"]
    assert np.array_equal(
        np.asarray(result.p_topic_g_word_).view(np.uint64),
        model.p_topic_g_word_.view(np.uint64),
    )
```

- [ ] **Step 2: Run to verify they fail or pass.** Run: `micromamba run -n nimenv pytest nimare/tests/test_gclda_rust.py -v`. If Tasks 8-13 are correct, several may pass immediately — that is fine. Fix any that fail.
- [ ] **Step 3: Commit**

```bash
git add nimare/tests/test_gclda_rust.py
git diff --cached --stat
git commit -m "[TST] Add end-to-end output regression tests for Rust GCLDA port

Asserts all four probability matrices are bit-identical to Python across
symmetric and asymmetric configurations, multiple seeds, and edge cases
covering empty subregions and documents lacking coordinates.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 18: Level 4 — downstream decoder integration

**Files:**
- Modify: `nimare/tests/test_gclda_rust.py`

- [ ] **Step 1: Write the failing test**

```python
@requires_rust
def test_rust_model_drives_existing_decoders_identically(small_corpus, mni_mask, tmp_path):
    """The three shipped GCLDA consumers must produce identical results
    whether driven by the Python model or a Rust-trained one."""
    import nibabel as nib

    from nimare import decode

    counts, coords = small_corpus
    mask_path = str(tmp_path / "mask.nii.gz")
    mni_mask.to_filename(mask_path)
    kwargs = dict(n_topics=4, n_regions=2, symmetric=True, seed_init=1)

    py_model = annotate.gclda.GCLDAModel(counts, coords, mask=mask_path, **kwargs)
    py_model.fit(n_iters=6, loglikely_freq=6)

    rs_model = annotate.gclda_rs.train_gclda_rust(
        counts, coords, mask=mask_path, out_dir=str(tmp_path / "out"),
        binary=BINARY, n_iters=6, loglikely_freq=6, **kwargs
    )

    arr = np.zeros(mni_mask.shape, np.int32)
    arr[40:44, 45:49, 40:44] = 1
    roi = nib.Nifti1Image(arr, mni_mask.affine)

    py_roi, _ = decode.discrete.gclda_decode_roi(py_model, roi)
    rs_roi, _ = decode.discrete.gclda_decode_roi(rs_model, roi)
    pd.testing.assert_frame_equal(py_roi, rs_roi)

    py_map, _ = decode.continuous.gclda_decode_map(py_model, roi)
    rs_map, _ = decode.continuous.gclda_decode_map(rs_model, roi)
    pd.testing.assert_frame_equal(py_map, rs_map)

    py_img, _ = decode.encode.gclda_encode(py_model, "term_1 term_2")
    rs_img, _ = decode.encode.gclda_encode(rs_model, "term_1 term_2")
    assert np.array_equal(py_img.get_fdata(), rs_img.get_fdata())
```

- [ ] **Step 2: Run to verify it fails.** Run: `micromamba run -n nimenv pytest nimare/tests/test_gclda_rust.py -k decoders -v`.
- [ ] **Step 3: Fix whatever it reveals.** If `GCLDAResult` is missing an attribute, add it. **Do not modify `nimare/decode/`** — if a change there seems necessary, the loader interface is wrong instead.
- [ ] **Step 4: Run the full suite.** Run: `micromamba run -n nimenv pytest nimare/tests/test_gclda_rust.py nimare/tests/test_annotate_gclda.py -v`. Expected: all PASS.
- [ ] **Step 5: Commit**

```bash
git add nimare/tests/test_gclda_rust.py nimare/annotate/gclda_rs.py
git diff --cached --stat
git commit -m "[TST] Verify Rust-trained GCLDA models drive existing decoders

Confirms gclda_decode_roi, gclda_decode_map, and gclda_encode produce
identical results from a Rust-trained model and a Python-trained one,
with no changes to nimare.decode.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

# Phase 4 — Profiling

### Task 19: Phase-level timing instrumentation

**Files:**
- Modify: `nimare/annotate/gclda.py`
- Modify: `rust/gclda/src/output.rs`, `rust/gclda/src/bin/gclda-train.rs`

**Interfaces:**
- Produces: `GCLDAModel.phase_times_` — a dict with keys `word_sampling`, `peak_sampling`, `region_update`, `loglikelihood`, `total`, values in seconds; the identical keys appear under `phase_times` in the Rust `model.json`

Attribution is the point: a single wall-clock ratio cannot tell you whether the port helped for the reasons the design predicted.

> **Deliberate substitution.** The spec listed criterion microbenchmarks on the Rust inner
> kernels. This instrumentation replaces them: criterion would measure Rust kernels against each
> other, whereas matched phase keys measure Rust against *Python*, which is the comparison the
> project is actually making. Adding criterion later is easy if a specific kernel needs isolated
> tuning; it is not needed to answer the question this port poses.

- [ ] **Step 1: Write the failing test**

```python
@requires_rust
def test_both_implementations_report_matching_phase_keys(small_corpus, mni_mask, tmp_path):
    """Phase timing keys must match so benchmarks can compare like with like."""
    counts, coords = small_corpus
    mask_path = str(tmp_path / "mask.nii.gz")
    mni_mask.to_filename(mask_path)

    model = annotate.gclda.GCLDAModel(
        counts, coords, mask=mask_path, n_topics=4, n_regions=2, symmetric=True
    )
    model.fit(n_iters=3, loglikely_freq=1)

    annotate.gclda_rs.train_gclda_rust(
        counts, coords, mask=mask_path, out_dir=str(tmp_path / "out"),
        binary=BINARY, n_topics=4, n_regions=2, symmetric=True,
        n_iters=3, loglikely_freq=1,
    )
    with open(tmp_path / "out" / "model.json") as fo:
        rust_meta = json.load(fo)

    expected = {"word_sampling", "peak_sampling", "region_update", "loglikelihood", "total"}
    assert set(model.phase_times_) == expected
    assert set(rust_meta["phase_times"]) == expected
    assert all(v >= 0 for v in model.phase_times_.values())
    assert rust_meta["phase_times"]["total"] > 0
```

- [ ] **Step 2: Run to verify it fails.** Run: `micromamba run -n nimenv pytest nimare/tests/test_gclda_rust.py -k phase_keys -v`.
- [ ] **Step 3: Implement.** In Python, accumulate `time.perf_counter()` deltas around each phase in `_update`, storing into `self.phase_times_` (initialize the dict in `__init__` with all keys at 0.0). In Rust, accumulate `std::time::Instant` durations and emit them in `model.json`.
- [ ] **Step 4: Run to verify it passes.** Run: `micromamba run -n nimenv pytest nimare/tests/test_gclda_rust.py -k phase_keys -v`.
- [ ] **Step 5: Confirm nothing regressed.** Run: `micromamba run -n nimenv pytest nimare/tests/test_gclda_rust.py nimare/tests/test_annotate_gclda.py -v`. Expected: all PASS — timing must not perturb results.
- [ ] **Step 6: Commit**

```bash
git add nimare/annotate/gclda.py rust/gclda/src/output.rs \
        rust/gclda/src/bin/gclda-train.rs nimare/tests/test_gclda_rust.py
git diff --cached --stat
git commit -m "[ENH] Add matching phase-level timing to both GCLDA implementations

Records word sampling, peak sampling, region update, and log-likelihood
time separately in both implementations, so benchmark differences can be
attributed to specific phases rather than reported as a single ratio.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 20: Synthetic generator and benchmark driver

**Files:**
- Create: `benchmarks/gclda_synthetic.py`, `benchmarks/bench_gclda_rust.py`

**Interfaces:**
- Produces:
  - `make_synthetic_corpus(n_docs, n_terms, n_peaks, seed=0) -> (count_df, coordinates_df)`
  - `bench_gclda_rust.py` CLI: `--scale {tiny,small,neurosynth}`, `--n-iters`, `--n-topics`, `--threads`, `--out`

The driver must measure, for each configuration: Python wall clock and per-phase times; Rust wall clock and per-phase times; peak RSS for both (via `/usr/bin/time -v` for Rust, `memory_profiler.memory_usage` for Python); and it must **verify outputs still match** before reporting any timing, so a benchmark can never report a speedup for a run that computed the wrong answer.

Neurosynth is fetched via `nimare.extract.fetch_neurosynth` and cached; the driver must skip that scale gracefully with a clear message if the download is unavailable.

- [ ] **Step 1: Write the failing test**

```python
def test_synthetic_corpus_is_deterministic_and_well_formed():
    """The generator must be reproducible and produce usable GCLDA input."""
    import sys
    sys.path.insert(0, os.path.join(REPO_ROOT, "benchmarks"))
    from gclda_synthetic import make_synthetic_corpus

    counts_a, coords_a = make_synthetic_corpus(n_docs=20, n_terms=15, n_peaks=100, seed=3)
    counts_b, coords_b = make_synthetic_corpus(n_docs=20, n_terms=15, n_peaks=100, seed=3)

    pd.testing.assert_frame_equal(counts_a, counts_b)
    pd.testing.assert_frame_equal(coords_a, coords_b)
    assert counts_a.shape == (20, 15)
    assert len(coords_a) == 100
    assert set(coords_a["id"]).issubset(set(counts_a.index))
    assert (counts_a.to_numpy() >= 0).all()
    assert (counts_a.to_numpy().sum(axis=1) > 0).all(), "every document needs tokens"
```

- [ ] **Step 2: Run to verify it fails.** Run: `micromamba run -n nimenv pytest nimare/tests/test_gclda_rust.py -k synthetic -v`.
- [ ] **Step 3: Implement both scripts.**
- [ ] **Step 4: Run to verify it passes.** Run: `micromamba run -n nimenv pytest nimare/tests/test_gclda_rust.py -k synthetic -v`.
- [ ] **Step 5: Smoke-test the driver.** Run: `micromamba run -n nimenv python benchmarks/bench_gclda_rust.py --scale tiny --n-iters 5 --out /tmp/gclda_bench_tiny.json`. Expected: completes, verifies output equality, prints a table.
- [ ] **Step 6: Commit**

```bash
git add benchmarks/gclda_synthetic.py benchmarks/bench_gclda_rust.py \
        nimare/tests/test_gclda_rust.py
git diff --cached --stat
git commit -m "[ENH] Add GCLDA benchmark driver and synthetic corpus generator

Measures wall clock, per-phase time, and peak RSS for both
implementations, and verifies output equality before reporting any
timing so a benchmark cannot credit a run that computed the wrong
answer.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 21: Run the benchmarks and write up results

**Files:**
- Create: `benchmarks/gclda_rust_results.md`

- [ ] **Step 1: Build the optimized binary**

```bash
cd rust/gclda && cargo build --release && cd ../..
```

- [ ] **Step 2: Run the scaling sweep**

```bash
micromamba run -n nimenv python benchmarks/bench_gclda_rust.py \
    --scale small --n-iters 50 --n-topics 25 --out /tmp/gclda_bench_small_t25.json
micromamba run -n nimenv python benchmarks/bench_gclda_rust.py \
    --scale small --n-iters 50 --n-topics 50 --out /tmp/gclda_bench_small_t50.json
micromamba run -n nimenv python benchmarks/bench_gclda_rust.py \
    --scale small --n-iters 50 --n-topics 100 --out /tmp/gclda_bench_small_t100.json
```

- [ ] **Step 3: Run the Neurosynth benchmark**

```bash
micromamba run -n nimenv python benchmarks/bench_gclda_rust.py \
    --scale neurosynth --n-iters 20 --n-topics 100 --out /tmp/gclda_bench_ns.json
```

Use a reduced iteration count and extrapolate per-iteration cost. **Label extrapolated figures as extrapolated.** If the download is unavailable, record that and report synthetic results only.

- [ ] **Step 4: Measure thread scaling**

```bash
for t in 1 2 4 8; do
  micromamba run -n nimenv python benchmarks/bench_gclda_rust.py \
      --scale small --n-iters 50 --n-topics 100 --threads $t \
      --out /tmp/gclda_bench_threads_$t.json
done
```

- [ ] **Step 5: Collect cache-behavior evidence for the fusion claim**

```bash
perf stat -e cache-misses,cache-references,instructions,cycles \
  rust/gclda/target/release/gclda-train \
  --counts /tmp/gclda_bench_inputs/counts.tsv \
  --coordinates /tmp/gclda_bench_inputs/coordinates.tsv \
  --mask nimare/resources/templates/MNI152_2x2x2_brainmask.nii.gz \
  --out-dir /tmp/gclda_perf_out --n-topics 100 --n-iters 20 2>&1 | tail -20
```

If `perf` is unavailable under WSL2, record that and rely on the peak-RSS and per-phase timing evidence instead.

- [ ] **Step 6: Write `benchmarks/gclda_rust_results.md`**

Include: hardware and software versions; a per-phase time table (Python vs Rust, with ratios); a peak-RSS table; scaling curves over `n_topics`; the thread-scaling table; `perf` counters if available; and a **"Where the wins came from"** section comparing measured results against the four predictions in the spec.

**Report what you measured.** If the word-sampler ratio is 1.0x, write 1.0x. If a prediction was wrong, say so plainly and explain what the profile actually showed. An honest table showing two large wins and one wash is a better artifact than a uniformly favorable one that nobody can reproduce.

- [ ] **Step 7: Run the full test suite one last time**

```bash
micromamba run -n nimenv pytest nimare/tests/test_gclda_rust.py nimare/tests/test_annotate_gclda.py -v
cd rust/gclda && cargo test --release && cd ../..
```

Expected: all PASS. Paste the actual summary lines into the results document.

- [ ] **Step 8: Commit**

```bash
git add benchmarks/gclda_rust_results.md
git diff --cached --stat
git commit -m "[DOC] Add measured GCLDA Rust port benchmark results

Per-phase timing, peak memory, thread scaling, and n_topics scaling for
both implementations, with measured results compared against the
predictions made in the design spec.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Completion Checklist

- [x] `cargo test --release` passes in `rust/gclda`
- [x] `micromamba run -n nimenv pytest nimare/tests/test_annotate_gclda.py nimare/tests/test_gclda_rust.py -v` passes
- [x] Level 2 per-iteration equality passes for all four configurations
- [x] All three shipped decoders produce identical results from a Rust-trained model, with `nimare/decode/` unmodified
- [x] `benchmarks/gclda_rust_results.md` contains measured numbers, with extrapolated figures labeled
- [x] `git log --oneline` shows the work; `git status` shows no unintended files staged
- [x] **Nothing has been pushed**

All items verified 2026-08-17. Measured results are in `benchmarks/gclda_rust_results.md`.

Two deviations from the plan, both recorded in the results document:

- **Task 21 Step 5 (`perf` counters) could not be run** — `perf` is unavailable under this WSL2 kernel. The fusion analysis rests on per-phase timing and peak-RSS evidence instead, as the plan allowed for.
- **The benchmark exposed a real defect before it could report timing** (commit `40f9f70`): pandas' default C float parser is not correctly rounded and disagreed with Rust's by 1 ULP on ~2.5% of real Neurosynth coordinates, so the two implementations trained on inputs differing in the last bit. Scoped to the benchmark harness, not the port — real NiMARE usage passes DataFrames directly and never reparses.
