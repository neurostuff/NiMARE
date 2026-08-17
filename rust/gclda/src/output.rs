//! Fit loop, probability distributions, and output writer.
//!
//! Ports two methods of `GCLDAModel` in `nimare/annotate/gclda.py`:
//! `fit`/`_update` (the training loop) and `get_probability_distributions`
//! (the four conditional probability matrices used for decoding/encoding).
//! The Python source is the specification: this file preserves its
//! arithmetic operation order exactly, except where explicitly noted below.
//!
//! ## Summation-order caveat (read before touching normalization code)
//!
//! `get_probability_distributions` divides by four different `np.sum(...)`
//! reductions. NumPy's low-level `pairwise_sum` kernel is used only when the
//! *reduced* axis is contiguous (unit stride); for a C-contiguous 2-D array
//! that means `axis=-1` (the last axis) goes through pairwise summation
//! (naive for length < 8, an 8-way-unrolled block for 8..=128, recursive
//! divide-and-conquer above that -- see [`crate::pairwise_sum`]), while
//! `axis=0` (non-contiguous stride) goes through NumPy's generic
//! strided-reduction loop, which accumulates sequentially over the reduced
//! axis. Concretely:
//!
//! - `p_voxel_g_topic`'s denominator (`spatial_dists.sum(axis=0)`, over
//!   voxels) is non-contiguous, so NumPy's actual algorithm IS plain
//!   sequential ascending-voxel accumulation. The code below matches this
//!   exactly, following the same precedent already established (and
//!   verified bit-exact) for `region_col_sum`/`word_col_sum` in
//!   `loglik.rs`.
//! - `p_topic_g_voxel`'s denominator (`spatial_dists.sum(axis=1)`, over
//!   topics) IS contiguous float data, so it goes through
//!   [`crate::pairwise_sum::numpy_sum`], not a plain loop -- plain
//!   accumulation diverges from `np.sum` once `n_topics >= 8` (measured:
//!   `np.sum` vs. naive differs in the majority of rows once `n_topics >=
//!   100`), which is exactly the production scale this port targets and
//!   exactly the case none of this crate's small fixtures (`n_topics` in
//!   {3, 4, 5}) can distinguish from a naive sum. See
//!   `tests/pairwise_sum.rs` for the verification against `np.sum` at
//!   `n_topics`-scale reduction lengths, including 100 and 228483.
//! - `p_word_g_topic`'s and `p_topic_g_word`'s denominators
//!   (`n_word_tokens_word_by_topic.sum(axis=0/1)`) sum `int64` counts:
//!   integer addition is exact and order-independent (no rounding to
//!   reassociate), so plain accumulation is correct for both regardless of
//!   axis or `n_topics`, even though one of them (`axis=1`, the
//!   `p_topic_g_word` denominator) is a contiguous reduction. Do not apply
//!   [`crate::pairwise_sum::numpy_sum`] there -- it is unneeded there and
//!   is written for `f64`, not counts.

use crate::gaussian::pdf;
use crate::io::npy::{self, Dtype, NpyWriter};
use crate::loglik::LogLikelihood;
use crate::model::Model;
use crate::pairwise_sum::numpy_sum;
use crate::GcldaError;
use rayon::prelude::*;
use std::io::Write;
use std::path::Path;

/// Number of voxel rows computed (in parallel, via rayon) before being
/// merged into the running column-sum accumulator / streamed to disk. Bounds
/// peak memory to `CHUNK_VOXELS * n_topics` floats regardless of how large
/// the mask is -- the full V x T matrix is never resident, per the task
/// spec. The value is not load-bearing for correctness, only for the
/// memory/parallelism tradeoff.
const CHUNK_VOXELS: usize = 4096;

/// NumPy's `np.nan_to_num(x, 0)` default: the second positional argument is
/// `copy`, not `nan` -- so `nan=0.0` (the default) applies, and this is NOT
/// an explicit "replace with 0", it just happens to look like one. `posinf`/
/// `neginf` default to the max/min finite `float64` values.
#[inline]
fn nan_to_num(v: f64) -> f64 {
    if v.is_nan() {
        0.0
    } else if v == f64::INFINITY {
        f64::MAX
    } else if v == f64::NEG_INFINITY {
        f64::MIN
    } else {
        v
    }
}

/// Fill `out` (length `n_topics`) with `spatial_dists[v][t] = sum_r
/// pdf(mask_xyz[v], regions_mu[t][r], regions_precision[t][r],
/// regions_log_norm[t][r])`, mirroring `_jit_get_spatial_dists`'s per-point,
/// per-topic inner loop (a hand-written sequential accumulation over
/// regions inside the Numba kernel, not a `np.sum` call -- so this part has
/// no summation-order ambiguity at all).
fn spatial_dists_row(model: &Model, i_voxel: usize, out: &mut [f64]) {
    let n_topics = model.params.n_topics;
    let n_regions = model.params.n_regions;
    let point = &model.mask.xyz[i_voxel];

    for i_topic in 0..n_topics {
        let mut topic_prob = 0.0f64;
        for j_region in 0..n_regions {
            let idx = Model::at(i_topic, j_region, n_regions);
            topic_prob += pdf(
                point,
                &model.regions_mu[idx],
                &model.regions_precision[idx],
                model.regions_log_norm[idx],
            );
        }
        out[i_topic] = topic_prob;
    }
}

/// Pass 1: stream `p_topic_g_voxel.npy` (row-local normalization, needs no
/// cross-voxel information) while accumulating `colsum[t] = sum_v
/// spatial_dists[v][t]` in strict ascending-voxel order for Pass 2 to use.
///
/// Each chunk's rows are computed in parallel (independent per voxel, no
/// shared accumulator -- the same structural argument as Task 11's
/// rayon-parallel region parameter phase), but the colsum accumulation and
/// the on-disk row order both walk the chunk sequentially afterward, so
/// bit-exactness does not depend on rayon's scheduling.
fn write_p_topic_g_voxel_and_accumulate_colsum(
    model: &Model,
    dtype: Dtype,
    dir: &Path,
) -> Result<Vec<f64>, GcldaError> {
    let n_topics = model.params.n_topics;
    let n_voxels = model.mask.xyz.len();

    let mut writer =
        NpyWriter::create(&dir.join("p_topic_g_voxel.npy"), &[n_voxels, n_topics], dtype)?;
    let mut colsum = vec![0.0f64; n_topics];
    let mut out_row = vec![0.0f64; n_topics];

    let mut start = 0usize;
    while start < n_voxels {
        let end = (start + CHUNK_VOXELS).min(n_voxels);
        let chunk_len = end - start;

        let mut chunk_rows = vec![0.0f64; chunk_len * n_topics];
        chunk_rows.par_chunks_mut(n_topics).enumerate().for_each(|(i, row)| {
            spatial_dists_row(model, start + i, row);
        });

        for i in 0..chunk_len {
            let row = &chunk_rows[i * n_topics..(i + 1) * n_topics];

            // rowsum = sum_t spatial_dists[v][t] -- a contiguous
            // floating-point reduction, so it must go through NumPy's
            // actual pairwise algorithm (see the module doc comment), not
            // a plain loop.
            let rowsum = numpy_sum(row);
            for (t, &v) in row.iter().enumerate() {
                // colsum, in contrast, is a strided (axis=0) reduction --
                // plain ascending-voxel accumulation is what NumPy itself
                // does here, see the module doc comment.
                colsum[t] += v;
                out_row[t] = nan_to_num(v / rowsum);
            }
            writer.write_row(&out_row)?;
        }
        start = end;
    }

    writer.finish()?;
    Ok(colsum)
}

/// Pass 2: recompute every voxel's spatial_dists row (same parallel-chunked
/// approach as Pass 1) and stream `p_voxel_g_topic.npy`, now that `colsum`
/// (computed in Pass 1) is known.
fn write_p_voxel_g_topic(
    model: &Model,
    colsum: &[f64],
    dtype: Dtype,
    dir: &Path,
) -> Result<(), GcldaError> {
    let n_topics = model.params.n_topics;
    let n_voxels = model.mask.xyz.len();

    let mut writer =
        NpyWriter::create(&dir.join("p_voxel_g_topic.npy"), &[n_voxels, n_topics], dtype)?;
    let mut out_row = vec![0.0f64; n_topics];

    let mut start = 0usize;
    while start < n_voxels {
        let end = (start + CHUNK_VOXELS).min(n_voxels);
        let chunk_len = end - start;

        let mut chunk_rows = vec![0.0f64; chunk_len * n_topics];
        chunk_rows.par_chunks_mut(n_topics).enumerate().for_each(|(i, row)| {
            spatial_dists_row(model, start + i, row);
        });

        for i in 0..chunk_len {
            let row = &chunk_rows[i * n_topics..(i + 1) * n_topics];
            for (t, &v) in row.iter().enumerate() {
                out_row[t] = nan_to_num(v / colsum[t]);
            }
            writer.write_row(&out_row)?;
        }
        start = end;
    }

    writer.finish()
}

/// `p_word_g_topic[w][t] = n_word_tokens_word_by_topic[w][t] /
/// colsum[t]` and `p_topic_g_word[w][t] = n_word_tokens_word_by_topic[w][t]
/// / rowsum[w]`, both `nan_to_num`'d. `W x T` is small (vocabulary-sized),
/// so unlike the voxel matrices this is computed fully in memory -- the
/// streaming requirement in the task spec is specific to the `V x T`
/// matrices.
fn compute_word_topic_matrices(model: &Model) -> (Vec<f64>, Vec<f64>) {
    let n_topics = model.params.n_topics;
    let n_words = model.corpus.vocabulary.len();
    let counts = &model.n_word_tokens_word_by_topic;

    // colsum[t] = sum_w counts[w][t] (axis=0, non-contiguous -- sequential
    // ascending-word order matches NumPy's actual strided-reduction here).
    let mut colsum = vec![0.0f64; n_topics];
    for w in 0..n_words {
        for t in 0..n_topics {
            colsum[t] += counts[Model::at(w, t, n_topics)] as f64;
        }
    }

    let mut p_word_g_topic = vec![0.0f64; n_words * n_topics];
    for w in 0..n_words {
        for t in 0..n_topics {
            let v = counts[Model::at(w, t, n_topics)] as f64 / colsum[t];
            p_word_g_topic[Model::at(w, t, n_topics)] = nan_to_num(v);
        }
    }

    // rowsum[w] = sum_t counts[w][t] (axis=1, contiguous, but these are
    // integer counts, not floats: plain accumulation is exact and
    // order-independent regardless of length, so no pairwise summation is
    // needed here even though the axis matches the case that does need it
    // for spatial_dists above -- see the module doc comment).
    let mut p_topic_g_word = vec![0.0f64; n_words * n_topics];
    for w in 0..n_words {
        let mut rowsum = 0.0f64;
        for t in 0..n_topics {
            rowsum += counts[Model::at(w, t, n_topics)] as f64;
        }
        for t in 0..n_topics {
            let v = counts[Model::at(w, t, n_topics)] as f64 / rowsum;
            p_topic_g_word[Model::at(w, t, n_topics)] = nan_to_num(v);
        }
    }

    (p_word_g_topic, p_topic_g_word)
}

fn write_vocabulary(model: &Model, dir: &Path) -> Result<(), GcldaError> {
    let mut f = std::fs::File::create(dir.join("vocabulary.txt"))?;
    for term in &model.corpus.vocabulary {
        f.write_all(term.as_bytes())?;
        f.write_all(b"\n")?;
    }
    Ok(())
}

fn write_regions_mu_sigma(model: &Model, dir: &Path) -> Result<(), GcldaError> {
    let n_topics = model.params.n_topics;
    let n_regions = model.params.n_regions;

    // regions_mu/regions_sigma are already stored topic-major, region-minor
    // (see model.rs), which is exactly the T x R x ... order requested for
    // these files -- flattening preserves that order as-is.
    let mu_flat: Vec<f64> =
        model.regions_mu.iter().flat_map(|m| m.iter().copied()).collect();
    npy::write_f64(&dir.join("regions_mu.npy"), &[n_topics, n_regions, 3], &mu_flat)?;

    let sigma_flat: Vec<f64> = model
        .regions_sigma
        .iter()
        .flat_map(|s| s.iter().flat_map(|row| row.iter().copied()))
        .collect();
    npy::write_f64(
        &dir.join("regions_sigma.npy"),
        &[n_topics, n_regions, 3, 3],
        &sigma_flat,
    )?;
    Ok(())
}

fn write_loglikelihood_tsv(model: &Model, dir: &Path) -> Result<(), GcldaError> {
    let mut f = std::fs::File::create(dir.join("loglikelihood.tsv"))?;
    f.write_all(b"iter\tx\tw\ttotal\n")?;
    for &(iter, x, w, total) in &model.loglikelihood_history {
        f.write_all(format!("{iter}\t{x}\t{w}\t{total}\n").as_bytes())?;
    }
    Ok(())
}

fn write_assignment_vectors(model: &Model, dir: &Path) -> Result<(), GcldaError> {
    let wtoken: Vec<i64> = model.wtoken_topic_idx.iter().map(|&v| v as i64).collect();
    npy::write_i64(&dir.join("wtoken_topic_idx.npy"), &[wtoken.len()], &wtoken)?;

    let ptopic: Vec<i64> = model.peak_topic_idx.iter().map(|&v| v as i64).collect();
    npy::write_i64(&dir.join("peak_topic_idx.npy"), &[ptopic.len()], &ptopic)?;

    let pregion: Vec<i64> = model.peak_region_idx.iter().map(|&v| v as i64).collect();
    npy::write_i64(&dir.join("peak_region_idx.npy"), &[pregion.len()], &pregion)?;

    Ok(())
}

fn write_model_json(model: &Model, dir: &Path) -> Result<(), GcldaError> {
    let p = &model.params;
    let affine: Vec<Vec<f64>> = model.mask.affine.iter().map(|row| row.to_vec()).collect();
    let shape: Vec<usize> = model.mask.shape.to_vec();

    let json = serde_json::json!({
        "n_topics": p.n_topics,
        "n_regions": p.n_regions,
        "symmetric": p.symmetric,
        "alpha": p.alpha,
        "beta": p.beta,
        "gamma": p.gamma,
        "delta": p.delta,
        "dobs": p.dobs,
        "roi_size": p.roi_size,
        "seed_init": p.seed_init,
        "n_iters": model.n_iters,
        "loglikely_freq": model.loglikely_freq,
        "ids": model.corpus.ids,
        "mask_path": model.mask.path.to_string_lossy(),
        "mask_affine": affine,
        "mask_shape": shape,
        "n_voxels": model.mask.xyz.len(),
        // Populated by Task 19; zeros are a valid, documented placeholder
        // until then (see that task's brief).
        "phase_times": {
            "word_sampling": 0.0,
            "peak_sampling": 0.0,
            "region_update": 0.0,
            "loglikelihood": 0.0,
            "total": 0.0,
        },
    });

    let text = serde_json::to_string_pretty(&json)?;
    std::fs::write(dir.join("model.json"), text)?;
    Ok(())
}

/// Write every GCLDA output file into `dir`, creating it if necessary.
/// Port of `get_probability_distributions` plus the surrounding output
/// bundle (counts, spatial parameters, assignment vectors, and metadata)
/// needed for decoding, encoding, and (eventually) resuming training.
///
/// `dtype` selects `f32` vs `f64` storage for the two large `V x T`
/// matrices ONLY (`p_topic_g_voxel.npy`, `p_voxel_g_topic.npy`); every other
/// float output (`p_topic_g_word.npy`, `p_word_g_topic.npy`, `regions_mu`,
/// `regions_sigma`) is always `f64`, and `dtype` has no effect on any
/// arithmetic -- it is a serialization choice, applied only inside
/// `NpyWriter::write_row`'s per-element cast.
pub fn write_outputs(model: &Model, dir: &Path, dtype: Dtype) -> Result<(), GcldaError> {
    std::fs::create_dir_all(dir)?;

    let colsum = write_p_topic_g_voxel_and_accumulate_colsum(model, dtype, dir)?;
    write_p_voxel_g_topic(model, &colsum, dtype, dir)?;

    let (p_word_g_topic, p_topic_g_word) = compute_word_topic_matrices(model);
    let n_words = model.corpus.vocabulary.len();
    let n_topics = model.params.n_topics;
    npy::write_f64(&dir.join("p_word_g_topic.npy"), &[n_words, n_topics], &p_word_g_topic)?;
    npy::write_f64(&dir.join("p_topic_g_word.npy"), &[n_words, n_topics], &p_topic_g_word)?;

    write_vocabulary(model, dir)?;

    let n_docs = model.corpus.ids.len();
    let n_regions = model.params.n_regions;
    npy::write_i64(
        &dir.join("n_word_tokens_word_by_topic.npy"),
        &[n_words, n_topics],
        &model.n_word_tokens_word_by_topic,
    )?;
    npy::write_i64(
        &dir.join("n_peak_tokens_doc_by_topic.npy"),
        &[n_docs, n_topics],
        &model.n_peak_tokens_doc_by_topic,
    )?;
    npy::write_i64(
        &dir.join("n_peak_tokens_region_by_topic.npy"),
        &[n_regions, n_topics],
        &model.n_peak_tokens_region_by_topic,
    )?;

    write_regions_mu_sigma(model, dir)?;
    write_loglikelihood_tsv(model, dir)?;
    write_assignment_vectors(model, dir)?;
    write_model_json(model, dir)?;

    Ok(())
}

impl Model {
    /// Run a complete update cycle (sample z, sample y&r, update regions).
    /// Port of `_update`. Kept private: the public entry point is
    /// [`Model::fit`], matching Python where `_update` is a private helper
    /// only `fit` calls directly.
    ///
    /// `on_loglikelihood` is invoked exactly where Python's `_update` calls
    /// `LGR.info` -- immediately after a log-likelihood is computed and
    /// recorded, still inside this iteration, so a caller driving a long
    /// `fit` loop can stream progress out (e.g. to a terminal) as each
    /// iteration completes rather than only after `fit` returns. It is a
    /// `&mut dyn FnMut` rather than a second type parameter so this private
    /// per-iteration method doesn't need its own generic instantiation.
    fn update(
        &mut self,
        loglikely_freq: usize,
        on_loglikelihood: &mut dyn FnMut(usize, &LogLikelihood),
    ) -> Result<(), GcldaError> {
        self.iter += 1;

        self.seed += 1;
        self.update_word_topic_assignments(self.seed)?;

        self.seed += 1;
        self.update_peak_assignments(self.seed)?;

        self.update_regions()?;

        if self.iter % loglikely_freq == 0 {
            let ll = self.compute_log_likelihood();
            self.loglikelihood_history.push((self.iter, ll.x, ll.w, ll.total));
            on_loglikelihood(self.iter, &ll);
        }

        Ok(())
    }

    /// Run `n_iters` total iterations (resuming from `self.iter` if this is
    /// not the first call). Port of `fit`.
    ///
    /// If `self.iter == 0`, first runs the initial spatial parameter
    /// estimate (`update_regions`) and records the initial log-likelihood,
    /// exactly as Python's `fit` does before its loop -- this is why a
    /// fresh model, even with `n_iters == 0`, still gets one recorded
    /// log-likelihood entry and populated `regions_*` fields. This initial
    /// entry does NOT invoke `on_loglikelihood`: Python's `fit` computes it
    /// directly (`self._update_regions()` + `self.compute_log_likelihood()`)
    /// rather than through `_update`, and never logs it via `LGR.info`
    /// either, so the callback -- which mirrors that exact log line -- must
    /// stay silent here too.
    ///
    /// `on_loglikelihood(iter, &ll)` is called once per iteration where a
    /// log-likelihood is actually recorded (i.e. `iter % loglikely_freq ==
    /// 0`), from inside the loop below, so a caller can emit progress (to
    /// stderr, a UI, etc.) while training is still running rather than only
    /// after every iteration has finished. Callers that don't need progress
    /// notifications can pass `|_, _| {}`.
    ///
    /// The loop below runs exactly `n_iters.saturating_sub(self.iter)`
    /// times: Rust's `self.iter..n_iters` range, like Python's
    /// `range(self.iter, n_iters)`, captures the starting bound once at
    /// loop entry, so mutating `self.iter` inside the loop body (via
    /// `update`) does not change the iteration count -- both empty out
    /// identically when `n_iters <= self.iter`.
    pub fn fit<F>(
        &mut self,
        n_iters: usize,
        loglikely_freq: usize,
        mut on_loglikelihood: F,
    ) -> Result<(), GcldaError>
    where
        F: FnMut(usize, &LogLikelihood),
    {
        self.n_iters = n_iters;
        self.loglikely_freq = loglikely_freq;

        if self.iter == 0 {
            self.update_regions()?;
            let ll = self.compute_log_likelihood();
            self.loglikelihood_history.push((self.iter, ll.x, ll.w, ll.total));
        }

        for _ in self.iter..n_iters {
            self.update(loglikely_freq, &mut on_loglikelihood)?;
        }

        Ok(())
    }
}
