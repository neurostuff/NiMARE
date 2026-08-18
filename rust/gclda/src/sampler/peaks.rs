//! Peak topic/subregion sampler with fused PDF evaluation.
//!
//! Port of `_jit_update_peak_assignments` in `nimare/annotate/gclda.py`.
//! The Python source is the specification: this file preserves its
//! arithmetic operation order exactly.
//!
//! Python's `_get_peak_probs` (`_jit_get_peak_probs` in
//! `nimare/annotate/gclda.py`) materializes an
//! `n_peaks x n_topics x n_regions` array of Gaussian densities every
//! iteration -- roughly 800 MB at full Neurosynth scale -- and then streams
//! through it exactly once, reading only the current peak's `T x R` block.
//! [`Model::peak_probs_for`] computes that one block on demand into a
//! caller-supplied buffer, so [`Model::update_peak_assignments`] can allocate
//! the buffer once outside the peak loop instead of materializing the full
//! array. The arithmetic per element is unchanged, so results are
//! bit-identical to the Python; only the allocation and memory traffic are
//! eliminated.

use crate::gaussian::pdf;
use crate::model::Model;
use crate::rng::Mt19937;
use crate::GcldaError;
use rayon::prelude::*;

/// Minimum number of Gaussian evaluations (`len * n_topics * n_regions`) in a
/// block before it is worth filling with rayon. Below this, task overhead
/// exceeds the work: `update_regions` was measured 4x *slower* at 8 threads
/// than at 1 on small corpora for exactly this reason (see
/// `benchmarks/gclda_rust_results.md`, thread-scaling table).
pub const PARALLEL_MIN_EVALS: usize = 32_768;

impl Model {
    /// Fill `out` (length `n_topics * n_regions`, indexed
    /// `topic * n_regions + region`) with `p(x_i | topic, region)` for peak
    /// `i_peak`, evaluated under each topic/subregion's cached Gaussian.
    ///
    /// Mirrors the per-point body of `_jit_get_peak_probs`, but for a single
    /// peak rather than the full `n_peaks x n_topics x n_regions` array.
    /// Used by both [`Model::update_peak_assignments`] and, later, the
    /// log-likelihood computation.
    pub fn peak_probs_for(&self, i_peak: usize, out: &mut [f64]) {
        let n_topics = self.params.n_topics;
        let n_regions = self.params.n_regions;
        let point = &self.corpus.ptoken_coords[i_peak];

        for i_topic in 0..n_topics {
            for j_region in 0..n_regions {
                let idx = Model::at(i_topic, j_region, n_regions);
                out[idx] = pdf(
                    point,
                    &self.regions_mu[idx],
                    &self.regions_precision[idx],
                    self.regions_log_norm[idx],
                );
            }
        }
    }

    /// Time one serial pass of `peak_probs_for` over every peak, into a single
    /// reusable buffer. Diagnostic only: used by `--profile-pdf` to measure what
    /// share of the peak-sampling phase is PDF evaluation, which is the number
    /// the block-wise parallelization decision is gated on. Not called during
    /// normal training.
    ///
    /// Returns `(seconds, n_evaluated)`: elapsed wall time for the whole pass,
    /// and the number of peaks actually evaluated, so a caller (test or CLI)
    /// can confirm the loop was not partially or fully elided by the
    /// optimizer. Every iteration's fill of `buf` is summed into an
    /// accumulator that is itself `black_box`-ed *inside* the loop -- unlike a
    /// single trailing `black_box(&buf)`, which only observes the final
    /// iteration's state and would not stop the compiler from proving the
    /// earlier `n - 1` calls are dead stores -- so every entry written by
    /// every call to `peak_probs_for` feeds an opaque value and no
    /// iteration's work can be eliminated.
    pub fn time_serial_pdf_pass(&self) -> (f64, usize) {
        let n_topics = self.params.n_topics;
        let n_regions = self.params.n_regions;
        let mut buf = vec![0.0f64; n_topics * n_regions];
        let n_peaks = self.corpus.ptoken_coords.len();
        let mut acc = 0.0f64;
        let start = std::time::Instant::now();
        for i_peak in 0..n_peaks {
            self.peak_probs_for(i_peak, &mut buf);
            let sum: f64 = buf.iter().sum();
            acc += std::hint::black_box(sum);
        }
        let elapsed = start.elapsed().as_secs_f64();
        // Consume `acc` so the accumulation itself cannot be optimized away.
        std::hint::black_box(acc);
        (elapsed, n_peaks)
    }

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

    /// Update peak-token -> topic/subregion assignments (y, c) via one Gibbs
    /// sweep.
    ///
    /// Re-seeds the RNG from `seed` at entry (mirrors `np.random.seed(randseed)`
    /// at the top of `_jit_update_peak_assignments`), precomputes
    /// `region_totals[t] = sum_r region_by_topic[r, t]` once, then visits peak
    /// tokens in order. For each token: decrement its current topic/region's
    /// counts (`region_by_topic`, `doc_by_topic`, `region_totals`), compute
    /// `peak_topic_probs` via the log1p/max-subtract/exp stabilization,
    /// build the sampling weight vector `probs_pdf` in region-major,
    /// topic-minor flat order, sample, decode
    /// `region = idx / n_topics`, `topic = idx % n_topics`, and re-increment.
    ///
    /// This has a true sequential dependency across tokens -- each token
    /// reads counts the previous token wrote -- and must never be
    /// parallelized.
    pub fn update_peak_assignments(&mut self, seed: u32) -> Result<(), GcldaError> {
        let mut rng = Mt19937::new(seed);

        let n_topics = self.params.n_topics;
        let n_regions = self.params.n_regions;
        let delta = self.params.delta;
        let alpha = self.params.alpha;
        let gamma = self.params.gamma;
        let region_total_prior = delta * n_regions as f64;

        // region_totals[t] = sum over r of region_by_topic[r][t], maintained
        // incrementally alongside region_by_topic itself.
        let mut region_totals = vec![0.0f64; n_topics];
        for i_topic in 0..n_topics {
            let mut topic_total = 0.0f64;
            for j_region in 0..n_regions {
                topic_total +=
                    self.n_peak_tokens_region_by_topic[Model::at(j_region, i_topic, n_topics)]
                        as f64;
            }
            region_totals[i_topic] = topic_total;
        }

        let mut peak_topic_probs = vec![0.0f64; n_topics];
        let mut probs_pdf = vec![0.0f64; n_regions * n_topics];

        let n_ptokens = self.corpus.ptoken_doc_idx.len();

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

                let doc = self.corpus.ptoken_doc_idx[i_ptoken] as usize;
                let topic = self.peak_topic_idx[i_ptoken] as usize;
                let region = self.peak_region_idx[i_ptoken] as usize;

                self.n_peak_tokens_region_by_topic[Model::at(region, topic, n_topics)] -= 1;
                self.n_peak_tokens_doc_by_topic[Model::at(doc, topic, n_topics)] -= 1;
                region_totals[topic] -= 1.0;

                let mut max_logp = f64::NEG_INFINITY;
                for i_topic in 0..n_topics {
                    let doc_topic_peak_counts =
                        self.n_peak_tokens_doc_by_topic[Model::at(doc, i_topic, n_topics)] as f64
                            + gamma;
                    let logp = self.n_word_tokens_doc_by_topic[Model::at(doc, i_topic, n_topics)]
                        as f64
                        * (1.0 / doc_topic_peak_counts).ln_1p();
                    peak_topic_probs[i_topic] = logp;
                    if logp > max_logp {
                        max_logp = logp;
                    }
                }

                for i_topic in 0..n_topics {
                    peak_topic_probs[i_topic] = (peak_topic_probs[i_topic] - max_logp).exp();
                }

                let mut flat_idx = 0usize;
                for j_region in 0..n_regions {
                    for i_topic in 0..n_topics {
                        probs_pdf[flat_idx] = peak_probs[Model::at(i_topic, j_region, n_regions)]
                            * ((self.n_peak_tokens_region_by_topic
                                [Model::at(j_region, i_topic, n_topics)]
                                as f64
                                + delta)
                                / (region_totals[i_topic] + region_total_prior))
                            * (self.n_peak_tokens_doc_by_topic[Model::at(doc, i_topic, n_topics)]
                                as f64
                                + alpha)
                            * peak_topic_probs[i_topic];
                        flat_idx += 1;
                    }
                }

                let sampled_idx = rng.sample_from_unnormalized(&probs_pdf)?;
                let region = sampled_idx / n_topics;
                let topic = sampled_idx % n_topics;

                self.n_peak_tokens_region_by_topic[Model::at(region, topic, n_topics)] += 1;
                self.n_peak_tokens_doc_by_topic[Model::at(doc, topic, n_topics)] += 1;
                region_totals[topic] += 1.0;
                self.peak_topic_idx[i_ptoken] = topic as u32;
                self.peak_region_idx[i_ptoken] = region as u32;
            }
            self.phase_times.peak_sample += t_sample.elapsed().as_secs_f64();

            block_start += len;
        }

        Ok(())
    }
}
