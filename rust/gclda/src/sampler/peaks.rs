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
        // Reused across peaks: peak_probs_for fills this with p(x|topic,region)
        // for the current peak, indexed topic * n_regions + region.
        let mut peak_probs = vec![0.0f64; n_topics * n_regions];

        let n_ptokens = self.corpus.ptoken_doc_idx.len();
        for i_ptoken in 0..n_ptokens {
            let doc = self.corpus.ptoken_doc_idx[i_ptoken] as usize;
            let topic = self.peak_topic_idx[i_ptoken] as usize;
            let region = self.peak_region_idx[i_ptoken] as usize;

            self.n_peak_tokens_region_by_topic[Model::at(region, topic, n_topics)] -= 1;
            self.n_peak_tokens_doc_by_topic[Model::at(doc, topic, n_topics)] -= 1;
            region_totals[topic] -= 1.0;

            self.peak_probs_for(i_ptoken, &mut peak_probs);

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

        Ok(())
    }
}
