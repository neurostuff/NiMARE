//! Log-likelihood of the training data under the current model.
//!
//! Port of `compute_log_likelihood` in `nimare/annotate/gclda.py`.
//!
//! Python materializes an `n_documents x n_word_types` matrix
//! (`p_wtoken_g_doc = np.dot(docprobs_z, wordprobs.T)`) and reads only the
//! entries for observed `(doc, word)` pairs -- ~340 MB and ~4.2 GFLOP at
//! Neurosynth scale. [`Model::compute_log_likelihood`] instead computes each
//! observed token's dot product directly (`O(nnz * n_topics)` instead of
//! `O(n_documents * n_word_types * n_topics)`, no dense allocation), and
//! reuses [`Model::peak_probs_block`] (one parallel block of peaks at a
//! time, same as `update_peak_assignments`) rather than materializing the
//! full `n_peaks x n_topics x n_regions` peak-probability array. The four cached
//! per-call probability matrices (`docprobs_y`, `docprobs_z`, `regionprobs`,
//! `wordprobs`) are computed with the same arithmetic Python uses, and both
//! the peak (x) and word (w) terms preserve Python's per-element formula.
//! See the comment on [`LogLikelihood`] for why NEITHER term is asserted
//! bit-exact, despite that.

use crate::model::Model;

/// The three totals returned by `compute_log_likelihood`: peak-token,
/// word-token, and combined log-likelihood.
///
/// `x`, `w`, and `total` are the only quantities in this whole crate that
/// are NOT asserted bit-exact against Python (see `tests/loglik.rs`, which
/// compares all three with a relative tolerance of `1e-10`). There are TWO
/// independent, unrelated causes -- both real, neither fixable by "just
/// summing more carefully" in Rust:
///
/// 1. **BLAS-mediated dot products.** Python's word term computes
///    `p_wtoken_g_doc = np.dot(docprobs_z, wordprobs.T)`, and its peak term
///    computes `p_x_rd = np.dot(p_region_g_doc, p_x_r)` per region -- both
///    are 1-D dot products of length `n_topics` dispatched to BLAS
///    (`cblas_ddot`). BLAS's summation order (and whether it fuses
///    multiply-adds) is a property of the BLAS build that generated the
///    reference fixture, not of the Python source, and is not reproducible
///    from scalar Rust. This affects BOTH `x` and `w`, not just `w`.
/// 2. **`np.sum`'s pairwise summation.** `docprobs_y` and `docprobs_z` are
///    built by dividing by `np.sum(doccounts, axis=1)` -- a sum along the
///    contiguous (stride-1) axis of a `(n_docs, n_topics)` array. NumPy's
///    pairwise-summation kernel diverges from naive left-to-right
///    accumulation once the reduction length reaches 8 (see
///    `pairwise_sum.rs`), so a plain `row_sum += v` loop over `n_topics`
///    stops matching once `n_topics >= 8`. This reaches `x_loglikely` via
///    `docprobs_y` and `w_loglikely` via `docprobs_z`. (`regionprobs` and
///    `wordprobs` sum along `axis=0` of their arrays instead -- a STRIDED
///    axis, which NumPy reduces with a plain sequential loop, not the
///    pairwise kernel -- so a plain Rust loop matches those exactly; see
///    the module doc on `pairwise_sum.rs` for why axis matters here.)
///
/// This crate has a `numpy_sum` helper (`pairwise_sum.rs`) that reproduces
/// `np.sum`'s pairwise kernel bit-for-bit, used elsewhere (the output
/// writer's `p_topic_g_voxel` row-sum) to keep a quantity bit-exact. It is
/// deliberately NOT used here: cause 1 (BLAS dot products) still breaks
/// bit-exactness for `x` and `w` regardless of how faithfully the row sums
/// are computed, so there is no bit-exact target to reach for these two
/// quantities, and a tolerance-based comparison is the correct design, not
/// a workaround pending a fix.
///
/// The fixture's `n_topics=3` measures a relative error of exactly `0e0`
/// for `x`, `w`, and `total` -- but that is an artifact of `n_topics=3`
/// being below BOTH thresholds above (BLAS `ddot` at length 3 has no room
/// to reorder; `np.sum` at length 3 falls back to a plain sequential loop
/// itself, below its own length-8 pairwise threshold). It is NOT evidence
/// that these quantities are bit-exact in general, and must not be used to
/// justify tightening this assertion to `assert_eq!` -- at production scale
/// (e.g. Neurosynth's dozens of topics) both causes are live. Conversely,
/// do not use this comment as license to loosen any OTHER assertion in the
/// crate by analogy -- every other quantity in this port has no BLAS or
/// `np.sum`-pairwise dependency and remains held to genuine bit-exactness.
pub struct LogLikelihood {
    pub x: f64,
    pub w: f64,
    pub total: f64,
}

impl Model {
    /// Compute the log-likelihood of the training data under the current
    /// model. Port of `compute_log_likelihood(model=None, update_vectors=True)`
    /// called with `update_vectors=False`: this method never mutates
    /// `self` and never appends to a log-likelihood history (the Rust port
    /// has no such history vector; task 13's fit loop stores what it needs
    /// itself).
    pub fn compute_log_likelihood(&self) -> LogLikelihood {
        let n_topics = self.params.n_topics;
        let n_regions = self.params.n_regions;
        let n_docs = self.corpus.ids.len();
        let n_words = self.corpus.vocabulary.len();
        let alpha = self.params.alpha;
        let beta = self.params.beta;
        let gamma = self.params.gamma;
        let delta = self.params.delta;

        // docprobs_y[d][t] = (n_peak_tokens_doc_by_topic[d][t] + alpha)
        //                    / sum_t' (n_peak_tokens_doc_by_topic[d][t'] + alpha)
        let mut docprobs_y = vec![0.0f64; n_docs * n_topics];
        for d in 0..n_docs {
            let mut row_sum = 0.0f64;
            for t in 0..n_topics {
                let v = self.n_peak_tokens_doc_by_topic[Model::at(d, t, n_topics)] as f64 + alpha;
                docprobs_y[Model::at(d, t, n_topics)] = v;
                row_sum += v;
            }
            for t in 0..n_topics {
                docprobs_y[Model::at(d, t, n_topics)] /= row_sum;
            }
        }

        // docprobs_z[d][t] = (n_peak_tokens_doc_by_topic[d][t] + gamma)
        //                    / sum_t' (n_peak_tokens_doc_by_topic[d][t'] + gamma)
        let mut docprobs_z = vec![0.0f64; n_docs * n_topics];
        for d in 0..n_docs {
            let mut row_sum = 0.0f64;
            for t in 0..n_topics {
                let v = self.n_peak_tokens_doc_by_topic[Model::at(d, t, n_topics)] as f64 + gamma;
                docprobs_z[Model::at(d, t, n_topics)] = v;
                row_sum += v;
            }
            for t in 0..n_topics {
                docprobs_z[Model::at(d, t, n_topics)] /= row_sum;
            }
        }

        // regionprobs[r][t] = (n_peak_tokens_region_by_topic[r][t] + delta)
        //                     / sum_r' (n_peak_tokens_region_by_topic[r'][t] + delta)
        // Column sums (over regions, per topic) computed in region-ascending
        // order, matching `np.sum(regioncounts, axis=0)`.
        let mut region_col_sum = vec![0.0f64; n_topics];
        for r in 0..n_regions {
            for t in 0..n_topics {
                region_col_sum[t] +=
                    self.n_peak_tokens_region_by_topic[Model::at(r, t, n_topics)] as f64 + delta;
            }
        }
        let mut regionprobs = vec![0.0f64; n_regions * n_topics];
        for r in 0..n_regions {
            for t in 0..n_topics {
                let v =
                    self.n_peak_tokens_region_by_topic[Model::at(r, t, n_topics)] as f64 + delta;
                regionprobs[Model::at(r, t, n_topics)] = v / region_col_sum[t];
            }
        }

        // wordprobs[w][t] = (n_word_tokens_word_by_topic[w][t] + beta)
        //                   / sum_w' (n_word_tokens_word_by_topic[w'][t] + beta)
        // Column sums (over words, per topic) computed in word-ascending
        // order, matching `np.sum(wordcounts, axis=0)`.
        let mut word_col_sum = vec![0.0f64; n_topics];
        for w in 0..n_words {
            for t in 0..n_topics {
                word_col_sum[t] +=
                    self.n_word_tokens_word_by_topic[Model::at(w, t, n_topics)] as f64 + beta;
            }
        }
        let mut wordprobs = vec![0.0f64; n_words * n_topics];
        for w in 0..n_words {
            for t in 0..n_topics {
                let v = self.n_word_tokens_word_by_topic[Model::at(w, t, n_topics)] as f64 + beta;
                wordprobs[Model::at(w, t, n_topics)] = v / word_col_sum[t];
            }
        }

        // --- Peak (x) term ---
        // p(x|model, doc) = sum_r p(topic|doc) . p(subregion=r|topic) . p(x|subregion=r)
        //                 = sum_r sum_t docprobs_y[doc][t] * regionprobs[r][t] * peak_probs[t][r]
        //
        // PDF evaluation is done one parallel block of peaks at a time (see
        // `Model::peak_probs_block`), mirroring `update_peak_assignments`.
        // The accumulation into `x_loglikely` below stays sequential and in
        // peak order -- it is a floating-point reduction, and reassociating
        // it would change the result.
        let mut x_loglikely = 0.0f64;
        let n_ptokens = self.corpus.ptoken_doc_idx.len();
        let stride = n_topics * n_regions;
        let block_size = self.params.peak_block_size.max(1);
        let mut block_buf = vec![0.0f64; block_size.min(n_ptokens.max(1)) * stride];
        let mut block_start = 0usize;
        while block_start < n_ptokens {
            let len = block_size.min(n_ptokens - block_start);
            self.peak_probs_block(block_start, len, &mut block_buf);
            for i in 0..len {
                let i_ptoken = block_start + i;
                let peak_probs = &block_buf[i * stride..(i + 1) * stride];

                let doc = self.corpus.ptoken_doc_idx[i_ptoken] as usize;

                let mut p_x = 0.0f64;
                for j_region in 0..n_regions {
                    // p_region_g_doc[t] = docprobs_y[doc][t] * regionprobs[j_region][t]
                    // p_x_rd = sum_t p_region_g_doc[t] * peak_probs[t][j_region]
                    let mut p_x_rd = 0.0f64;
                    for t in 0..n_topics {
                        let p_topic_g_doc = docprobs_y[Model::at(doc, t, n_topics)];
                        let p_region_g_topic = regionprobs[Model::at(j_region, t, n_topics)];
                        let p_region_g_doc = p_topic_g_doc * p_region_g_topic;
                        let p_x_r = peak_probs[Model::at(t, j_region, n_regions)];
                        p_x_rd += p_region_g_doc * p_x_r;
                    }
                    p_x += p_x_rd;
                }
                x_loglikely += p_x.ln();
            }
            block_start += len;
        }

        // --- Word (w) term ---
        // p(w|model, doc) = sum_t docprobs_z[doc][t] * wordprobs[word][t]
        //
        // Python computes this via a dense np.dot (BLAS); here we compute
        // the same sum-of-products per observed token directly, in
        // topic-ascending order, without materializing the dense
        // n_documents x n_word_types matrix. See LogLikelihood's docs for
        // why this term is not asserted bit-exact.
        let mut w_loglikely = 0.0f64;
        let n_wtokens = self.corpus.wtoken_word_idx.len();
        for i_wtoken in 0..n_wtokens {
            let word_token = self.corpus.wtoken_word_idx[i_wtoken] as usize;
            let doc = self.corpus.wtoken_doc_idx[i_wtoken] as usize;

            let mut p_wtoken = 0.0f64;
            for t in 0..n_topics {
                p_wtoken += docprobs_z[Model::at(doc, t, n_topics)]
                    * wordprobs[Model::at(word_token, t, n_topics)];
            }
            w_loglikely += p_wtoken.ln();
        }

        let tot_loglikely = x_loglikely + w_loglikely;

        LogLikelihood { x: x_loglikely, w: w_loglikely, total: tot_loglikely }
    }
}
