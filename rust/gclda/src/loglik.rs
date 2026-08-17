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
//! reuses [`Model::peak_probs_for`] rather than materializing the full
//! `n_peaks x n_topics x n_regions` peak-probability array. The four cached
//! per-call probability matrices (`docprobs_y`, `docprobs_z`, `regionprobs`,
//! `wordprobs`) are still computed exactly as Python computes them, and the
//! peak (x) term's arithmetic order is unchanged from Python -- only the
//! word (w) term's summation order is BLAS-free scalar code instead of
//! `np.dot`. See the comment on [`LogLikelihood`] for why that one quantity
//! is not asserted bit-exact.

use crate::model::Model;

/// The three totals returned by `compute_log_likelihood`: peak-token,
/// word-token, and combined log-likelihood.
///
/// `w` (and therefore `total = x + w`) is the one quantity in this whole
/// crate that is NOT asserted bit-exact against Python. Python computes it
/// via `p_wtoken_g_doc = np.dot(docprobs_z, wordprobs.T)`, which is routed
/// through BLAS; BLAS's summation order (and whether it fuses multiply-adds)
/// is a property of the BLAS build that generated the reference fixture, not
/// of the Python source, so scalar Rust code cannot reproduce it bit-for-bit
/// even while preserving the mathematically-equivalent left-to-right
/// summation order used everywhere else in this port. The test compares `w`
/// (and `x`/`total`) with a relative tolerance of `1e-10` instead. Do not
/// use this as license to loosen any other assertion in the crate -- `x`
/// and all four cached probability matrices below are ordinary scalar
/// arithmetic and remain held to bit-exactness.
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
        let mut x_loglikely = 0.0f64;
        let mut peak_probs = vec![0.0f64; n_topics * n_regions];
        for i_ptoken in 0..self.corpus.ptoken_doc_idx.len() {
            let doc = self.corpus.ptoken_doc_idx[i_ptoken] as usize;
            self.peak_probs_for(i_ptoken, &mut peak_probs);

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
