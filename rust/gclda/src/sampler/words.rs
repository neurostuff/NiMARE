//! Word-topic sampler: sequential collapsed Gibbs sweep over word tokens.
//!
//! Port of `_jit_update_word_topic_assignments` in
//! `nimare/annotate/gclda.py`. The Python source is the specification: this
//! file preserves its arithmetic operation order exactly, including keeping
//! the count-ratio a division rather than a reciprocal multiply.

use crate::model::Model;
use crate::rng::Mt19937;
use crate::GcldaError;

impl Model {
    /// Update word-token -> topic assignments (z) via one Gibbs sweep.
    ///
    /// Re-seeds the RNG from `seed` at entry (mirrors `np.random.seed(randseed)`
    /// at the top of `_jit_update_word_topic_assignments`), then visits word
    /// tokens in order. For each token: decrement its current topic's three
    /// counts (`word_by_topic`, `total_word_by_topic`, `word_doc_by_topic`),
    /// recompute per-topic probabilities from the now-decremented counts,
    /// sample a new topic, and re-increment the same three counts under the
    /// new topic.
    ///
    /// This has a true sequential dependency across tokens -- each token
    /// reads counts the previous token wrote -- and must never be
    /// parallelized.
    pub fn update_word_topic_assignments(&mut self, seed: u32) -> Result<(), GcldaError> {
        let mut rng = Mt19937::new(seed);

        let n_topics = self.params.n_topics;
        let beta = self.params.beta;
        let gamma = self.params.gamma;
        // Computed once, from the vocabulary size, not from the number of
        // distinct words that actually appear in the token stream.
        let beta_vocabulary = beta * self.corpus.vocabulary.len() as f64;

        let mut probs = vec![0.0f64; n_topics];
        let n_wtokens = self.corpus.wtoken_word_idx.len();

        for i in 0..n_wtokens {
            let word = self.corpus.wtoken_word_idx[i] as usize;
            let doc = self.corpus.wtoken_doc_idx[i] as usize;
            let topic = self.wtoken_topic_idx[i] as usize;

            self.n_word_tokens_word_by_topic[Model::at(word, topic, n_topics)] -= 1;
            self.total_n_word_tokens_by_topic[topic] -= 1;
            self.n_word_tokens_doc_by_topic[Model::at(doc, topic, n_topics)] -= 1;

            for t in 0..n_topics {
                probs[t] = ((self.n_word_tokens_word_by_topic[Model::at(word, t, n_topics)] as f64
                    + beta)
                    / (self.total_n_word_tokens_by_topic[t] as f64 + beta_vocabulary))
                    * (self.n_peak_tokens_doc_by_topic[Model::at(doc, t, n_topics)] as f64
                        + gamma);
            }

            let new_topic = rng.sample_from_unnormalized(&probs)?;
            self.wtoken_topic_idx[i] = new_topic as u32;
            self.n_word_tokens_word_by_topic[Model::at(word, new_topic, n_topics)] += 1;
            self.total_n_word_tokens_by_topic[new_topic] += 1;
            self.n_word_tokens_doc_by_topic[Model::at(doc, new_topic, n_topics)] += 1;
        }

        Ok(())
    }
}
