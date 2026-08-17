//! GCLDA model state and initialization.
//!
//! [`Model::new`] reproduces `GCLDAModel.__init__`'s random assignment and
//! count initialization exactly (`nimare/annotate/gclda.py`). The RNG
//! consumption order is load-bearing -- see the comments inline.

use crate::io::nifti::MaskInfo;
use crate::io::tsv::Corpus;
use crate::rng::Mt19937;
use crate::GcldaError;

/// Model hyperparameters, mirroring `GCLDAModel.params` in the Python class.
pub struct Params {
    pub n_topics: usize,
    pub n_regions: usize,
    pub symmetric: bool,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub delta: f64,
    pub dobs: f64,
    pub roi_size: f64,
    pub seed_init: u32,
}

/// Cumulative per-phase wall-clock time (seconds), accumulated across every
/// [`Model::update`] call within the most recent [`Model::fit`]. Mirrors
/// Python's `GCLDAModel.phase_times_` exactly (same five keys), so a
/// benchmark driver can compare Rust against Python phase-by-phase rather
/// than as one aggregate ratio.
///
/// `total` is the wall-clock time of the four phases below PLUS the untimed
/// bookkeeping interleaved with them (iteration/seed increments) inside
/// [`Model::update`] -- i.e. NOT simply the sum of the other four fields, but
/// it deliberately EXCLUDES the optional per-iteration state dump
/// (`dump_state`, called from [`Model::fit`] after `update` returns, not from
/// `update` itself): that write is diagnostic I/O for the equality harness,
/// not model computation. This matches Python's `_update` exactly, where the
/// `total` timestamp is likewise taken before Python's own `_dump_state`
/// call -- so `total` means the same thing on both sides whether or not
/// `dump_state_dir`/`--dump-state-dir` is used.
#[derive(Default)]
pub struct PhaseTimes {
    pub word_sampling: f64,
    pub peak_sampling: f64,
    pub region_update: f64,
    pub loglikelihood: f64,
    pub total: f64,
}

/// Full model state: corpus, mask geometry, hyperparameters, and all
/// assignment/count arrays. All 2-D fields are row-major flat `Vec`s; use
/// [`Model::at`] to compute `row * n_cols + col` consistently.
pub struct Model {
    pub corpus: Corpus,
    pub mask: MaskInfo,
    pub params: Params,

    /// Word-token -> topic assignment (z), length = number of word tokens.
    pub wtoken_topic_idx: Vec<u32>,
    /// Peak-token -> topic assignment (y), length = number of peak tokens.
    pub peak_topic_idx: Vec<u32>,
    /// Peak-token -> subregion assignment (c), length = number of peak tokens.
    pub peak_region_idx: Vec<u32>,

    /// D x T: number of peak-tokens assigned to each topic, per document.
    pub n_peak_tokens_doc_by_topic: Vec<i64>,
    /// R x T: number of peak-tokens assigned to each subregion, per topic.
    pub n_peak_tokens_region_by_topic: Vec<i64>,
    /// W x T: number of word-tokens assigned to each topic, per word-type.
    pub n_word_tokens_word_by_topic: Vec<i64>,
    /// D x T: number of word-tokens assigned to each topic, per document.
    pub n_word_tokens_doc_by_topic: Vec<i64>,
    /// T: total number of word-tokens assigned to each topic.
    pub total_n_word_tokens_by_topic: Vec<i64>,

    /// T*R (topic-major, region-minor): Gaussian mean per topic/subregion.
    pub regions_mu: Vec<[f64; 3]>,
    /// T*R: Gaussian covariance per topic/subregion.
    pub regions_sigma: Vec<[[f64; 3]; 3]>,
    /// T*R: Gaussian precision (inverse covariance) per topic/subregion.
    pub regions_precision: Vec<[[f64; 3]; 3]>,
    /// T*R: Gaussian log-normalization constant per topic/subregion.
    pub regions_log_norm: Vec<f64>,

    /// The global sampling iteration of the model.
    pub iter: usize,
    /// Current random seed, incremented after initialization and each
    /// sampling update.
    pub seed: u32,

    /// The `n_iters` argument passed to the most recent [`Model::fit`] call.
    /// Not part of `GCLDAModel.__init__` -- Python's `fit`/`_update` take
    /// this as a plain argument and never store it. Rust stores it because
    /// `write_outputs`'s signature is `(model, dir, dtype)` with no separate
    /// channel to report it in `model.json`.
    pub n_iters: usize,
    /// The `loglikely_freq` argument passed to the most recent
    /// [`Model::fit`] call. See `n_iters` above for why this is stored.
    pub loglikely_freq: usize,
    /// `(iter, x, w, total)` recorded each time `fit`'s loop computes the
    /// log-likelihood (mirrors Python's `self.loglikelihood` dict of lists).
    pub loglikelihood_history: Vec<(usize, f64, f64, f64)>,
    /// Cumulative per-phase timing, mirrors Python's `phase_times_`. See
    /// [`PhaseTimes`] for field-by-field semantics.
    pub phase_times: PhaseTimes,
}

impl Model {
    /// Row-major index into an (n_rows x n_cols) flat `Vec`.
    #[inline]
    pub fn at(row: usize, col: usize, n_cols: usize) -> usize {
        row * n_cols + col
    }

    pub fn new(corpus: Corpus, mask: MaskInfo, params: Params) -> Result<Model, GcldaError> {
        // Step 1: validate. A symmetric model requires an even n_regions so
        // that subregions can be paired up (see step 4 below).
        if params.symmetric && params.n_regions % 2 != 0 {
            return Err(GcldaError::Parse(
                "Cannot run a symmetric model unless n_regions is even.".to_string(),
            ));
        }

        let n_topics = params.n_topics;
        let n_regions = params.n_regions;
        let n_docs = corpus.ids.len();
        let n_words = corpus.vocabulary.len();
        let n_peaks = corpus.ptoken_doc_idx.len();

        // Step 2: seed one RNG stream for everything that follows in this
        // block (peak_topic_idx and peak_region_idx).
        let mut rng = Mt19937::new(params.seed_init);

        // Step 3: peak->topic assignments (y) ~ unif(0..n_topics), one
        // bounded draw per peak, in peak order.
        let peak_topic_idx: Vec<u32> = (0..n_peaks)
            .map(|_| rng.randint(n_topics as u64) as u32)
            .collect();

        // Step 4: peak->subregion assignments (r).
        let peak_region_idx: Vec<u32> = if params.symmetric {
            // Symmetric: draw a pair index in [0, n_pairs), then assign the
            // even/odd region within the pair deterministically from the
            // sign of the peak's x-coordinate. n_pairs may be 1 (n_regions
            // == 2): np.random.randint(1, size=n) still draws 0 for every
            // element (see rng.rs for the bound == 1 note), and here the
            // final result cannot distinguish "consumed a draw" from
            // "didn't", because the constructor re-seeds the stream from
            // scratch immediately afterward (step 6) with nothing in
            // between that reads the RNG.
            let n_pairs = n_regions / 2;
            (0..n_peaks)
                .map(|i| {
                    let initial = rng.randint(n_pairs as u64) as u32;
                    let sign = if corpus.ptoken_coords[i][0] > 0.0 { 1u32 } else { 0u32 };
                    initial * 2 + sign
                })
                .collect()
        } else {
            // Asymmetric: r ~ unif(0..n_regions), one bounded draw per peak.
            (0..n_peaks)
                .map(|_| rng.randint(n_regions as u64) as u32)
                .collect()
        };

        // Step 5: accumulate peak-token counts from the assignments above.
        let mut n_peak_tokens_doc_by_topic = vec![0i64; n_docs * n_topics];
        let mut n_peak_tokens_region_by_topic = vec![0i64; n_regions * n_topics];
        for i in 0..n_peaks {
            let doc = corpus.ptoken_doc_idx[i] as usize;
            let topic = peak_topic_idx[i] as usize;
            let region = peak_region_idx[i] as usize;
            n_peak_tokens_doc_by_topic[Model::at(doc, topic, n_topics)] += 1;
            n_peak_tokens_region_by_topic[Model::at(region, topic, n_topics)] += 1;
        }

        // Step 6: word->topic assignments (z), sampled proportional to
        // p(topic|doc). This RE-SEEDS the same stream from scratch --
        // mirrors `_jit_initialize_word_topic_assignments`'s
        // `np.random.seed(randseed)` call.
        rng.reseed(params.seed_init);
        let n_wtokens = corpus.wtoken_word_idx.len();
        let mut wtoken_topic_idx = vec![0u32; n_wtokens];
        let mut n_word_tokens_word_by_topic = vec![0i64; n_words * n_topics];
        let mut n_word_tokens_doc_by_topic = vec![0i64; n_docs * n_topics];
        let mut total_n_word_tokens_by_topic = vec![0i64; n_topics];
        let mut probs = vec![0.0f64; n_topics];

        for i in 0..n_wtokens {
            let doc = corpus.wtoken_doc_idx[i] as usize;
            let word = corpus.wtoken_word_idx[i] as usize;

            for t in 0..n_topics {
                probs[t] = n_peak_tokens_doc_by_topic[Model::at(doc, t, n_topics)] as f64
                    + params.gamma;
            }

            let topic = rng.sample_from_unnormalized(&probs)?;
            wtoken_topic_idx[i] = topic as u32;
            n_word_tokens_word_by_topic[Model::at(word, topic, n_topics)] += 1;
            total_n_word_tokens_by_topic[topic] += 1;
            n_word_tokens_doc_by_topic[Model::at(doc, topic, n_topics)] += 1;
        }

        // Spatial Gaussians are preallocated as zeros; they are populated by
        // `_update_regions` (a later task), not by the constructor.
        let n_tr = n_topics * n_regions;
        let regions_mu = vec![[0.0f64; 3]; n_tr];
        let regions_sigma = vec![[[0.0f64; 3]; 3]; n_tr];
        let regions_precision = vec![[[0.0f64; 3]; 3]; n_tr];
        let regions_log_norm = vec![0.0f64; n_tr];

        Ok(Model {
            corpus,
            mask,
            params,
            wtoken_topic_idx,
            peak_topic_idx,
            peak_region_idx,
            n_peak_tokens_doc_by_topic,
            n_peak_tokens_region_by_topic,
            n_word_tokens_word_by_topic,
            n_word_tokens_doc_by_topic,
            total_n_word_tokens_by_topic,
            regions_mu,
            regions_sigma,
            regions_precision,
            regions_log_norm,
            // Step 7.
            iter: 0,
            seed: 0,
            n_iters: 0,
            loglikely_freq: 0,
            loglikelihood_history: Vec::new(),
            phase_times: PhaseTimes::default(),
        })
    }
}
