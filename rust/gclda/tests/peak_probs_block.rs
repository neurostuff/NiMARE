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
        peak_block_size: 8192,
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
