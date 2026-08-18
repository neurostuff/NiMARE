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
        peak_block_size: 8192,
    };
    let mut model = Model::new(corpus, mask, params).unwrap();
    model.update_regions().unwrap();
    model
}

#[test]
fn serial_pdf_pass_evaluates_every_peak_and_reports_positive_time() {
    let model = fixture_model();
    let n_peaks = model.corpus.ptoken_coords.len();
    assert!(n_peaks > 0, "fixture corpus has no peaks; this test would pass vacuously");

    let (seconds, n_evaluated) = model.time_serial_pdf_pass();

    assert_eq!(
        n_evaluated, n_peaks,
        "serial PDF pass evaluated {n_evaluated} peaks, expected all {n_peaks} fixture peaks"
    );
    assert!(
        seconds > 0.0,
        "serial PDF pass reported {seconds} seconds; timer is not measuring anything"
    );
}
