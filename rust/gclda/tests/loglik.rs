use gclda::io::{nifti::load_mask_xyz, tsv::load_corpus};
use gclda::model::{Model, Params};
use std::path::PathBuf;

mod common;
use common::{bits_to_f64, load};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

/// Compare `Model::compute_log_likelihood` against Python's
/// `compute_log_likelihood` for two configurations (symmetric n_regions=2,
/// asymmetric n_regions=1) dumped by `gen_loglik` in
/// `nimare/tests/generate_gclda_fixtures.py`.
///
/// `x` (the peak term) and `total` are asserted with a relative tolerance of
/// 1e-10 -- not the bit-exactness used everywhere else in this crate --
/// because `total = x + w` and `w` (the word term) is itself only
/// tolerance-comparable: the Python reference computes it via
/// `np.dot(docprobs_z, wordprobs.T)`, which is routed through BLAS.  BLAS's
/// summation order (and its use of fused multiply-add) is a property of the
/// BLAS implementation on the machine that generated the fixture, not of the
/// Python source, so it cannot be reproduced from scalar Rust code. This is
/// the ONLY quantity in the whole crate that is not asserted bit-exact; do
/// not widen any other assertion by analogy with this one, and do not
/// tighten this one back to bit-exactness -- that would make the test flaky
/// across BLAS builds.
#[test]
fn log_likelihood_matches_python() {
    let cases = load("loglik.json");
    let cases = cases.as_array().unwrap();
    assert_eq!(cases.len(), 2, "expected 2 configs in loglik.json");

    for case in cases {
        let cfg = &case["config"];
        let n_topics = cfg["n_topics"].as_u64().unwrap() as usize;
        let n_regions = cfg["n_regions"].as_u64().unwrap() as usize;
        let symmetric = cfg["symmetric"].as_bool().unwrap();
        let seed_init = cfg["seed_init"].as_u64().unwrap() as u32;

        let mask_meta = load("mask_xyz.json");
        let mask_path = common::repo_path(mask_meta["path"].as_str().unwrap());
        let corpus = load_corpus(&fixture("counts.tsv"), &fixture("coordinates.tsv")).unwrap();
        let mask = load_mask_xyz(&mask_path).unwrap();
        let params = Params {
            n_topics,
            n_regions,
            symmetric,
            alpha: 0.1,
            beta: 0.01,
            gamma: 0.01,
            delta: 1.0,
            dobs: 25.0,
            roi_size: 50.0,
            seed_init,
        };
        let mut model = Model::new(corpus, mask, params).unwrap();
        model.update_regions().unwrap();

        let ll = model.compute_log_likelihood();

        let label = format!(
            "n_topics={n_topics} n_regions={n_regions} symmetric={symmetric} seed_init={seed_init}"
        );

        let want_x = bits_to_f64(case["x"].as_str().unwrap());
        let want_w = bits_to_f64(case["w"].as_str().unwrap());
        let want_total = bits_to_f64(case["total"].as_str().unwrap());

        let rel_err = |got: f64, want: f64| -> f64 { (got - want).abs() / want.abs() };

        assert!(
            rel_err(ll.x, want_x) < 1e-10,
            "{label} x: got {:?} want {want_x:?} rel_err={}",
            ll.x,
            rel_err(ll.x, want_x)
        );
        assert!(
            rel_err(ll.w, want_w) < 1e-10,
            "{label} w: got {:?} want {want_w:?} rel_err={}",
            ll.w,
            rel_err(ll.w, want_w)
        );
        assert!(
            rel_err(ll.total, want_total) < 1e-10,
            "{label} total: got {:?} want {want_total:?} rel_err={}",
            ll.total,
            rel_err(ll.total, want_total)
        );
    }
}
