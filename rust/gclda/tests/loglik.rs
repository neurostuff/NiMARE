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
/// `x`, `w`, and `total` are ALL asserted with a relative tolerance of
/// 1e-10 -- not the bit-exactness used everywhere else in this crate --
/// for two independent reasons (see the doc comment on `LogLikelihood` in
/// `src/loglik.rs` for the full explanation):
///
/// 1. Python routes BOTH the word term (`np.dot(docprobs_z, wordprobs.T)`)
///    and the peak term (`np.dot(p_region_g_doc, p_x_r)` per region)
///    through BLAS, whose summation order/FMA use is a property of the
///    BLAS build, not the Python source.
/// 2. `docprobs_y`/`docprobs_z` are normalized by `np.sum(..., axis=1)`,
///    a contiguous-axis sum that NumPy computes via pairwise summation,
///    which diverges from a plain accumulator once the reduction length
///    (`n_topics`) reaches 8 -- see `src/pairwise_sum.rs`.
///
/// At this fixture's `n_topics=3`, both effects are below their thresholds,
/// so the measured relative error here is exactly 0e0. That is a property
/// of this small fixture, NOT evidence that `x`/`w`/`total` are bit-exact
/// in general -- do not tighten this to `assert_eq!` on the strength of
/// that observation, and do not widen any other assertion in the crate by
/// analogy with this one; every other quantity here has neither a BLAS dot
/// product nor an `np.sum`-pairwise dependency and remains genuinely
/// bit-exact.
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
            peak_block_size: 8192,
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
