use gclda::io::{nifti::load_mask_xyz, tsv::load_corpus};
use gclda::model::{Model, Params};
use std::path::PathBuf;

mod common;
use common::{bits_to_f64, load};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

/// Compare `model.update_regions()` output against Python's `_update_regions`
/// (`nimare/annotate/gclda.py:827-961`) bit-for-bit, across the four
/// configurations dumped by `gen_region_update` in
/// `nimare/tests/generate_gclda_fixtures.py`: symmetric with n_regions 2 and
/// 4 (exercising the paired-subregion branch, `gclda.py:847-927`), and
/// asymmetric with n_regions 1 and 3 (`gclda.py:928-961`).
#[test]
fn region_update_matches_python() {
    let cases = load("region_update.json");
    let cases = cases.as_array().unwrap();
    assert_eq!(cases.len(), 4, "expected 4 configs in region_update.json");

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

        let label = format!(
            "n_topics={n_topics} n_regions={n_regions} symmetric={symmetric} seed_init={seed_init}"
        );

        let mu_arr = case["regions_mu"].as_array().unwrap();
        let sigma_arr = case["regions_sigma"].as_array().unwrap();
        let prec_arr = case["regions_precision"].as_array().unwrap();
        let logn_arr = case["regions_log_norm"].as_array().unwrap();

        for t in 0..n_topics {
            for r in 0..n_regions {
                let idx = Model::at(t, r, n_regions);

                let mu_v = mu_arr[t][r].as_array().unwrap();
                for d in 0..3 {
                    let want = bits_to_f64(mu_v[d].as_str().unwrap());
                    let got = model.regions_mu[idx][d];
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{label} regions_mu[topic={t}][region={r}][{d}]: got {got:?} want {want:?}"
                    );
                }

                let sigma_v = sigma_arr[t][r].as_array().unwrap();
                let prec_v = prec_arr[t][r].as_array().unwrap();
                for i in 0..3 {
                    let sigma_row = sigma_v[i].as_array().unwrap();
                    let prec_row = prec_v[i].as_array().unwrap();
                    for j in 0..3 {
                        let want_s = bits_to_f64(sigma_row[j].as_str().unwrap());
                        let got_s = model.regions_sigma[idx][i][j];
                        assert_eq!(
                            got_s.to_bits(),
                            want_s.to_bits(),
                            "{label} regions_sigma[topic={t}][region={r}][{i}][{j}]: got {got_s:?} want {want_s:?}"
                        );

                        let want_p = bits_to_f64(prec_row[j].as_str().unwrap());
                        let got_p = model.regions_precision[idx][i][j];
                        assert_eq!(
                            got_p.to_bits(),
                            want_p.to_bits(),
                            "{label} regions_precision[topic={t}][region={r}][{i}][{j}]: got {got_p:?} want {want_p:?}"
                        );
                    }
                }

                let want_ln = bits_to_f64(logn_arr[t][r].as_str().unwrap());
                let got_ln = model.regions_log_norm[idx];
                assert_eq!(
                    got_ln.to_bits(),
                    want_ln.to_bits(),
                    "{label} regions_log_norm[topic={t}][region={r}]: got {got_ln:?} want {want_ln:?}"
                );
            }
        }
    }
}
