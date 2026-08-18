use gclda::io::{nifti::load_mask_xyz, tsv::load_corpus};
use gclda::model::{Model, Params};
use std::path::PathBuf;

mod common;
use common::{bits_to_f64, load};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

#[test]
fn peak_topic_region_sweep_matches_python() {
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

    let fx = load("peak_sampler.json");

    // Task 11 (`_update_regions`) has not been ported yet. Load the region
    // Gaussian parameters the fixture pins directly into the model's region
    // fields, so this test can exercise `update_peak_assignments` in
    // isolation without reimplementing region estimation here.
    let n_topics = model.params.n_topics;
    let n_regions = model.params.n_regions;
    let mu_arr = fx["regions_mu"].as_array().unwrap();
    let prec_arr = fx["regions_precision"].as_array().unwrap();
    let logn_arr = fx["regions_log_norm"].as_array().unwrap();
    for t in 0..n_topics {
        for r in 0..n_regions {
            let idx = Model::at(t, r, n_regions);
            let mu_v = mu_arr[t][r].as_array().unwrap();
            let mut mu = [0.0f64; 3];
            for d in 0..3 {
                mu[d] = bits_to_f64(mu_v[d].as_str().unwrap());
            }
            model.regions_mu[idx] = mu;

            let prec_v = prec_arr[t][r].as_array().unwrap();
            let mut prec = [[0.0f64; 3]; 3];
            for i in 0..3 {
                let row = prec_v[i].as_array().unwrap();
                for j in 0..3 {
                    prec[i][j] = bits_to_f64(row[j].as_str().unwrap());
                }
            }
            model.regions_precision[idx] = prec;

            model.regions_log_norm[idx] =
                bits_to_f64(logn_arr[t][r].as_str().unwrap());
        }
    }

    let seed = fx["seed"].as_u64().unwrap() as u32;
    model.update_peak_assignments(seed).unwrap();

    let want_u32 = |k: &str| -> Vec<u32> {
        fx[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect()
    };
    let flat = |k: &str| -> Vec<i64> {
        fx[k]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| match row.as_array() {
                Some(r) => r.iter().map(|v| v.as_i64().unwrap()).collect::<Vec<_>>(),
                None => vec![row.as_i64().unwrap()],
            })
            .collect()
    };

    assert_eq!(model.peak_topic_idx, want_u32("peak_topic_idx"), "peak_topic_idx");
    assert_eq!(model.peak_region_idx, want_u32("peak_region_idx"), "peak_region_idx");
    assert_eq!(
        model.n_peak_tokens_doc_by_topic,
        flat("n_peak_tokens_doc_by_topic"),
        "n_peak_tokens_doc_by_topic"
    );
    assert_eq!(
        model.n_peak_tokens_region_by_topic,
        flat("n_peak_tokens_region_by_topic"),
        "n_peak_tokens_region_by_topic"
    );
}
