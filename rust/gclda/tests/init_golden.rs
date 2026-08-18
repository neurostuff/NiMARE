use gclda::io::{nifti::load_mask_xyz, tsv::load_corpus};
use gclda::model::{Model, Params};
use std::path::PathBuf;

mod common;
use common::load;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

#[test]
fn initial_state_matches_python_constructor() {
    let mask_meta = load("mask_xyz.json");
    let mask_path = common::repo_path(mask_meta["path"].as_str().unwrap());

    for (c, case) in load("init_state.json").as_array().unwrap().iter().enumerate() {
        let cfg = &case["config"];
        let params = Params {
            n_topics: cfg["n_topics"].as_u64().unwrap() as usize,
            n_regions: cfg["n_regions"].as_u64().unwrap() as usize,
            symmetric: cfg["symmetric"].as_bool().unwrap(),
            alpha: 0.1, beta: 0.01, gamma: 0.01, delta: 1.0,
            dobs: 25.0, roi_size: 50.0,
            seed_init: cfg["seed_init"].as_u64().unwrap() as u32,
            peak_block_size: 8192,
        };
        let corpus = load_corpus(&fixture("counts.tsv"), &fixture("coordinates.tsv")).unwrap();
        let mask = load_mask_xyz(&mask_path).unwrap();
        let model = Model::new(corpus, mask, params).unwrap();

        let want_u32 = |k: &str| -> Vec<u32> {
            case[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect()
        };
        assert_eq!(model.peak_topic_idx, want_u32("peak_topic_idx"), "case {c} peak_topic_idx");
        assert_eq!(model.peak_region_idx, want_u32("peak_region_idx"), "case {c} peak_region_idx");
        assert_eq!(model.wtoken_topic_idx, want_u32("wtoken_topic_idx"), "case {c} wtoken_topic_idx");

        let flat = |k: &str| -> Vec<i64> {
            case[k].as_array().unwrap().iter()
                .flat_map(|row| match row.as_array() {
                    Some(r) => r.iter().map(|v| v.as_i64().unwrap()).collect::<Vec<_>>(),
                    None => vec![row.as_i64().unwrap()],
                })
                .collect()
        };
        assert_eq!(model.n_peak_tokens_doc_by_topic, flat("n_peak_tokens_doc_by_topic"), "case {c}");
        assert_eq!(model.n_peak_tokens_region_by_topic, flat("n_peak_tokens_region_by_topic"), "case {c}");
        assert_eq!(model.n_word_tokens_word_by_topic, flat("n_word_tokens_word_by_topic"), "case {c}");
        assert_eq!(model.n_word_tokens_doc_by_topic, flat("n_word_tokens_doc_by_topic"), "case {c}");
        assert_eq!(model.total_n_word_tokens_by_topic, flat("total_n_word_tokens_by_topic"), "case {c}");
    }
}

#[test]
fn symmetric_with_odd_regions_is_rejected() {
    let corpus = load_corpus(&fixture("counts.tsv"), &fixture("coordinates.tsv")).unwrap();
    let mask_meta = load("mask_xyz.json");
    let mask = load_mask_xyz(&common::repo_path(mask_meta["path"].as_str().unwrap())).unwrap();
    let params = Params {
        n_topics: 3, n_regions: 3, symmetric: true,
        alpha: 0.1, beta: 0.01, gamma: 0.01, delta: 1.0,
        dobs: 25.0, roi_size: 50.0, seed_init: 1, peak_block_size: 8192,
    };
    assert!(Model::new(corpus, mask, params).is_err());
}
