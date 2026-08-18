use gclda::io::{nifti::load_mask_xyz, tsv::load_corpus};
use gclda::model::{Model, Params};
use std::path::PathBuf;

mod common;
use common::load;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

#[test]
fn word_topic_sweep_matches_python() {
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

    let fx = load("word_sampler.json");
    let seed = fx["seed"].as_u64().unwrap() as u32;
    model.update_word_topic_assignments(seed).unwrap();

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

    assert_eq!(model.wtoken_topic_idx, want_u32("wtoken_topic_idx"), "wtoken_topic_idx");
    assert_eq!(
        model.n_word_tokens_word_by_topic,
        flat("n_word_tokens_word_by_topic"),
        "n_word_tokens_word_by_topic"
    );
    assert_eq!(
        model.n_word_tokens_doc_by_topic,
        flat("n_word_tokens_doc_by_topic"),
        "n_word_tokens_doc_by_topic"
    );
    assert_eq!(
        model.total_n_word_tokens_by_topic,
        flat("total_n_word_tokens_by_topic"),
        "total_n_word_tokens_by_topic"
    );
}
