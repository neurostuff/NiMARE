use gclda::io::tsv::load_corpus;
use std::path::PathBuf;

mod common;
use common::{bits_to_f64, load};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

#[test]
fn ingest_matches_python_constructor() {
    let expected = load("ingest.json");
    let corpus = load_corpus(&fixture("counts.tsv"), &fixture("coordinates.tsv")).unwrap();

    let want_ids: Vec<String> = expected["ids"]
        .as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    assert_eq!(corpus.ids, want_ids, "document IDs (sorted as STRINGS)");

    let want_vocab: Vec<String> = expected["vocabulary"]
        .as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    assert_eq!(corpus.vocabulary, want_vocab, "vocabulary after dropping all-zero terms");

    let as_u32 = |k: &str| -> Vec<u32> {
        expected[k].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect()
    };
    assert_eq!(corpus.wtoken_doc_idx, as_u32("wtoken_doc_idx"));
    assert_eq!(corpus.wtoken_word_idx, as_u32("wtoken_word_idx"));
    assert_eq!(corpus.ptoken_doc_idx, as_u32("ptoken_doc_idx"));

    let want_coords = expected["ptoken_coords"].as_array().unwrap();
    assert_eq!(corpus.ptoken_coords.len(), want_coords.len());
    for (i, row) in want_coords.iter().enumerate() {
        let r = row.as_array().unwrap();
        for j in 0..3 {
            let want = bits_to_f64(r[j].as_str().unwrap());
            assert_eq!(corpus.ptoken_coords[i][j].to_bits(), want.to_bits(), "coord[{i}][{j}]");
        }
    }
}
