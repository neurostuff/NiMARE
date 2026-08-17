//! End-to-end test for `Model::fit` and `output::write_outputs`.
//!
//! Fits a few iterations on the fixture corpus, writes every output file to
//! a temp directory, then shells out to NumPy to check that each file loads
//! with the expected shape and that the four probability matrices are
//! properly normalized (rows/columns sum to 1, allowing all-zero rows/
//! columns for topics or words with no observations -- `nan_to_num` is what
//! makes those legal instead of NaN).
//!
//! This is a structural/shape test, not a bit-exact golden comparison: the
//! project's bit-exact regression harness against the Python reference
//! lives in a later task (`nimare/tests/test_gclda_rust.py`), which needs
//! the CLI binary (a later task) and a Python-side loader (a later task) to
//! exist first.

use gclda::io::npy::Dtype;
use gclda::io::{nifti::load_mask_xyz, tsv::load_corpus};
use gclda::model::{Model, Params};
use gclda::output::write_outputs;
use std::path::PathBuf;
use std::process::Command;

mod common;
use common::load;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

/// Build a model on the fixture corpus with the same config already known
/// (from Task 10/11's region-update fixtures) to exercise the n_obs == 0
/// zero-observation fallback for one topic: n_topics=3, n_regions=2,
/// symmetric=true, seed_init=1.
fn build_model() -> Model {
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
    };
    Model::new(corpus, mask, params).unwrap()
}

/// Run a Python script (via micromamba) that loads every expected output
/// file and checks shapes/normalization; prints "OK" on success and raises
/// (causing a nonzero exit + AssertionError on stderr) otherwise.
fn verify_outputs(dir: &std::path::Path, n_topics: usize, n_regions: usize) -> String {
    let script = format!(
        r#"
import json
import numpy as np

d = r"{dir}"
n_topics = {n_topics}
n_regions = {n_regions}

def sums_to_one_or_zero(v, axis):
    s = v.sum(axis=axis)
    ok = np.isclose(s, 1.0) | np.isclose(s, 0.0)
    assert ok.all(), f"sums not 0 or 1 along axis {{axis}}: {{s[~ok]}}"

ids = json.load(open(d + "/model.json"))
vocab = open(d + "/vocabulary.txt").read().splitlines()
n_words = len(vocab)
n_docs = len(ids["ids"])

assert ids["n_topics"] == n_topics
assert ids["n_regions"] == n_regions
assert ids["symmetric"] is True
assert ids["alpha"] == 0.1
assert ids["beta"] == 0.01
assert ids["gamma"] == 0.01
assert ids["delta"] == 1.0
assert ids["dobs"] == 25.0
assert ids["roi_size"] == 50.0
assert ids["seed_init"] == 1
assert ids["n_iters"] == 3
assert ids["loglikely_freq"] == 1
assert ids["mask_path"], "mask_path must be non-empty"
assert len(ids["mask_affine"]) == 4 and all(len(r) == 4 for r in ids["mask_affine"])
assert len(ids["mask_shape"]) == 3
n_voxels = ids["n_voxels"]
assert n_voxels > 0
assert set(ids["phase_times"].keys()) == {{
    "word_sampling", "peak_sampling", "region_update", "loglikelihood", "total"
}}
assert all(v == 0 for v in ids["phase_times"].values())

p_topic_g_voxel = np.load(d + "/p_topic_g_voxel.npy")
p_voxel_g_topic = np.load(d + "/p_voxel_g_topic.npy")
p_topic_g_word = np.load(d + "/p_topic_g_word.npy")
p_word_g_topic = np.load(d + "/p_word_g_topic.npy")

assert p_topic_g_voxel.shape == (n_voxels, n_topics), p_topic_g_voxel.shape
assert p_voxel_g_topic.shape == (n_voxels, n_topics), p_voxel_g_topic.shape
assert p_topic_g_word.shape == (n_words, n_topics), p_topic_g_word.shape
assert p_word_g_topic.shape == (n_words, n_topics), p_word_g_topic.shape

assert p_topic_g_voxel.dtype == np.float64
assert p_voxel_g_topic.dtype == np.float64
assert p_topic_g_word.dtype == np.float64
assert p_word_g_topic.dtype == np.float64

sums_to_one_or_zero(p_topic_g_voxel, axis=1)   # rows sum to 1
sums_to_one_or_zero(p_voxel_g_topic, axis=0)   # columns sum to 1
sums_to_one_or_zero(p_topic_g_word, axis=1)    # rows sum to 1
sums_to_one_or_zero(p_word_g_topic, axis=0)    # columns sum to 1
assert (p_topic_g_voxel >= 0).all()
assert (p_voxel_g_topic >= 0).all()
assert (p_topic_g_word >= 0).all()
assert (p_word_g_topic >= 0).all()

n_word_tokens_word_by_topic = np.load(d + "/n_word_tokens_word_by_topic.npy")
n_peak_tokens_doc_by_topic = np.load(d + "/n_peak_tokens_doc_by_topic.npy")
n_peak_tokens_region_by_topic = np.load(d + "/n_peak_tokens_region_by_topic.npy")
assert n_word_tokens_word_by_topic.shape == (n_words, n_topics)
assert n_word_tokens_word_by_topic.dtype == np.int64
assert n_peak_tokens_doc_by_topic.shape == (n_docs, n_topics)
assert n_peak_tokens_doc_by_topic.dtype == np.int64
assert n_peak_tokens_region_by_topic.shape == (n_regions, n_topics)
assert n_peak_tokens_region_by_topic.dtype == np.int64

regions_mu = np.load(d + "/regions_mu.npy")
regions_sigma = np.load(d + "/regions_sigma.npy")
assert regions_mu.shape == (n_topics, n_regions, 3)
assert regions_sigma.shape == (n_topics, n_regions, 3, 3)

wtoken_topic_idx = np.load(d + "/wtoken_topic_idx.npy")
peak_topic_idx = np.load(d + "/peak_topic_idx.npy")
peak_region_idx = np.load(d + "/peak_region_idx.npy")
assert wtoken_topic_idx.dtype == np.int64
assert peak_topic_idx.dtype == np.int64
assert peak_region_idx.dtype == np.int64
assert peak_topic_idx.shape == peak_region_idx.shape
assert (wtoken_topic_idx >= 0).all() and (wtoken_topic_idx < n_topics).all()
assert (peak_topic_idx >= 0).all() and (peak_topic_idx < n_topics).all()
assert (peak_region_idx >= 0).all() and (peak_region_idx < n_regions).all()

with open(d + "/loglikelihood.tsv") as fo:
    header = fo.readline().strip().split("\t")
    rows = [line.strip().split("\t") for line in fo if line.strip()]
assert header == ["iter", "x", "w", "total"], header
# iter==0 (initial) plus one row per fit iteration at loglikely_freq==1.
assert [r[0] for r in rows] == ["0", "1", "2", "3"], rows

print("OK")
"#,
        dir = dir.display(),
        n_topics = n_topics,
        n_regions = n_regions,
    );

    let out = Command::new("micromamba")
        .args(["run", "-n", "nimenv", "python", "-c", &script])
        .output()
        .expect("failed to run micromamba");
    assert!(
        out.status.success(),
        "verification script failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

#[test]
fn fit_and_write_outputs_produce_valid_files() {
    let mut model = build_model();
    // `fit` now takes a progress callback (invoked from inside its loop --
    // see src/output.rs); this test only checks recorded history/output
    // files, not progress output, so pass a no-op.
    model.fit(3, 1, |_, _| {}).unwrap();

    assert_eq!(model.iter, 3);
    // iter 0 (initial) + iters 1,2,3 at loglikely_freq=1 => 4 recorded entries.
    assert_eq!(model.loglikelihood_history.len(), 4);

    let dir = std::env::temp_dir().join(format!(
        "gclda_outputs_test_{}",
        std::process::id()
    ));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }

    write_outputs(&model, &dir, Dtype::F64).unwrap();

    let desc = verify_outputs(&dir, 3, 2);
    assert_eq!(desc, "OK");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn fit_zero_iters_still_records_initial_loglikelihood_and_regions() {
    let mut model = build_model();
    model.fit(0, 1, |_, _| {}).unwrap();

    assert_eq!(model.iter, 0);
    assert_eq!(
        model.loglikelihood_history.len(),
        1,
        "n_iters=0 must still run the iter==0 initial region update + log-likelihood"
    );
    // regions_log_norm must have been populated by the initial update_regions()
    // call, not left at its zeroed constructor default.
    assert!(model.regions_log_norm.iter().any(|&v| v != 0.0));
}
