//! End-to-end, bit-exact check that `output.rs` actually uses
//! `pairwise_sum::numpy_sum` for `p_topic_g_voxel`'s row-sum, at the
//! `n_topics=100` production scale that motivated it.
//!
//! `tests/pairwise_sum.rs` verifies the `numpy_sum` algorithm in isolation
//! against `np.sum`. This test verifies it is wired into the right place:
//! it trains a real (tiny-corpus, tiny-mask) model through
//! `Model::fit`/`write_outputs`, then has Python recompute the reference
//! `p_topic_g_voxel` from the SAME `regions_mu`/`regions_sigma` Rust wrote
//! out and the SAME voxel coordinates, by calling `nimare.annotate.gclda`'s
//! own `_inv3_logdet` and `_jit_get_spatial_dists` (the actual Python
//! reference implementations, not a hand-rolled reimplementation of them)
//! followed by a real `np.sum(..., axis=1)`. If `output.rs` used a plain
//! accumulation loop instead of `numpy_sum`, this comparison would fail at
//! `n_topics=100` (see `tests/pairwise_sum.rs`'s
//! `pairwise_and_naive_summation_genuinely_differ_at_production_scale` for
//! why that divergence is real and not just theoretical) even though every
//! other test in this crate's suite -- all at `n_topics <= 5` -- would
//! still pass.
//!
//! Uses a synthetic 6-voxel `MaskInfo` (constructed directly, not loaded
//! from a NIfTI file) instead of the real ~228k-voxel MNI mask used
//! elsewhere in this crate's tests, purely to keep this test fast: the
//! `numpy_sum` row-sum path being checked operates per-voxel over
//! `n_topics` elements, so voxel count doesn't change what's being tested,
//! only the runtime.

use gclda::io::nifti::MaskInfo;
use gclda::io::npy::Dtype;
use gclda::io::tsv::load_corpus;
use gclda::model::{Model, Params};
use gclda::output::write_outputs;
use std::path::PathBuf;
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

const VOXELS: [[f64; 3]; 6] = [
    [0.0, 0.0, 0.0],
    [10.0, -10.0, 5.0],
    [-20.0, 30.0, -15.0],
    [5.0, 5.0, 5.0],
    [-5.0, -5.0, -5.0],
    [15.0, 0.0, -10.0],
];

#[test]
fn p_topic_g_voxel_row_sum_matches_python_at_n_topics_100() {
    let mask = MaskInfo {
        xyz: VOXELS.to_vec(),
        affine: [
            [2.0, 0.0, 0.0, -90.0],
            [0.0, 2.0, 0.0, -126.0],
            [0.0, 0.0, 2.0, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        shape: [91, 109, 91],
        path: PathBuf::from("synthetic-test-mask.nii.gz"),
    };
    let corpus = load_corpus(&fixture("counts.tsv"), &fixture("coordinates.tsv")).unwrap();
    let params = Params {
        n_topics: 100,
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
    let mut model = Model::new(corpus, mask, params).unwrap();
    model.fit(2, 1, None, |_, _| {}).unwrap();

    let dir = std::env::temp_dir()
        .join(format!("gclda_pairwise_wiring_test_{}", std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    write_outputs(&model, &dir, Dtype::F64).unwrap();

    let voxel_literal: Vec<String> = VOXELS
        .iter()
        .map(|p| format!("[{}, {}, {}]", p[0], p[1], p[2]))
        .collect();
    let script = format!(
        r#"
import numpy as np
from nimare.annotate.gclda import _inv3_logdet, _jit_get_spatial_dists

d = r"{dir}"
regions_mu = np.load(d + "/regions_mu.npy")       # (T, R, 3)
regions_sigma = np.load(d + "/regions_sigma.npy")  # (T, R, 3, 3)
n_topics, n_regions, _ = regions_mu.shape

regions_precision = np.zeros((n_topics, n_regions, 3, 3))
regions_log_norm = np.zeros((n_topics, n_regions))
for t in range(n_topics):
    for r in range(n_regions):
        inv, logdet = _inv3_logdet(regions_sigma[t, r])
        regions_precision[t, r] = inv
        regions_log_norm[t, r] = -0.5 * (3 * np.log(2 * np.pi) + logdet)

mask_xyz = np.array([{voxels}])

spatial_dists = _jit_get_spatial_dists(mask_xyz, regions_mu, regions_precision, regions_log_norm)
# The real np.sum(axis=1), exercising NumPy's actual pairwise_sum kernel at
# n_topics=100 -- this is the exact reduction output.rs must reproduce.
ref_p_topic_g_voxel = spatial_dists / np.sum(spatial_dists, axis=1)[:, None]
ref_p_topic_g_voxel = np.nan_to_num(ref_p_topic_g_voxel, 0)

got = np.load(d + "/p_topic_g_voxel.npy")
assert got.shape == ref_p_topic_g_voxel.shape, (got.shape, ref_p_topic_g_voxel.shape)
match = np.array_equal(got.view(np.uint64), ref_p_topic_g_voxel.view(np.uint64))
if not match:
    bad = np.flatnonzero(got.ravel().view(np.uint64) != ref_p_topic_g_voxel.ravel().view(np.uint64))
    i = bad[0]
    raise AssertionError(
        f"p_topic_g_voxel diverges from Python reference at flat index {{i}}: "
        f"rust={{got.ravel()[i]!r}} python={{ref_p_topic_g_voxel.ravel()[i]!r}}"
    )
print("OK")
"#,
        dir = dir.display(),
        voxels = voxel_literal.join(", "),
    );

    let out = Command::new("micromamba")
        .args(["run", "-n", "nimenv", "python", "-c", &script])
        .output()
        .expect("failed to run micromamba");
    assert!(
        out.status.success(),
        "wiring check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "OK");

    std::fs::remove_dir_all(&dir).ok();
}
