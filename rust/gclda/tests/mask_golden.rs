use gclda::io::nifti::load_mask_xyz;
use std::path::Path;

mod common;
use common::{bits_to_f64, load};

#[test]
fn mask_xyz_matches_nibabel() {
    let expected = load("mask_xyz.json");
    let path = expected["path"].as_str().unwrap();
    let info = load_mask_xyz(Path::new(path)).unwrap();

    let want_shape: Vec<usize> = expected["shape"]
        .as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    assert_eq!(info.shape.to_vec(), want_shape);

    for i in 0..4 {
        for j in 0..4 {
            let want = bits_to_f64(
                expected["affine"].as_array().unwrap()[i].as_array().unwrap()[j].as_str().unwrap(),
            );
            assert_eq!(info.affine[i][j].to_bits(), want.to_bits(), "affine[{i}][{j}]");
        }
    }

    assert_eq!(info.xyz.len(), expected["n_voxels"].as_u64().unwrap() as usize);

    // Column sums catch any ordering or off-by-one error across all voxels.
    let mut sums = [0.0f64; 3];
    for row in &info.xyz {
        for j in 0..3 {
            sums[j] += row[j];
        }
    }
    for j in 0..3 {
        let want = bits_to_f64(expected["sum_bits"].as_array().unwrap()[j].as_str().unwrap());
        assert!(
            (sums[j] - want).abs() <= want.abs() * 1e-12,
            "column sum {j}: got {} want {want}", sums[j]
        );
    }

    // Sampled rows verify exact ordering, not just aggregate agreement.
    let idx = expected["sample_indices"].as_array().unwrap();
    let xyz = expected["sample_xyz"].as_array().unwrap();
    for (s, i) in idx.iter().enumerate() {
        let i = i.as_u64().unwrap() as usize;
        for j in 0..3 {
            let want = bits_to_f64(xyz[s].as_array().unwrap()[j].as_str().unwrap());
            assert_eq!(info.xyz[i][j].to_bits(), want.to_bits(), "xyz[{i}][{j}]");
        }
    }
}
