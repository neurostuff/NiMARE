use gclda::io::nifti::load_mask_xyz;
use std::path::Path;
use std::process::Command;

mod common;
use common::{bits_to_f64, load, repo_path};

#[test]
fn mask_xyz_matches_nibabel() {
    let expected = load("mask_xyz.json");
    let path = repo_path(expected["path"].as_str().unwrap());
    let info = load_mask_xyz(&path).unwrap();

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

/// Ask nibabel to write an uncompressed copy of `src_path` to `dst_path`.
///
/// The bundled mask carries an AFNI header extension, so its real data offset
/// is 448 (not 0) whether it's gzipped or not — `to_filename` preserves the
/// extension and the offset. This test therefore does NOT exercise the
/// `vox_offset == 0` -> 352 fallback (see
/// `io::nifti::tests::vox_offset_zero_reads_data_from_352_not_348` for that);
/// what it does check is that gzip-decompress-then-parse and
/// read-the-plain-file-directly agree bit-for-bit on the `vox_offset != 0`
/// path, which is a real and separate risk (e.g. an off-by-N bug in the gzip
/// branch of `read_all_bytes`).
fn write_uncompressed_copy(src_path: &str, dst_path: &Path) {
    let script = format!(
        "import nibabel as nib; nib.load(r'{src_path}').to_filename(r'{}')",
        dst_path.display()
    );
    let out = Command::new("micromamba")
        .args(["run", "-n", "nimenv", "python", "-c", &script])
        .output()
        .expect("failed to run micromamba");
    assert!(
        out.status.success(),
        "nibabel failed to write uncompressed copy: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn uncompressed_nii_matches_gzipped_nii_gz() {
    let expected = load("mask_xyz.json");
    let gz_path = repo_path(expected["path"].as_str().unwrap());

    let dir = std::env::temp_dir().join("gclda_nifti_test");
    std::fs::create_dir_all(&dir).unwrap();
    let nii_path = dir.join("mask_uncompressed.nii");
    write_uncompressed_copy(gz_path.to_str().unwrap(), &nii_path);

    let gz_info = load_mask_xyz(&gz_path).unwrap();
    let nii_info = load_mask_xyz(&nii_path).unwrap();

    assert_eq!(nii_info.shape, gz_info.shape);
    for i in 0..4 {
        for j in 0..4 {
            assert_eq!(
                nii_info.affine[i][j].to_bits(),
                gz_info.affine[i][j].to_bits(),
                "affine[{i}][{j}]"
            );
        }
    }
    assert_eq!(nii_info.xyz.len(), gz_info.xyz.len());
    for (i, (a, b)) in nii_info.xyz.iter().zip(gz_info.xyz.iter()).enumerate() {
        for j in 0..3 {
            assert_eq!(
                a[j].to_bits(),
                b[j].to_bits(),
                "xyz[{i}][{j}] differs between .nii and .nii.gz readers"
            );
        }
    }
}
