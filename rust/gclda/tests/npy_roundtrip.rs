use gclda::io::npy;
use std::process::Command;

/// Write a .npy from Rust, then have NumPy read it back and report what it saw.
fn numpy_describe(path: &str) -> String {
    let script = format!(
        "import numpy as np; a = np.load(r'{path}'); \
         print(a.dtype, a.shape, float(a.sum()), float(a.ravel()[0]), float(a.ravel()[-1]))"
    );
    let out = Command::new("micromamba")
        .args(["run", "-n", "nimenv", "python", "-c", &script])
        .output()
        .expect("failed to run micromamba");
    assert!(
        out.status.success(),
        "numpy failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

#[test]
fn f64_matrix_roundtrips_through_numpy() {
    let dir = std::env::temp_dir().join("gclda_npy_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.npy");

    let data: Vec<f64> = (0..12).map(|i| i as f64 * 0.5).collect();
    npy::write_f64(&path, &[3, 4], &data).unwrap();

    let desc = numpy_describe(path.to_str().unwrap());
    assert_eq!(desc, "float64 (3, 4) 33.0 0.0 5.5");
}

#[test]
fn i64_matrix_roundtrips_through_numpy() {
    let dir = std::env::temp_dir().join("gclda_npy_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("i.npy");

    let data: Vec<i64> = (0..6).collect();
    npy::write_i64(&path, &[2, 3], &data).unwrap();

    let desc = numpy_describe(path.to_str().unwrap());
    assert_eq!(desc, "int64 (2, 3) 15.0 0.0 5.0");
}

#[test]
fn streamed_rows_match_a_single_shot_write() {
    let dir = std::env::temp_dir().join("gclda_npy_test");
    std::fs::create_dir_all(&dir).unwrap();

    let data: Vec<f64> = (0..20).map(|i| (i as f64).sin()).collect();
    let one_shot = dir.join("one.npy");
    npy::write_f64(&one_shot, &[4, 5], &data).unwrap();

    let streamed = dir.join("stream.npy");
    let mut w = npy::NpyWriter::create(&streamed, &[4, 5], npy::Dtype::F64).unwrap();
    for row in data.chunks(5) {
        w.write_row(row).unwrap();
    }
    w.finish().unwrap();

    assert_eq!(
        std::fs::read(&one_shot).unwrap(),
        std::fs::read(&streamed).unwrap()
    );
}
