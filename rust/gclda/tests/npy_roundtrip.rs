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

/// `write_f32_from_f64` had zero test coverage before this task, flagged by
/// an earlier review -- and this task (output.rs's two large V x T
/// matrices) is the first thing in the crate that actually uses the f32
/// path in production. Check dtype, shape, and that values match the f64
/// originals cast to f32 (not left as f64, and not silently truncated to
/// integers or similar).
#[test]
fn f32_matrix_roundtrips_through_numpy_as_f32() {
    let dir = std::env::temp_dir().join("gclda_npy_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m32.npy");

    // Values chosen so the f64 -> f32 cast is lossy (not representable
    // exactly in f32), so a test that forgot the cast (e.g. wrote raw f64
    // bytes under an f32 header) would be caught by the value comparison
    // below, not just the dtype/shape checks.
    let data: Vec<f64> = vec![0.1, 1.0 / 3.0, -2.5, 1e10, 123456.789, 0.0, -0.0, 42.0];
    npy::write_f32_from_f64(&path, &[2, 4], &data).unwrap();

    let desc = numpy_describe(path.to_str().unwrap());
    assert!(desc.starts_with("float32 (2, 4)"), "unexpected dtype/shape: {desc}");

    let script = format!(
        "import numpy as np; a = np.load(r'{}'); \
         print(' '.join(repr(float(v)) for v in a.ravel()))",
        path.to_str().unwrap()
    );
    let out = std::process::Command::new("micromamba")
        .args(["run", "-n", "nimenv", "python", "-c", &script])
        .output()
        .expect("failed to run micromamba");
    assert!(out.status.success(), "numpy failed: {}", String::from_utf8_lossy(&out.stderr));
    let got: Vec<f64> = String::from_utf8_lossy(&out.stdout)
        .trim()
        .split(' ')
        .map(|s| s.parse().unwrap())
        .collect();

    let want: Vec<f64> = data.iter().map(|&v| (v as f32) as f64).collect();
    assert_eq!(got, want, "NumPy's f32 values must match the f64 originals cast to f32");
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
