//! Verifies `gclda::pairwise_sum::numpy_sum` reproduces `np.sum` bit-for-bit
//! for contiguous f64 reductions.
//!
//! This exists because a plain left-to-right accumulation loop matches
//! `np.sum` only for reduction lengths below 8 -- exactly the range every
//! small fixture in this crate (`n_topics` in {3, 4, 5}) exercises, and
//! therefore exactly the range in which a naive-but-wrong implementation of
//! `output.rs`'s row-sum normalization would still pass every other test in
//! the suite. This test spans all three of NumPy's `pairwise_sum` regimes
//! (naive below 8, 8-way-unrolled block from 8 to 128, recursive
//! divide-and-conquer above 128) plus the exact scale this port targets in
//! production: `n_topics=100` and `n_voxels=228483` (the real MNI152 2mm
//! brain mask's voxel count, used as a stand-in for "large contiguous
//! reduction" even though in `output.rs` the actual per-row reduction
//! length is `n_topics`, not `n_voxels` -- the algorithm doesn't care what
//! the numbers represent, only how many there are).
//!
//! Random input data is generated via `gclda::rng::Mt19937::random()`,
//! which `tests/rng_golden.rs` already established reproduces
//! `np.random.seed(seed); np.random.random()` bit-for-bit -- so seeding
//! both sides with the same seed and drawing `n` values from each gives
//! identical input arrays without shipping a data file.

use gclda::pairwise_sum::numpy_sum;
use gclda::rng::Mt19937;
use std::process::Command;

fn python_np_sum(seed: u32, n: usize) -> f64 {
    let script = format!(
        "import numpy as np; np.random.seed({seed}); a = np.random.random({n}); \
         print(a.sum().hex())"
    );
    let out = Command::new("micromamba")
        .args(["run", "-n", "nimenv", "python", "-c", &script])
        .output()
        .expect("failed to run micromamba");
    assert!(out.status.success(), "numpy failed: {}", String::from_utf8_lossy(&out.stderr));
    let hex = String::from_utf8_lossy(&out.stdout).trim().to_string();
    parse_python_float_hex(&hex)
}

/// Parse the output of Python `float.hex()`, e.g. `0x1.8p+1`, into an f64.
/// `float.hex()` round-trips exactly (it's a direct rendering of the IEEE
/// 754 bit pattern), so this avoids any decimal-string rounding on the way
/// back into Rust.
fn parse_python_float_hex(s: &str) -> f64 {
    let neg = s.starts_with('-');
    let s = s.trim_start_matches('-');
    let s = s.strip_prefix("0x").expect("expected 0x prefix");
    let (mantissa_str, exp_str) = s.split_once('p').expect("expected p exponent separator");
    let exp: i32 = exp_str.parse().unwrap();
    let (int_part, frac_part) = mantissa_str.split_once('.').unwrap_or((mantissa_str, ""));
    let int_val = i64::from_str_radix(int_part, 16).unwrap() as f64;
    let mut frac_val = 0.0f64;
    let mut scale = 1.0f64 / 16.0;
    for c in frac_part.chars() {
        frac_val += (c.to_digit(16).unwrap() as f64) * scale;
        scale /= 16.0;
    }
    let mut v = (int_val + frac_val) * 2f64.powi(exp);
    if neg {
        v = -v;
    }
    v
}

#[test]
fn parse_python_float_hex_round_trips_known_values() {
    // 1.5 == 0x1.8p+0
    assert_eq!(parse_python_float_hex("0x1.8p+0"), 1.5);
    // -2.0 == -0x1.0p+1
    assert_eq!(parse_python_float_hex("-0x1.0p+1"), -2.0);
    assert_eq!(parse_python_float_hex("0x0.0p+0"), 0.0);
}

#[test]
fn numpy_sum_matches_np_sum_across_regimes() {
    // (seed, n): n spans NumPy's three pairwise_sum regimes -- naive
    // (< 8), 8-way-unrolled block (8..=128), and recursive
    // divide-and-conquer (> 128) -- plus n=100 (this port's production
    // n_topics) and n=228483 (the real brain mask's voxel count, as a
    // stress case for the recursive regime).
    let cases: &[(u32, usize)] = &[
        (1, 1),
        (2, 3),
        (3, 7),
        (4, 8),
        (5, 9),
        (6, 33),
        (7, 64),
        (8, 100),
        (9, 127),
        (10, 128),
        (11, 129),
        (12, 256),
        (13, 1000),
        (14, 4096),
        (15, 228_483),
    ];

    for &(seed, n) in cases {
        let mut rng = Mt19937::new(seed);
        let data: Vec<f64> = (0..n).map(|_| rng.random()).collect();

        let got = numpy_sum(&data);
        let want = python_np_sum(seed, n);

        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "n={n} seed={seed}: got {got:?} (bits {:#018x}) want {want:?} (bits {:#018x})",
            got.to_bits(),
            want.to_bits()
        );
    }
}

/// A direct, minimal check that the pairwise algorithm's result actually
/// differs from naive summation once `n >= 8` -- if this ever stopped being
/// true (e.g. because the test data changed to all-equal values, which sum
/// associatively without rounding differences), the test above would still
/// pass even for a naive implementation, silently losing its ability to
/// catch the bug it exists to catch.
#[test]
fn pairwise_and_naive_summation_genuinely_differ_at_production_scale() {
    let mut rng = Mt19937::new(100);
    let data: Vec<f64> = (0..100).map(|_| rng.random()).collect();

    let pairwise = numpy_sum(&data);
    let naive = data.iter().fold(0.0f64, |acc, &v| acc + v);

    assert_ne!(
        pairwise.to_bits(),
        naive.to_bits(),
        "test data must exercise a real rounding difference between pairwise and naive \
         summation, or the bit-exact test above cannot distinguish a correct implementation \
         from a naive regression"
    );
}
