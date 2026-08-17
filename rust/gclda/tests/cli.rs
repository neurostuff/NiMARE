//! End-to-end test for the `gclda-train` binary.
//!
//! Invokes the actual compiled binary (via `CARGO_BIN_EXE_gclda-train`) on
//! the fixture TSVs and the repo's bundled MNI mask, the same combination
//! `tests/outputs.rs` already exercises at the library level. This test
//! checks only the CLI's own responsibilities -- argument parsing, wiring,
//! exit status, and stderr messaging -- not bit-exactness (that lives in the
//! Python-side regression harness added by a later task).

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};

mod common;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

fn mask_path() -> PathBuf {
    let mask_meta = common::load("mask_xyz.json");
    common::repo_path(mask_meta["path"].as_str().unwrap())
}

/// Every file `write_outputs` (src/output.rs) is documented to produce.
const EXPECTED_OUTPUT_FILES: &[&str] = &[
    "p_topic_g_voxel.npy",
    "p_voxel_g_topic.npy",
    "p_topic_g_word.npy",
    "p_word_g_topic.npy",
    "n_word_tokens_word_by_topic.npy",
    "n_peak_tokens_doc_by_topic.npy",
    "n_peak_tokens_region_by_topic.npy",
    "regions_mu.npy",
    "regions_sigma.npy",
    "wtoken_topic_idx.npy",
    "peak_topic_idx.npy",
    "peak_region_idx.npy",
    "loglikelihood.tsv",
    "vocabulary.txt",
    "model.json",
];

#[test]
fn cli_runs_end_to_end_and_writes_expected_outputs() {
    let dir =
        std::env::temp_dir().join(format!("gclda_cli_test_ok_{}", std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }

    let output = Command::new(env!("CARGO_BIN_EXE_gclda-train"))
        .args([
            "--counts",
            fixture("counts.tsv").to_str().unwrap(),
            "--coordinates",
            fixture("coordinates.tsv").to_str().unwrap(),
            "--mask",
            mask_path().to_str().unwrap(),
            "--out-dir",
            dir.to_str().unwrap(),
            "--n-topics",
            "3",
            "--n-regions",
            "2",
            "--symmetric",
            "true",
            "--seed-init",
            "1",
            "--n-iters",
            "3",
            "--loglikely-freq",
            "1",
            "--threads",
            "1",
        ])
        .output()
        .expect("failed to run gclda-train binary");

    assert!(
        output.status.success(),
        "gclda-train exited with {:?}\nstdout: {}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    for name in EXPECTED_OUTPUT_FILES {
        assert!(dir.join(name).exists(), "missing expected output file {name}");
    }

    // Progress lines must appear on stderr in the same format as Python's
    // `_update`'s `LGR.info` line: "Iter %04d Log-likely: x = %10.1f, w =
    // %10.1f, tot = %10.1f". iter=0 (the pre-loop initial log-likelihood) is
    // never printed by Python's `_update` -- it's computed by `fit` directly,
    // outside `_update` -- so only iters 1..=3 should appear here.
    let stderr = String::from_utf8_lossy(&output.stderr);
    for iter in 1..=3 {
        let needle = format!("Iter {iter:04} Log-likely: x = ");
        assert!(stderr.contains(&needle), "expected {needle:?} in stderr:\n{stderr}");
    }
    assert!(!stderr.contains("Iter 0000 Log-likely:"), "stderr:\n{stderr}");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn cli_rejects_symmetric_true_with_odd_n_regions() {
    let dir = std::env::temp_dir()
        .join(format!("gclda_cli_test_invalid_{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();

    let output = Command::new(env!("CARGO_BIN_EXE_gclda-train"))
        .args([
            "--counts",
            fixture("counts.tsv").to_str().unwrap(),
            "--coordinates",
            fixture("coordinates.tsv").to_str().unwrap(),
            "--mask",
            mask_path().to_str().unwrap(),
            "--out-dir",
            dir.to_str().unwrap(),
            "--n-topics",
            "3",
            "--n-regions",
            "3",
            "--symmetric",
            "true",
            "--n-iters",
            "1",
        ])
        .output()
        .expect("failed to run gclda-train binary");

    assert!(
        !output.status.success(),
        "expected a nonzero exit for --symmetric true --n-regions 3 (odd), got success"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    let lower = stderr.to_lowercase();
    assert!(
        lower.contains("symmetric") && lower.contains("even"),
        "expected a useful error message mentioning the symmetric/even-n_regions \
         constraint, got stderr:\n{stderr}"
    );
    // No backtrace/panic noise -- a clean error path, not a Rust panic.
    assert!(!stderr.contains("panicked"), "stderr:\n{stderr}");
    assert!(!dir.exists(), "out-dir should not be created when construction fails");
}

/// clap's rendered `--help` text is a cheap way to pin every default value
/// against the Python signature (`GCLDAModel.__init__`/`fit`) without
/// actually running a full (slow) default-sized job.
#[test]
fn cli_help_lists_python_matching_defaults() {
    let output = Command::new(env!("CARGO_BIN_EXE_gclda-train"))
        .arg("--help")
        .output()
        .expect("failed to run gclda-train --help");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    for expected_default in [
        "100",  // n-topics
        "2",    // n-regions
        "true", // symmetric
        "0.1",  // alpha
        "0.01", // beta
        "0.01", // gamma
        "1",    // delta (rendered as "1" by clap's f64 Display)
        "25",   // dobs
        "50",   // roi-size
        "5000", // n-iters
        "10",   // loglikely-freq
        "f64",  // output-dtype
    ] {
        assert!(
            stdout.contains(expected_default),
            "expected default {expected_default:?} to appear in --help output:\n{stdout}"
        );
    }
}

/// `--symmetric` must accept an explicit value (`true`/`false`) rather than
/// behaving as a presence flag -- later regression tasks invoke it as
/// `--symmetric false`, which a presence flag cannot express.
#[test]
fn cli_symmetric_accepts_explicit_false() {
    let dir =
        std::env::temp_dir().join(format!("gclda_cli_test_asym_{}", std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }

    let output = Command::new(env!("CARGO_BIN_EXE_gclda-train"))
        .args([
            "--counts",
            fixture("counts.tsv").to_str().unwrap(),
            "--coordinates",
            fixture("coordinates.tsv").to_str().unwrap(),
            "--mask",
            mask_path().to_str().unwrap(),
            "--out-dir",
            dir.to_str().unwrap(),
            "--n-topics",
            "3",
            "--n-regions",
            "3",
            "--symmetric",
            "false",
            "--n-iters",
            "1",
        ])
        .output()
        .expect("failed to run gclda-train binary");

    assert!(
        output.status.success(),
        "--symmetric false with odd n-regions=3 must be accepted (asymmetric has no \
         parity constraint)\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// Progress must reach stderr WHILE training runs, not only after `fit()`
/// returns -- a 5000-iteration production run can take hours, and output
/// that only appears on completion is useless for monitoring it, comparing
/// it against a live Python run, or noticing a stalled run.
///
/// This spawns the child directly (rather than using `.output()`, which
/// waits for exit before handing back anything) and reads its stderr line
/// by line as it is produced. The first time a progress line is read, it
/// polls `child.try_wait()`: if the child has NOT exited yet, that line
/// necessarily reached this test while training was still running --
/// concrete proof of live streaming, not a timing guess. `--n-iters 40` at
/// `--loglikely-freq 1` gives 39 more iterations (plus the whole output-
/// writing phase) of headroom between the first line and process exit, so
/// this is not a tight race.
#[test]
fn cli_progress_streams_during_the_run_not_only_after_it_finishes() {
    let dir = std::env::temp_dir()
        .join(format!("gclda_cli_test_stream_{}", std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }

    let mut child = Command::new(env!("CARGO_BIN_EXE_gclda-train"))
        .args([
            "--counts",
            fixture("counts.tsv").to_str().unwrap(),
            "--coordinates",
            fixture("coordinates.tsv").to_str().unwrap(),
            "--mask",
            mask_path().to_str().unwrap(),
            "--out-dir",
            dir.to_str().unwrap(),
            "--n-topics",
            "3",
            "--n-regions",
            "2",
            "--symmetric",
            "true",
            "--seed-init",
            "1",
            "--n-iters",
            "40",
            "--loglikely-freq",
            "1",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn gclda-train binary");

    let stderr = child.stderr.take().expect("child stderr was not piped");
    let mut reader = BufReader::new(stderr);
    let mut line = String::new();
    let mut n_progress_lines = 0usize;
    let mut observed_still_running = false;

    loop {
        line.clear();
        let n_bytes = reader.read_line(&mut line).expect("failed to read child stderr");
        if n_bytes == 0 {
            break; // EOF: the child closed its stderr.
        }
        if line.starts_with("Iter ") {
            n_progress_lines += 1;
            if n_progress_lines == 1 {
                if let Ok(None) = child.try_wait() {
                    observed_still_running = true;
                }
            }
        }
    }

    let status = child.wait().expect("failed to wait on child");
    assert!(status.success(), "gclda-train exited with {:?}", status.code());

    // Every recorded log-likelihood (iters 1..=40 at loglikely_freq=1) must
    // produce exactly one progress line, regardless of the timing race above.
    assert_eq!(
        n_progress_lines, 40,
        "expected one progress line per recorded iteration (true streaming is verified \
         by construction here too: the callback is invoked from inside fit()'s loop in \
         src/output.rs, at the same point Python's `_update` calls `LGR.info`)"
    );

    assert!(
        observed_still_running,
        "expected the child process to still be running when the first progress line \
         arrived, proving progress streams live rather than batching after fit() \
         returns"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// `GcldaError::Io`'s `Display` (src/lib.rs) is a bare `io error: <message>`
/// with no path -- on its own it can't tell the user whether `--counts`,
/// `--coordinates`, or `--mask` was the problem. The CLI's `check_readable`
/// preflight (src/bin/gclda-train.rs) must name both the failing input
/// ("counts") and its path before that bare message ever reaches the user.
#[test]
fn cli_missing_counts_file_names_the_path_in_the_error() {
    let dir = std::env::temp_dir()
        .join(format!("gclda_cli_test_missing_counts_{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();

    let missing_counts = fixture("does_not_exist_counts.tsv");
    assert!(!missing_counts.exists(), "test setup bug: fixture unexpectedly exists");

    let output = Command::new(env!("CARGO_BIN_EXE_gclda-train"))
        .args([
            "--counts",
            missing_counts.to_str().unwrap(),
            "--coordinates",
            fixture("coordinates.tsv").to_str().unwrap(),
            "--mask",
            mask_path().to_str().unwrap(),
            "--out-dir",
            dir.to_str().unwrap(),
            "--n-iters",
            "1",
        ])
        .output()
        .expect("failed to run gclda-train binary");

    assert!(
        !output.status.success(),
        "expected a nonzero exit for a missing --counts file"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("counts") && stderr.contains(missing_counts.to_str().unwrap()),
        "expected the error to name both the \"counts\" input and its path, got: {stderr}"
    );
    assert!(!stderr.contains("panicked"), "stderr:\n{stderr}");
    assert!(!dir.exists(), "out-dir should not be created when input loading fails");
}
