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
use std::time::Instant;

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

/// Number of iterations for the streaming test below, chosen empirically
/// (see the fix report for task 14) so that `fit()`'s own sampling work
/// dominates wall time -- roughly 1.7-2.0s of `fit()` against roughly
/// 0.3s of fixed per-process overhead (mostly `write_outputs` against the
/// full ~228,483-voxel MNI mask, which is otherwise-constant regardless of
/// `n_iters`). That gap is what gives the timing assertion below a wide,
/// unambiguous margin rather than a coin flip.
const N_STREAM_ITERS: usize = 20_000;

/// Progress must reach stderr WHILE training runs, not only after `fit()`
/// returns -- a 5000-iteration production run can take hours, and output
/// that only appears on completion is useless for monitoring it, comparing
/// it against a live Python run, or noticing a stalled run.
///
/// ## Why this is a timing assertion, not a `try_wait()` check
///
/// An earlier version of this test spawned the child and, on the first
/// `"Iter "` line read from its stderr, called `child.try_wait()`,
/// asserting it returned `Ok(None)` (child still alive) as "proof" of
/// streaming. That assertion is worthless: `run()` in
/// `src/bin/gclda-train.rs` calls `write_outputs` (real Gaussian-PDF work
/// against every voxel in the mask) AFTER `fit()` returns, and that alone
/// takes measurable wall time (~0.3s here). So even under the ORIGINAL
/// batched design this test was meant to catch -- print all of
/// `loglikelihood_history` in a tight loop right after `fit()` returns,
/// THEN call `write_outputs` -- the first stderr line would still arrive
/// while the child was alive and about to enter `write_outputs`.
/// `try_wait()` would still see `Ok(None)`. That version of this test would
/// have passed under the exact regression it was written to catch, which
/// defeats the point of having it. (Checking that `--out-dir` doesn't exist
/// yet at that point doesn't help either, for the same reason: progress
/// precedes `write_outputs` under BOTH designs.)
///
/// The only thing that actually discriminates batched-after-`fit()` from
/// genuinely-streamed-during-`fit()` is INTER-LINE TIMING: under batched
/// printing, all N lines arrive within microseconds of each other (a tight
/// Rust loop over an in-memory `Vec`); under real streaming, consecutive
/// lines are separated by real per-iteration sampling work. So this test
/// measures the wall-clock span from the first progress line to the last,
/// and requires it to be a substantial fraction of the process's total
/// wall time. The 25% threshold is deliberately generous (observed ratios
/// in practice are ~80-90%, see the fix report) -- the goal is only to
/// distinguish "~0" from "seconds," not to pin down a precise number, so
/// this stays robust on slower/loaded CI machines without becoming flaky.
///
/// This was verified to have teeth: temporarily reverting `run()` to
/// collect all progress after `fit()` returns (the original design) makes
/// this test fail with a near-zero ratio. See the fix report for the exact
/// numbers.
#[test]
fn cli_progress_streams_during_the_run_not_only_after_it_finishes() {
    let dir = std::env::temp_dir()
        .join(format!("gclda_cli_test_stream_{}", std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }

    let overall_start = Instant::now();
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
            &N_STREAM_ITERS.to_string(),
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
    let mut first_line_at: Option<Instant> = None;
    let mut last_line_at: Option<Instant> = None;

    loop {
        line.clear();
        let n_bytes = reader.read_line(&mut line).expect("failed to read child stderr");
        if n_bytes == 0 {
            break; // EOF: the child closed its stderr.
        }
        if line.starts_with("Iter ") {
            let now = Instant::now();
            first_line_at.get_or_insert(now);
            last_line_at = Some(now);
            n_progress_lines += 1;
        }
    }

    let status = child.wait().expect("failed to wait on child");
    let overall_elapsed = overall_start.elapsed();
    assert!(status.success(), "gclda-train exited with {:?}", status.code());

    // One progress line per recorded log-likelihood (true by construction --
    // the callback lives inside fit()'s loop -- so this alone can't catch a
    // regression to batching, but it's still a real correctness check).
    assert_eq!(n_progress_lines, N_STREAM_ITERS, "expected one progress line per iteration");

    let first_line_at = first_line_at.expect("no progress lines were read at all");
    let last_line_at = last_line_at.expect("no progress lines were read at all");
    let span = last_line_at.duration_since(first_line_at);
    let ratio = span.as_secs_f64() / overall_elapsed.as_secs_f64();

    assert!(
        ratio >= 0.25,
        "progress lines spanned only {:.1}% of the process's total wall time \
         ({:?} of {:?}) -- expected at least 25%. A ratio this low means \
         progress is arriving in a tight batch rather than streaming \
         throughout the run (see this test's doc comment for why an \
         inter-line-timing check, not a try_wait() check, is what actually \
         proves this)",
        ratio * 100.0,
        span,
        overall_elapsed,
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
