//! `gclda-train`: thin CLI wiring the GCLDA library together.
//!
//! Mirrors `GCLDAModel.__init__`/`fit` in `nimare/annotate/gclda.py`: read
//! counts + coordinates + a mask from disk, build a [`Model`], run
//! [`Model::fit`], then write every output file with [`write_outputs`].
//! Nothing in this file reimplements any of that logic -- it only parses
//! arguments, translates them into the library's types, and reports errors
//! and progress to the user.

use clap::Parser;
use gclda::io::nifti::load_mask_xyz;
use gclda::io::npy::Dtype;
use gclda::io::tsv::load_corpus;
use gclda::model::{Model, Params};
use gclda::output::write_outputs;
use gclda::GcldaError;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Storage dtype for the two large V x T probability matrices. See
/// `write_outputs`'s doc comment for exactly which files this affects.
#[derive(Clone, Copy, clap::ValueEnum)]
enum OutputDtype {
    F64,
    F32,
}

impl From<OutputDtype> for Dtype {
    fn from(d: OutputDtype) -> Dtype {
        match d {
            OutputDtype::F64 => Dtype::F64,
            OutputDtype::F32 => Dtype::F32,
        }
    }
}

/// Train a GCLDA topic model, mirroring `GCLDAModel(...).fit(...)` in
/// `nimare/annotate/gclda.py`.
#[derive(Parser)]
#[command(name = "gclda-train", about = "Train a GCLDA topic model")]
struct Args {
    /// Path to the word-count TSV (rows = documents, columns = terms).
    #[arg(long)]
    counts: PathBuf,

    /// Path to the peak-coordinates TSV (columns: id, x, y, z).
    #[arg(long)]
    coordinates: PathBuf,

    /// Path to the brain mask NIfTI image (.nii or .nii.gz).
    #[arg(long)]
    mask: PathBuf,

    /// Directory to write model outputs into (created if missing).
    #[arg(long)]
    out_dir: PathBuf,

    /// Number of topics.
    #[arg(long, default_value_t = 100)]
    n_topics: usize,

    /// Number of subregions per topic.
    #[arg(long, default_value_t = 2)]
    n_regions: usize,

    /// Whether subregions are constrained to bilaterally symmetric pairs.
    /// Takes an explicit value (`--symmetric true` / `--symmetric false`),
    /// not a presence flag.
    #[arg(long, action = clap::ArgAction::Set, value_parser = clap::value_parser!(bool), default_value_t = true)]
    symmetric: bool,

    /// Prior count on topics per document.
    #[arg(long, default_value_t = 0.1)]
    alpha: f64,

    /// Prior count on word-types per topic.
    #[arg(long, default_value_t = 0.01)]
    beta: f64,

    /// Prior count on topics per word (via co-occurring peaks).
    #[arg(long, default_value_t = 0.01)]
    gamma: f64,

    /// Prior count on subregions per topic.
    #[arg(long, default_value_t = 1.0)]
    delta: f64,

    /// Estimated number of observations per topic used to regularize the
    /// spatial covariance estimate.
    #[arg(long, default_value_t = 25.0)]
    dobs: f64,

    /// Prior "region of interest" size (mm), used in covariance
    /// regularization.
    #[arg(long, default_value_t = 50.0)]
    roi_size: f64,

    /// Seed for the initial random assignment.
    #[arg(long, default_value_t = 1)]
    seed_init: u32,

    /// Peaks per parallel PDF-evaluation block. Larger blocks expose more
    /// parallelism; buffer cost is
    /// `peak_block_size * n_topics * n_regions * 8` bytes.
    #[arg(long, default_value_t = 8192)]
    peak_block_size: usize,

    /// Total number of training iterations to run.
    #[arg(long, default_value_t = 5000)]
    n_iters: usize,

    /// Compute (and log) log-likelihood every this-many iterations.
    #[arg(long, default_value_t = 10)]
    loglikely_freq: usize,

    /// Storage dtype for the two large V x T probability matrices.
    #[arg(long, value_enum, default_value = "f64")]
    output_dtype: OutputDtype,

    /// Size of the rayon global thread pool. 0 uses rayon's default (one
    /// thread per logical CPU).
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// If set, write full sampler state (`iter_{n:05d}/<name>.npy`) after
    /// every iteration into this directory, for the per-iteration equality
    /// harness against the Python port. Not used in normal training runs.
    #[arg(long)]
    dump_state_dir: Option<PathBuf>,

    /// Diagnostic: before training, time one serial pass of Gaussian PDF
    /// evaluation over every peak and print the result, then continue.
    /// Used to measure what fraction of the peak-sampling phase is PDF
    /// evaluation. Not part of normal training.
    #[arg(long, default_value_t = false)]
    profile_pdf: bool,
}

/// Preflight-check that `path` can be opened, and if not, produce an error
/// naming both `label` and `path` -- `load_corpus`/`load_mask_xyz` return
/// `GcldaError::Io`, whose `Display` (`src/lib.rs`) is a bare `io error:
/// <message>` with no path at all, which leaves the user unable to tell
/// whether `--counts`, `--coordinates`, or `--mask` was the culprit. This
/// check runs before handing `path` to the library, so it does not require
/// restructuring `GcldaError` to carry a path.
///
/// This is a diagnostic aid, not a substitute for the library's own error
/// handling: a path that passes this check but then genuinely fails inside
/// the library (a race, or a parse error unrelated to file access) still
/// surfaces via `main`'s bare `GcldaError` fallback, just without a path
/// name.
fn check_readable(label: &str, path: &Path) -> Result<(), GcldaError> {
    std::fs::File::open(path)
        .map(|_| ())
        .map_err(|e| GcldaError::Parse(format!("reading {label} file \"{}\": {e}", path.display())))
}

fn run(args: Args) -> Result<(), GcldaError> {
    if args.threads != 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .map_err(|e| {
                GcldaError::Parse(format!("failed to configure {} worker threads: {e}", args.threads))
            })?;
    }

    check_readable("counts", &args.counts)?;
    check_readable("coordinates", &args.coordinates)?;
    let corpus = load_corpus(&args.counts, &args.coordinates)?;

    check_readable("mask", &args.mask)?;
    let mask = load_mask_xyz(&args.mask)?;

    let params = Params {
        n_topics: args.n_topics,
        n_regions: args.n_regions,
        symmetric: args.symmetric,
        alpha: args.alpha,
        beta: args.beta,
        gamma: args.gamma,
        delta: args.delta,
        dobs: args.dobs,
        roi_size: args.roi_size,
        seed_init: args.seed_init,
        peak_block_size: args.peak_block_size,
    };

    let mut model = Model::new(corpus, mask, params)?;

    if args.profile_pdf {
        let (seconds, _n_evaluated) = model.time_serial_pdf_pass();
        println!("profile_pdf: serial_pdf_pass_seconds={seconds:.6}");
    }

    // The callback is invoked from inside `fit`'s loop (src/output.rs),
    // right where Python's `_update` calls `LGR.info`, so progress reaches
    // stderr WHILE training runs rather than only after `fit` returns --
    // essential for a 5000-iteration production run that can take hours.
    // Format matches `_update`'s `LGR.info` f-string exactly (located by
    // function name, not line number): `Iter {iter:04d} Log-likely: x =
    // {x:10.1f}, w = {w:10.1f}, tot = {total:10.1f}`. `fit`'s iter==0 entry
    // (computed directly by Python's `fit`, not through `_update`) never
    // invokes this callback, matching Python never logging it either.
    model.fit(
        args.n_iters,
        args.loglikely_freq,
        args.dump_state_dir.as_deref(),
        |iter, ll| {
            eprintln!(
                "Iter {iter:04} Log-likely: x = {x:10.1}, w = {w:10.1}, tot = {total:10.1}",
                x = ll.x,
                w = ll.w,
                total = ll.total,
            );
        },
    )?;

    write_outputs(&model, &args.out_dir, args.output_dtype.into())?;

    Ok(())
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("gclda-train: error: {e}");
            ExitCode::FAILURE
        }
    }
}
