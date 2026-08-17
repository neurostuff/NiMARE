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
use std::path::PathBuf;
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

    let corpus = load_corpus(&args.counts, &args.coordinates)?;
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
    };

    let mut model = Model::new(corpus, mask, params)?;
    model.fit(args.n_iters, args.loglikely_freq)?;

    // Replay the recorded log-likelihood history to stderr in the same
    // format as Python's `_update`'s `LGR.info` line. `fit` (src/output.rs)
    // computes an iter==0 entry before the loop, exactly as Python's `fit`
    // does directly (not through `_update`) -- Python never logs that one,
    // so skip it here too.
    for &(iter, x, w, total) in &model.loglikelihood_history {
        if iter == 0 {
            continue;
        }
        eprintln!(
            "Iter {iter:04} Log-likely: x = {x:10.1}, w = {w:10.1}, tot = {total:10.1}"
        );
    }

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
