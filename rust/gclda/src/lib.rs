//! Generalized Correspondence LDA, bit-compatible with NiMARE's Python implementation.
//!
//! Bit-exactness is a hard requirement, not a nicety: it is the correctness
//! oracle for this port. Do not reorder floating-point operations, do not
//! replace division with reciprocal multiplication, and do not enable
//! fast-math anywhere in this crate.

pub mod gaussian;
pub mod io;
pub mod loglik;
pub mod model;
pub mod output;
pub mod pairwise_sum;
pub mod rng;
pub mod sampler;

#[derive(Debug)]
pub enum GcldaError {
    NonPositiveWeights,
    NotPositiveDefinite,
    Parse(String),
    Io(std::io::Error),
}

impl std::fmt::Display for GcldaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GcldaError::NonPositiveWeights => {
                write!(f, "Sampling weights must sum to a positive value.")
            }
            GcldaError::NotPositiveDefinite => {
                write!(f, "Region covariance must be positive definite.")
            }
            GcldaError::Parse(m) => write!(f, "parse error: {m}"),
            GcldaError::Io(e) => write!(f, "io error: {e}"),
        }
    }
}

impl std::error::Error for GcldaError {}

impl From<std::io::Error> for GcldaError {
    fn from(e: std::io::Error) -> Self {
        GcldaError::Io(e)
    }
}

impl From<serde_json::Error> for GcldaError {
    fn from(e: serde_json::Error) -> Self {
        GcldaError::Parse(format!("json error: {e}"))
    }
}
