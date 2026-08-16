//! Gibbs samplers for GCLDA's collapsed sampling scheme.
//!
//! Each sampler here mutates [`crate::model::Model`] in place via a
//! sequential sweep over one token stream. None of these loops may be
//! parallelized: each token's sampling probabilities are computed from
//! counts that the immediately preceding token in the same sweep wrote.

pub mod peaks;
pub mod words;
