//! Gibbs samplers for GCLDA's collapsed sampling scheme, plus the spatial
//! region parameter update.
//!
//! `peaks` and `words` each mutate [`crate::model::Model`] in place via a
//! sequential sweep over one token stream. None of those loops may be
//! parallelized: each token's sampling probabilities are computed from
//! counts that the immediately preceding token in the same sweep wrote.
//!
//! `regions` is different: it recomputes each topic's subregion Gaussians
//! from peak assignments that this update itself does not mutate, so its
//! per-topic parameter computation is order-independent and may be
//! parallelized (see `regions.rs` for what stays sequential and why).

pub mod peaks;
pub mod regions;
pub mod words;
