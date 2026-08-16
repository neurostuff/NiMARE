//! Spatial subregion Gaussian parameter update.
//!
//! Port of `GCLDAModel._update_regions` in `nimare/annotate/gclda.py`, via
//! its three helpers: `_jit_accumulate_region_stats` (sufficient statistics),
//! `_compute_covariance_from_stats` (regularized covariance), and
//! `_cache_region_pdf_params` (precision + log-normalizer caching). The
//! Python source is the specification: this file preserves its arithmetic
//! operation order exactly, including the two-branch split between symmetric
//! and asymmetric subregion layouts.
//!
//! Peak coordinates are always (x, y, z) (see `gaussian.rs`), so all sums,
//! cross-products, and covariances here are fixed at 3 dimensions.

use crate::gaussian::{inv3_logdet, log_norm};
use crate::model::Model;
use crate::GcldaError;
use rayon::prelude::*;

/// Sample covariance from sufficient statistics:
/// `(cross - outer(sum, sum) / n_obs) / (n_obs - 1)`.
///
/// Mirrors `_compute_covariance_from_stats` element-by-element: the outer
/// product is divided by `n_obs` and subtracted from `cross` first, and only
/// the resulting matrix is divided by `n_obs - 1`. Do not reassociate this
/// into e.g. dividing `cross` and the outer product separately -- that
/// changes rounding.
fn compute_covariance_from_stats(
    sum: &[f64; 3],
    cross: &[[f64; 3]; 3],
    n_obs: f64,
) -> [[f64; 3]; 3] {
    let mut out = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let centered = cross[i][j] - (sum[i] * sum[j]) / n_obs;
            out[i][j] = centered / (n_obs - 1.0);
        }
    }
    out
}

/// `d_c * c_hat + (1 - d_c) * default_roi`, elementwise, matching the
/// regularization step shared by both branches of `_update_regions`.
fn regularize(d_c: f64, c_hat: &[[f64; 3]; 3], default_roi: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = (d_c * c_hat[i][j]) + ((1.0 - d_c) * default_roi[i][j]);
        }
    }
    out
}

/// Mean and regularized covariance for one subregion, prior to caching
/// precision/log-norm.
struct RegionEstimate {
    mu: [f64; 3],
    sigma: [[f64; 3]; 3],
}

/// Compute the mean and regularized covariance for one asymmetric subregion.
/// Port of the `else` branch (non-symmetric case) of `_update_regions`.
fn estimate_asymmetric(
    sum_vector: &[f64; 3],
    cross_matrix: &[[f64; 3]; 3],
    n_obs: i64,
    default_roi: &[[f64; 3]; 3],
    dobs: f64,
) -> RegionEstimate {
    let mu = if n_obs == 0 {
        [0.0f64; 3]
    } else {
        let n = n_obs as f64;
        [sum_vector[0] / n, sum_vector[1] / n, sum_vector[2] / n]
    };

    let c_hat = if n_obs <= 1 {
        *default_roi
    } else {
        compute_covariance_from_stats(sum_vector, cross_matrix, n_obs as f64)
    };

    let d_c = n_obs as f64 / (n_obs as f64 + dobs);
    let sigma = regularize(d_c, &c_hat, default_roi);

    RegionEstimate { mu, sigma }
}

/// Compute the paired means and regularized covariances for one symmetric
/// subregion pair. Port of the `if self.params["symmetric"]` branch of
/// `_update_regions`. Means are constrained symmetric about the origin along
/// dimension 0 (x); dimensions 1-2 share an unconstrained weighted mean
/// between the pair.
#[allow(clippy::too_many_arguments)]
fn estimate_symmetric_pair(
    sum1: &[f64; 3],
    cross1: &[[f64; 3]; 3],
    n_obs1: i64,
    sum2: &[f64; 3],
    cross2: &[[f64; 3]; 3],
    n_obs2: i64,
    default_roi: &[[f64; 3]; 3],
    dobs: f64,
) -> (RegionEstimate, RegionEstimate) {
    let total_obs = n_obs1 + n_obs2;

    // Independent per-subregion centroids, used only to build the
    // dimension-0 weighted mean below. Note this recomputes sum/n_obs and
    // then multiplies back by n_obs rather than using sum[0] directly --
    // that round-trip through division is NOT a no-op in floating point,
    // and Python takes it, so this must too for bit-exactness.
    let reg1_center = if n_obs1 == 0 {
        [0.0f64; 3]
    } else {
        let n = n_obs1 as f64;
        [sum1[0] / n, sum1[1] / n, sum1[2] / n]
    };
    let reg2_center = if n_obs2 == 0 {
        [0.0f64; 3]
    } else {
        let n = n_obs2 as f64;
        [sum2[0] / n, sum2[1] / n, sum2[2] / n]
    };

    let (weighted_dim1, weighted_other1, weighted_other2) = if total_obs == 0 {
        (0.0f64, 0.0f64, 0.0f64)
    } else {
        let t = total_obs as f64;
        let dim1 = ((-reg1_center[0] * n_obs1 as f64) + (reg2_center[0] * n_obs2 as f64)) / t;
        let other1 = (sum1[1] + sum2[1]) / t;
        let other2 = (sum1[2] + sum2[2]) / t;
        (dim1, other1, other2)
    };

    let mu1 = [-weighted_dim1, weighted_other1, weighted_other2];
    let mu2 = [weighted_dim1, weighted_other1, weighted_other2];

    let c_hat1 = if n_obs1 <= 1 {
        *default_roi
    } else {
        compute_covariance_from_stats(sum1, cross1, n_obs1 as f64)
    };
    let c_hat2 = if n_obs2 <= 1 {
        *default_roi
    } else {
        compute_covariance_from_stats(sum2, cross2, n_obs2 as f64)
    };

    let d_c_1 = n_obs1 as f64 / (n_obs1 as f64 + dobs);
    let d_c_2 = n_obs2 as f64 / (n_obs2 as f64 + dobs);
    let sigma1 = regularize(d_c_1, &c_hat1, default_roi);
    let sigma2 = regularize(d_c_2, &c_hat2, default_roi);

    (
        RegionEstimate { mu: mu1, sigma: sigma1 },
        RegionEstimate { mu: mu2, sigma: sigma2 },
    )
}

/// Estimate every (topic, subregion) mean and regularized covariance for one
/// topic, writing into `mu_row`/`sigma_row` (each length `n_regions`,
/// region-indexed). Shared by the sequential and rayon-parallel drivers in
/// [`Model::update_regions`] so both take the identical per-topic code path.
#[allow(clippy::too_many_arguments)]
fn estimate_topic(
    i_topic: usize,
    n_topics: usize,
    n_regions: usize,
    symmetric: bool,
    region_counts: &[i64],
    region_sums: &[[f64; 3]],
    region_cross: &[[[f64; 3]; 3]],
    default_roi: &[[f64; 3]; 3],
    dobs: f64,
    mu_row: &mut [[f64; 3]],
    sigma_row: &mut [[[f64; 3]; 3]],
) {
    if symmetric {
        let n_pairs = n_regions / 2;
        for j_pair in 0..n_pairs {
            let region1 = j_pair * 2;
            let region2 = region1 + 1;
            let idx1 = Model::at(region1, i_topic, n_topics);
            let idx2 = Model::at(region2, i_topic, n_topics);
            let n_obs1 = region_counts[idx1];
            let n_obs2 = region_counts[idx2];

            let (est1, est2) = estimate_symmetric_pair(
                &region_sums[idx1],
                &region_cross[idx1],
                n_obs1,
                &region_sums[idx2],
                &region_cross[idx2],
                n_obs2,
                default_roi,
                dobs,
            );

            mu_row[region1] = est1.mu;
            mu_row[region2] = est2.mu;
            sigma_row[region1] = est1.sigma;
            sigma_row[region2] = est2.sigma;
        }
    } else {
        for j_region in 0..n_regions {
            let idx = Model::at(j_region, i_topic, n_topics);
            let n_obs = region_counts[idx];

            let est =
                estimate_asymmetric(&region_sums[idx], &region_cross[idx], n_obs, default_roi, dobs);

            mu_row[j_region] = est.mu;
            sigma_row[j_region] = est.sigma;
        }
    }
}

impl Model {
    /// Accumulate per (region, topic) sums and cross-products of peak
    /// coordinates. Port of `_jit_accumulate_region_stats`. Both output
    /// `Vec`s are flat, region-major/topic-minor (`Model::at(region, topic,
    /// n_topics)`), matching `n_peak_tokens_region_by_topic`'s layout.
    ///
    /// Kept strictly sequential: this is a single pass of floating-point
    /// summation, and floating-point addition is not associative, so a
    /// naive parallel reduction over peaks could reorder additions into a
    /// different (still "correct", but not bit-identical) result. Nothing in
    /// this task's golden fixtures is large enough to justify the added risk
    /// of a chunked-reduction version, so this phase stays sequential.
    fn accumulate_region_stats(&self) -> (Vec<[f64; 3]>, Vec<[[f64; 3]; 3]>) {
        let n_topics = self.params.n_topics;
        let n_regions = self.params.n_regions;
        let mut region_sums = vec![[0.0f64; 3]; n_regions * n_topics];
        let mut region_cross = vec![[[0.0f64; 3]; 3]; n_regions * n_topics];

        let n_ptokens = self.corpus.ptoken_coords.len();
        for i in 0..n_ptokens {
            let topic = self.peak_topic_idx[i] as usize;
            let region = self.peak_region_idx[i] as usize;
            let idx = Model::at(region, topic, n_topics);
            let coords = self.corpus.ptoken_coords[i];

            for d in 0..3 {
                let val = coords[d];
                region_sums[idx][d] += val;
                for e in 0..3 {
                    region_cross[idx][d][e] += val * coords[e];
                }
            }
        }

        (region_sums, region_cross)
    }

    /// Cache precision and log-norm for the Gaussian at flat index `idx`
    /// (topic-major, `Model::at(topic, region, n_regions)`) from
    /// `regions_sigma[idx]`, which must already be populated. Port of
    /// `_cache_region_pdf_params`.
    fn cache_region_pdf_params(&mut self, idx: usize) -> Result<(), GcldaError> {
        let sigma = self.regions_sigma[idx];
        let (inv, logdet) = inv3_logdet(&sigma)?;
        self.regions_precision[idx] = inv;
        self.regions_log_norm[idx] = log_norm(logdet);
        Ok(())
    }

    /// Update `regions_mu`, `regions_sigma`, `regions_precision`, and
    /// `regions_log_norm` for every (topic, subregion) pair from the current
    /// peak assignments. Port of `_update_regions`.
    ///
    /// Two phases: accumulate sufficient statistics over all peaks
    /// (sequential, see [`Model::accumulate_region_stats`]), then compute
    /// per-topic parameters. The second phase is parallelized over topics
    /// with rayon: each topic reads only the shared, read-only statistics
    /// computed in phase one and writes only its own `n_regions`-wide slice
    /// of `regions_mu`/`regions_sigma` (disjoint per-topic chunks, no
    /// cross-topic floating-point reduction involved), so this is
    /// order-independent. It was verified bit-exact against the golden
    /// fixtures both with sequential accumulation and this rayon-parallel
    /// parameter phase (see task report for the sequential-only baseline
    /// run that preceded parallelizing this).
    pub fn update_regions(&mut self) -> Result<(), GcldaError> {
        let n_topics = self.params.n_topics;
        let n_regions = self.params.n_regions;
        let dobs = self.params.dobs;
        let roi_size = self.params.roi_size;
        let symmetric = self.params.symmetric;
        let default_roi = [
            [roi_size, 0.0, 0.0],
            [0.0, roi_size, 0.0],
            [0.0, 0.0, roi_size],
        ];

        let (region_sums, region_cross) = self.accumulate_region_stats();
        // Snapshot counts before taking mutable borrows of the output
        // arrays below; this is a small (n_regions * n_topics) copy.
        let region_counts = self.n_peak_tokens_region_by_topic.clone();

        // Compute every topic's means/covariances directly into
        // `self.regions_mu`/`self.regions_sigma`, in parallel over topics.
        // Writes are disjoint by construction (`par_chunks_mut(n_regions)`),
        // so there is no reduction to combine afterward -- each topic owns
        // its own slice of the output.
        compute_topic_means_and_covariances(
            self,
            n_topics,
            n_regions,
            symmetric,
            &region_counts,
            &region_sums,
            &region_cross,
            &default_roi,
            dobs,
        );

        // Cache precision/log-norm sequentially from the now-populated
        // regions_sigma; this is cheap (one 3x3 inverse per subregion) and
        // keeps error propagation (`?`) simple.
        for i_topic in 0..n_topics {
            for j_region in 0..n_regions {
                let idx = Model::at(i_topic, j_region, n_regions);
                self.cache_region_pdf_params(idx)?;
            }
        }

        Ok(())
    }
}

/// Fill `model.regions_mu`/`model.regions_sigma` for every topic, computing
/// topics in parallel via rayon over disjoint `n_regions`-wide chunks.
#[allow(clippy::too_many_arguments)]
fn compute_topic_means_and_covariances(
    model: &mut Model,
    n_topics: usize,
    n_regions: usize,
    symmetric: bool,
    region_counts: &[i64],
    region_sums: &[[f64; 3]],
    region_cross: &[[[f64; 3]; 3]],
    default_roi: &[[f64; 3]; 3],
    dobs: f64,
) {
    model
        .regions_mu
        .par_chunks_mut(n_regions)
        .zip(model.regions_sigma.par_chunks_mut(n_regions))
        .enumerate()
        .for_each(|(i_topic, (mu_row, sigma_row))| {
            estimate_topic(
                i_topic,
                n_topics,
                n_regions,
                symmetric,
                region_counts,
                region_sums,
                region_cross,
                default_roi,
                dobs,
                mu_row,
                sigma_row,
            );
        });
}
