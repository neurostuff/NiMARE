//! 3x3 Gaussian helpers.
//!
//! Peak coordinates are always (x, y, z), so every covariance is 3x3 and the
//! closed-form adjugate inverse applies. Operation order here MUST match
//! `_inv3_logdet` in nimare/annotate/gclda.py exactly.

use crate::GcldaError;

#[inline]
fn log_2pi() -> f64 {
    (2.0 * std::f64::consts::PI).ln() // matches np.log(2 * np.pi) bit-for-bit
}

pub fn inv3_logdet(sigma: &[[f64; 3]; 3]) -> Result<([[f64; 3]; 3], f64), GcldaError> {
    let (a00, a01, a02) = (sigma[0][0], sigma[0][1], sigma[0][2]);
    let (a10, a11, a12) = (sigma[1][0], sigma[1][1], sigma[1][2]);
    let (a20, a21, a22) = (sigma[2][0], sigma[2][1], sigma[2][2]);

    let c00 = a11 * a22 - a12 * a21;
    let c01 = a02 * a21 - a01 * a22;
    let c02 = a01 * a12 - a02 * a11;
    let c10 = a12 * a20 - a10 * a22;
    let c11 = a00 * a22 - a02 * a20;
    let c12 = a02 * a10 - a00 * a12;
    let c20 = a10 * a21 - a11 * a20;
    let c21 = a01 * a20 - a00 * a21;
    let c22 = a00 * a11 - a01 * a10;

    let det = a00 * c00 + a01 * c10 + a02 * c20;
    if !(det > 0.0) {
        return Err(GcldaError::NotPositiveDefinite);
    }

    let inv = [
        [c00 / det, c01 / det, c02 / det],
        [c10 / det, c11 / det, c12 / det],
        [c20 / det, c21 / det, c22 / det],
    ];
    Ok((inv, det.ln()))
}

#[inline]
pub fn log_norm(logdet: f64) -> f64 {
    -0.5 * (3.0 * log_2pi() + logdet)
}

/// Evaluate a Gaussian PDF. Mirrors the loop structure of `_jit_spatial_pdf`;
/// the nested accumulation order is load-bearing.
#[inline]
pub fn pdf(
    point: &[f64; 3],
    mean: &[f64; 3],
    precision: &[[f64; 3]; 3],
    log_norm: f64,
) -> f64 {
    let mut quad = 0.0f64;
    for i in 0..3 {
        let centered_i = point[i] - mean[i];
        let mut inner = 0.0f64;
        for j in 0..3 {
            inner += precision[i][j] * (point[j] - mean[j]);
        }
        quad += centered_i * inner;
    }
    (log_norm - 0.5 * quad).exp()
}
