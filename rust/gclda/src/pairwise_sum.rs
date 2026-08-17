//! NumPy-compatible pairwise summation.
//!
//! `np.sum` (and other ufunc reductions) over a contiguous (unit-stride)
//! float axis does NOT sum left-to-right: NumPy's `pairwise_sum` C kernel
//! splits the range into blocks of up to 128 elements, accumulates each
//! block via an 8-way-unrolled loop, and combines the eight partial sums in
//! a fixed tree shape; blocks larger than 128 elements are split
//! recursively in half (rounded down to a multiple of 8) and the two halves
//! are added together. This is faster and more accurate than naive
//! sequential accumulation, but is a genuinely different floating-point
//! result once the reduction length reaches 8 -- below that NumPy itself
//! falls back to a plain sequential loop, which is why a naive
//! implementation can look correct on small fixtures and still be wrong at
//! production scale. Any Rust code that needs to match a `np.sum(...)`
//! reduction along a contiguous axis bit-for-bit must use [`numpy_sum`]
//! instead of a plain accumulation loop.
//!
//! This does NOT apply to every reduction in this crate:
//! - Integer (e.g. `i64`) sums are exact in any summation order (no
//!   rounding to reassociate), so plain accumulation is correct for those
//!   regardless of axis or length.
//! - `np.sum` reductions along a NON-contiguous (strided) axis of a
//!   C-contiguous array -- e.g. `axis=0` of a `(V, T)` array, where the
//!   reduced axis has stride `T` -- do not go through this pairwise kernel
//!   at all. NumPy's generic strided-reduction loop instead accumulates
//!   sequentially over the reduced axis (vectorized across the kept,
//!   contiguous axis), which is exactly what a plain per-column
//!   accumulation loop already computes. Applying [`numpy_sum`] there would
//!   not match `np.sum` -- it would just not obviously mismatch on a small
//!   fixture, which is the exact failure mode this function exists to
//!   avoid elsewhere. Use a plain loop for axis=0-style reductions.
//!
//! Verified (see `tests/pairwise_sum.rs`) to reproduce `np.sum` bit-for-bit
//! against NumPy for reduction lengths spanning all three regimes below,
//! from length 3 up to 228483 (the real MNI152 2mm brain mask's voxel
//! count).

/// NumPy's block size for the unrolled-then-recursive regime of
/// `pairwise_sum`. Reductions of at most this many elements (and at least
/// 8) are summed by the 8-way-unrolled loop below; longer reductions are
/// split recursively.
const PW_BLOCKSIZE: usize = 128;

/// Reproduce `np.sum` over a contiguous `f64` slice bit-for-bit. See the
/// module doc for exactly when this applies (contiguous, floating-point
/// reductions) and when a plain loop is correct instead (integer sums,
/// or strided/`axis=0`-style reductions).
pub fn numpy_sum(a: &[f64]) -> f64 {
    let n = a.len();

    if n < 8 {
        let mut r = 0.0f64;
        for &v in a {
            r += v;
        }
        return r;
    }

    if n <= PW_BLOCKSIZE {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8usize;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        // This grouping -- ((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7)) -- is
        // load-bearing: a left-to-right fold over the eight accumulators
        // gives different rounding and would not match NumPy.
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }

    // Divide by two, rounding the left half down to a multiple of 8 so
    // both halves' unrolled loops (recursively) stay aligned the same way
    // NumPy's do.
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    numpy_sum(&a[..n2]) + numpy_sum(&a[n2..])
}
