//! MT19937, reproducing NumPy's legacy `RandomState` bit-for-bit.
//!
//! Verified facts this encodes:
//!   * scalar seeding uses `init_genrand` (Knuth multiplier 1812433253)
//!   * `random()` consumes two u32 draws: (a >> 5) * 2^26 + (b >> 6), over 2^53
//!   * `randint(bound)` uses 32-bit masked rejection sampling
//!   * numba's in-njit RNG produces the identical stream, so this single
//!     implementation covers both model initialization and all sampling

use crate::GcldaError;

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

pub struct Mt19937 {
    state: [u32; N],
    index: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; N];
        state[0] = seed;
        for i in 1..N {
            let prev = state[i - 1];
            state[i] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { state, index: N }
    }

    /// Re-seed in place, matching `np.random.seed(seed)`.
    pub fn reseed(&mut self, seed: u32) {
        *self = Mt19937::new(seed);
    }

    fn generate(&mut self) {
        for i in 0..N {
            let y = (self.state[i] & UPPER_MASK) | (self.state[(i + 1) % N] & LOWER_MASK);
            let mut next = self.state[(i + M) % N] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= MATRIX_A;
            }
            self.state[i] = next;
        }
        self.index = 0;
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.generate();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// Equivalent to `np.random.random()`.
    #[inline]
    pub fn random(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67108864.0 + b) / 9007199254740992.0
    }

    /// Equivalent to `np.random.randint(bound)`, i.e. uniform on [0, bound).
    ///
    /// NumPy's legacy path is 32-bit masked rejection sampling. This was
    /// verified against bounds 2, 3, 7, 64, 100, and 1000.
    pub fn randint(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0);
        let rng_range = bound - 1;
        if rng_range == 0 {
            return 0;
        }
        let mut mask = rng_range;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        loop {
            let value = (self.next_u32() as u64) & mask;
            if value <= rng_range {
                return value;
            }
        }
    }

    /// Port of `_sample_from_unnormalized` in nimare/annotate/gclda.py.
    ///
    /// The accumulation order here is load-bearing: it must match the Python
    /// loop exactly, or sampled indices can differ.
    pub fn sample_from_unnormalized(&mut self, weights: &[f64]) -> Result<usize, GcldaError> {
        let mut total = 0.0f64;
        for &w in weights {
            total += w;
        }
        if total <= 0.0 {
            return Err(GcldaError::NonPositiveWeights);
        }
        let threshold = self.random() * total;
        let mut cumulative = 0.0f64;
        for (i, &w) in weights.iter().enumerate() {
            cumulative += w;
            if threshold < cumulative {
                return Ok(i);
            }
        }
        Ok(weights.len() - 1)
    }
}
