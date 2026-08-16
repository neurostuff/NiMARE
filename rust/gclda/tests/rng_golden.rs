use gclda::rng::Mt19937;

mod common;
use common::{bits_to_f64, load};

#[test]
fn random_stream_matches_numpy_bit_for_bit() {
    for case in load("rng_random.json").as_array().unwrap() {
        let seed = case["seed"].as_u64().unwrap() as u32;
        let mut rng = Mt19937::new(seed);
        for (i, expected) in case["draws"].as_array().unwrap().iter().enumerate() {
            let want = bits_to_f64(expected.as_str().unwrap());
            let got = rng.random();
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "seed {seed} draw {i}: got {got:?} want {want:?}"
            );
        }
    }
}

#[test]
fn randint_matches_numpy_legacy_masked_rejection() {
    for case in load("rng_randint.json").as_array().unwrap() {
        let seed = case["seed"].as_u64().unwrap() as u32;
        let bound = case["bound"].as_u64().unwrap();
        let mut rng = Mt19937::new(seed);
        for (i, expected) in case["values"].as_array().unwrap().iter().enumerate() {
            let want = expected.as_u64().unwrap();
            let got = rng.randint(bound);
            assert_eq!(got, want, "seed {seed} bound {bound} draw {i}");
        }
    }
}
