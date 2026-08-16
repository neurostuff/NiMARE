use gclda::rng::Mt19937;

fn load(name: &str) -> serde_json::Value {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing fixture {path}: {e}. Run generate_gclda_fixtures.py"));
    serde_json::from_str(&text).unwrap()
}

fn bits_to_f64(hex: &str) -> f64 {
    let raw = (0..8)
        .map(|i| u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap())
        .collect::<Vec<u8>>();
    f64::from_le_bytes(raw.try_into().unwrap())
}

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
