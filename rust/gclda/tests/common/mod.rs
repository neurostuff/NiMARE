use std::path::PathBuf;

/// Resolve a repo-relative fixture path against the repository root.
/// Fixtures MUST NOT store absolute paths — a test that hard-codes the
/// generating machine's paths passes only on that machine.
///
/// `common` is compiled fresh into each integration-test binary, and not
/// every test binary calls this yet (more will, e.g. task 8), so allow it to
/// go unused in the others rather than warn.
#[allow(dead_code)]
pub fn repo_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

pub fn load(name: &str) -> serde_json::Value {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing fixture {path}: {e}. Run generate_gclda_fixtures.py"));
    serde_json::from_str(&text).unwrap()
}

pub fn bits_to_f64(hex: &str) -> f64 {
    let raw: Vec<u8> = (0..8)
        .map(|i| u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap())
        .collect();
    f64::from_le_bytes(raw.try_into().unwrap())
}
