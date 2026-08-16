use gclda::gaussian::{inv3_logdet, log_norm, pdf};

mod common;
use common::{bits_to_f64, load};

#[test]
fn inverse_logdet_and_pdf_match_python_bit_for_bit() {
    for (c, case) in load("gaussian.json").as_array().unwrap().iter().enumerate() {
        let sigma = mat3(&case["sigma"]);
        let (inv, logdet) = inv3_logdet(&sigma).expect("positive definite");

        let want_inv = mat3(&case["inv"]);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(inv[i][j].to_bits(), want_inv[i][j].to_bits(), "case {c} inv[{i}][{j}]");
            }
        }
        let want_logdet = bits_to_f64(case["logdet"].as_str().unwrap());
        assert_eq!(logdet.to_bits(), want_logdet.to_bits(), "case {c} logdet");

        let ln = log_norm(logdet);
        assert_eq!(
            ln.to_bits(),
            bits_to_f64(case["log_norm"].as_str().unwrap()).to_bits(),
            "case {c} log_norm"
        );

        let mean = vec3(&case["mean"]);
        for (p, point_json) in case["points"].as_array().unwrap().iter().enumerate() {
            let point = vec3(point_json);
            let got = pdf(&point, &mean, &inv, ln);
            let want = bits_to_f64(case["pdfs"].as_array().unwrap()[p].as_str().unwrap());
            assert_eq!(got.to_bits(), want.to_bits(), "case {c} pdf {p}");
        }
    }
}

#[test]
fn singular_matrix_is_rejected() {
    let singular = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
    assert!(inv3_logdet(&singular).is_err());
}

fn vec3(v: &serde_json::Value) -> [f64; 3] {
    let a = v.as_array().unwrap();
    [
        bits_to_f64(a[0].as_str().unwrap()),
        bits_to_f64(a[1].as_str().unwrap()),
        bits_to_f64(a[2].as_str().unwrap()),
    ]
}

fn mat3(v: &serde_json::Value) -> [[f64; 3]; 3] {
    let a = v.as_array().unwrap();
    [vec3(&a[0]), vec3(&a[1]), vec3(&a[2])]
}
