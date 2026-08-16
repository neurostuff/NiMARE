"""Generate golden fixtures pinning Python behavior for the Rust GCLDA port.

Run with:
    micromamba run -n nimenv python nimare/tests/generate_gclda_fixtures.py

Writes JSON fixtures into rust/gclda/tests/fixtures/. These pin the exact
numeric behavior the Rust implementation must reproduce bit-for-bit. Floats
are serialized as hex bit patterns so JSON round-tripping cannot lose
precision.
"""

import json
import os
import struct

import numpy as np

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "rust",
    "gclda",
    "tests",
    "fixtures",
)


def f64_bits(x):
    """Serialize a float64 as a hex bit pattern, losslessly."""
    return struct.pack("<d", float(x)).hex()


def write(name, obj):
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    path = os.path.join(FIXTURE_DIR, name)
    with open(path, "w") as fo:
        json.dump(obj, fo, indent=2)
    print(f"wrote {path}")


def gen_rng_random():
    """np.random.random() streams for several seeds."""
    cases = []
    for seed in (0, 1, 42, 12345, 2**31 - 1):
        np.random.seed(seed)
        draws = [f64_bits(np.random.random()) for _ in range(64)]
        cases.append({"seed": int(seed), "draws": draws})
    write("rng_random.json", cases)


def gen_rng_randint():
    """np.random.randint(bound, size=n) for bounds that do and do not straddle
    a power of two, exercising the masked-rejection path."""
    cases = []
    for seed in (1, 42):
        for bound in (2, 3, 7, 8, 64, 100, 1000, 1024, 65537):
            np.random.seed(seed)
            values = np.random.randint(bound, size=64).tolist()
            cases.append({"seed": int(seed), "bound": int(bound), "values": values})
    write("rng_randint.json", cases)


def gen_gaussian():
    """Closed-form 3x3 inverse/logdet and the Gaussian PDF, on fixed matrices."""
    from nimare.annotate.gclda import _inv3_logdet

    rng = np.random.default_rng(0)
    cases = []
    for _ in range(50):
        m = rng.normal(size=(3, 3)) * rng.uniform(1, 60)
        sigma = m @ m.T + 50.0 * np.eye(3) * rng.uniform(0.1, 3)
        inv, logdet = _inv3_logdet(sigma)
        log_norm = -0.5 * (3 * np.log(2 * np.pi) + logdet)
        mean = rng.normal(size=3) * 30.0
        points = rng.normal(size=(4, 3)) * 40.0
        pdfs = []
        for p in points:
            centered = p - mean
            quad = 0.0
            for i in range(3):
                inner = 0.0
                for j in range(3):
                    inner += inv[i, j] * (p[j] - mean[j])
                quad += centered[i] * inner
            pdfs.append(f64_bits(np.exp(log_norm - 0.5 * quad)))
        cases.append(
            {
                "sigma": [[f64_bits(v) for v in row] for row in sigma],
                "inv": [[f64_bits(v) for v in row] for row in inv],
                "logdet": f64_bits(logdet),
                "log_norm": f64_bits(log_norm),
                "mean": [f64_bits(v) for v in mean],
                "points": [[f64_bits(v) for v in p] for p in points],
                "pdfs": pdfs,
            }
        )
    write("gaussian.json", cases)


def gen_ingest():
    """Pin the constructor's index-determining behavior.

    Deliberately adversarial: document IDs are NOT in sorted order in the
    file, IDs differ between the two tables, one term is all-zero, and
    string sorting differs from numeric sorting ("10" < "9").
    """
    import pandas as pd

    from nimare.annotate.gclda import GCLDAModel
    from nimare.utils import get_template

    counts = pd.DataFrame(
        {
            "alpha": [2, 0, 1, 3],
            "beta": [0, 0, 0, 0],  # dropped: zero everywhere
            "gamma": [1, 4, 0, 0],
            "delta": [0, 2, 5, 1],
        },
        index=["9", "10", "2", "extra_count_only"],
    )
    coords = pd.DataFrame(
        {
            "id": ["2", "9", "9", "10", "10", "10", "coord_only"],
            "x": [10.0, -20.0, 30.0, -5.0, 15.0, -25.0, 0.0],
            "y": [-30.0, 40.0, -50.0, 12.0, -22.0, 32.0, 0.0],
            "z": [50.0, -60.0, 20.0, -18.0, 28.0, -38.0, 0.0],
        }
    )

    counts.to_csv(os.path.join(FIXTURE_DIR, "counts.tsv"), sep="\t", index_label="id")
    coords.to_csv(os.path.join(FIXTURE_DIR, "coordinates.tsv"), sep="\t", index=False)

    model = GCLDAModel(counts, coords, mask=get_template("mni152_2mm", mask="brain"), n_topics=3)
    write(
        "ingest.json",
        {
            "ids": list(model.ids),
            "vocabulary": list(model.vocabulary),
            "wtoken_doc_idx": model.data["wtoken_doc_idx"].tolist(),
            "wtoken_word_idx": model.data["wtoken_word_idx"].tolist(),
            "ptoken_doc_idx": model.data["ptoken_doc_idx"].tolist(),
            "ptoken_coords": [[f64_bits(v) for v in row] for row in model.data["ptoken_coords"]],
        },
    )


if __name__ == "__main__":
    gen_rng_random()
    gen_rng_randint()
    gen_gaussian()
    gen_ingest()
