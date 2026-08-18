"""Benchmark driver comparing the Python and Rust GCLDA trainers.

Correctness of the Rust port is established elsewhere (``nimare/tests/test_gclda_rust.py``
proves bit-identical sampler state and bit-identical probability matrices). This script's
only job is measurement: for a given corpus scale, it trains the same corpus with both
implementations, **verifies the four output probability matrices are still bit-identical**,
and only then reports wall-clock, per-phase, and peak-RSS numbers. If the matrices differ,
it refuses to emit timings and instead reports which matrix diverged and at what index --
a benchmark that credits a run which computed the wrong answer is worse than no benchmark.

Both implementations are run as fresh subprocesses (Python via ``sys.executable -c``, Rust
via the ``gclda-train`` binary), each wrapped with ``/usr/bin/time -v`` when available so
that peak RSS is measured cleanly per run rather than as a cumulative high-water mark. This
also isolates Python interpreter/numba-import startup cost into ``process_wall_clock_seconds``
(reported separately from the in-model ``phase_times``), and lets a dedicated warm-up run
absorb numba's one-time JIT compilation cost (cached to disk via ``@njit(cache=True)``, so it
is normally paid only once per environment, not once per subprocess) before any timed repeat.

Two comparability caveats (see module-level ``CAVEATS`` and the emitted JSON's ``"caveats"``
key) are surfaced rather than silently absorbed into a favorable-looking number:

1. Python's per-iteration ``LGR.debug``/``LGR.info`` calls use f-strings that are evaluated
   eagerly regardless of log level, so real overhead lands inside Python's ``total`` phase
   time. Rust has no equivalent per-iteration logging inside its timed region. Consequently
   ``total - sum(four phases)`` is not comparable *between* implementations, only *within*
   each one.
2. Rust's output writer recomputes ``spatial_dists`` in two passes (rather than caching it)
   when writing the final probability matrices, trading roughly 2x Gaussian evaluation for a
   bounded memory footprint. That cost lands once, at the end of training (outside the four
   timed phases, inside ``process_wall_clock_seconds``), not per-iteration -- but it is a real
   time/memory tradeoff and belongs in the record.

Usage
-----
    micromamba run -n nimenv python benchmarks/bench_gclda_rust.py \\
        --scale tiny --n-iters 20 --n-topics 20 --out /tmp/gclda_bench_tiny.json
"""

import argparse
import json
import os
import platform
import re
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
sys.path.insert(0, str(BENCH_DIR))

from gclda_synthetic import make_synthetic_corpus  # noqa: E402

from nimare.annotate.gclda_rs import DEFAULT_TRAIN_PARAMS, export_gclda_tsvs  # noqa: E402
from nimare.utils import get_resource_path  # noqa: E402

DEFAULT_MASK = os.path.join(get_resource_path(), "templates", "MNI152_2x2x2_brainmask.nii.gz")
DEFAULT_BINARY = REPO_ROOT / "rust" / "gclda" / "target" / "release" / "gclda-train"

# Synthetic corpus presets for the "tiny" and "small" scales. "neurosynth" is handled
# separately (real, cached data) in load_neurosynth_corpus().
SYNTHETIC_SCALES = {
    "tiny": {"n_docs": 30, "n_terms": 60, "n_peaks": 200},
    "small": {"n_docs": 300, "n_terms": 250, "n_peaks": 3000},
}

PHASE_KEYS = ("word_sampling", "peak_sampling", "region_update", "loglikelihood", "total")
# Both implementations write the four probability matrices under these exact filenames
# (Rust natively -- rust/gclda/src/output.rs; Python because the child script below saves
# to match), so the same name list drives both loading and the equality check.
P_MATRIX_FILES = (
    "p_topic_g_voxel.npy",
    "p_voxel_g_topic.npy",
    "p_topic_g_word.npy",
    "p_word_g_topic.npy",
)

CAVEATS = [
    "total - sum(four phases) is NOT comparable between implementations, only within each: "
    "Python's per-iteration LGR.debug/LGR.info calls use f-strings evaluated eagerly "
    "regardless of log level, so that overhead lands inside Python's phase_times['total']; "
    "Rust has no equivalent per-iteration logging inside its timed region.",
    "Rust's output writer recomputes spatial_dists in two passes rather than caching it, "
    "trading roughly 2x Gaussian evaluation for a bounded memory footprint. This cost lands "
    "once at the end of training (inside process_wall_clock_seconds, outside the four timed "
    "phases), not per-iteration, but it is a real time/memory tradeoff.",
    "python 'process_wall_clock_seconds' is dominated by CPython interpreter startup plus "
    "importing nimare.annotate.gclda (which pulls in numba/nibabel/nilearn/sklearn) -- "
    "measured in isolation, that import alone costs several seconds, versus tens to hundreds "
    "of milliseconds for fit_wall_clock_seconds (the model.fit() call itself) at small "
    "scales. This is a real, paid-every-invocation cost of running the Python tool as a fresh "
    "subprocess, but it swamps 'time to train' at small scale and should not be read as "
    "training throughput -- use phase_times/fit_wall_clock_seconds for that. Rust pays an "
    "analogous but much smaller binary-startup cost, not separately broken out here because "
    "gclda-train does not report it. fit_wall_clock_seconds also exceeds phase_total_seconds "
    "by roughly the cost of get_probability_distributions() (computed once, after the "
    "iteration loop, over all V mask voxels) -- Python's analog of caveat 2's Rust output-"
    "writing cost.",
]

# Executed by a fresh `sys.executable -c` subprocess. Reads a JSON args file (path given as
# argv[1]), trains a GCLDAModel, and prints one line of JSON to stdout: per-phase times (the
# same accounting as GCLDAModel.phase_times_) plus the wall-clock time of the fit() call
# alone (construction, e.g. mask loading and token expansion, is charged separately). Always
# also saves the four probability matrices under out_dir, using the same filenames Rust
# writes, so the driver's equality check can compare Python and Rust with one code path.
CHILD_SCRIPT = r"""
import json
import sys
import time

import numpy as np
import pandas as pd

from nimare.annotate.gclda import GCLDAModel

with open(sys.argv[1], encoding="utf-8") as fo:
    args = json.load(fo)

counts = pd.read_csv(args["counts_path"], sep="\t", index_col="id")
# float_precision="round_trip" is required, not cosmetic: pandas' default C
# float parser is not correctly rounded and disagrees with Rust's
# `str::parse::<f64>` by 1 ULP on ~2.5% of real Neurosynth coordinates, which
# propagates through the region statistics into the voxel probability matrices
# and breaks the equality check for reasons unrelated to the port.
coords = pd.read_csv(
    args["coordinates_path"], sep="\t", float_precision="round_trip"
)

t0 = time.perf_counter()
model = GCLDAModel(
    counts,
    coords,
    mask=args["mask_path"],
    n_topics=args["n_topics"],
    n_regions=args["n_regions"],
    symmetric=args["symmetric"],
    alpha=args["alpha"],
    beta=args["beta"],
    gamma=args["gamma"],
    delta=args["delta"],
    dobs=args["dobs"],
    roi_size=args["roi_size"],
    seed_init=args["seed_init"],
)
construct_seconds = time.perf_counter() - t0

t0 = time.perf_counter()
model.fit(n_iters=args["n_iters"], loglikely_freq=args["loglikely_freq"])
fit_wall_clock_seconds = time.perf_counter() - t0

import os as _os

_os.makedirs(args["out_dir"], exist_ok=True)
np.save(_os.path.join(args["out_dir"], "p_topic_g_voxel.npy"), model.p_topic_g_voxel_)
np.save(_os.path.join(args["out_dir"], "p_voxel_g_topic.npy"), model.p_voxel_g_topic_)
np.save(_os.path.join(args["out_dir"], "p_topic_g_word.npy"), model.p_topic_g_word_)
np.save(_os.path.join(args["out_dir"], "p_word_g_topic.npy"), model.p_word_g_topic_)

result = {
    "phase_times": model.phase_times_,
    "construct_seconds": construct_seconds,
    "fit_wall_clock_seconds": fit_wall_clock_seconds,
}
print(json.dumps(result))
"""


def _read_cpu_model():
    """Best-effort CPU model string; falls back to platform.processor() off Linux."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fo:
            for line in fo:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def _read_ram_gb():
    """Best-effort total RAM in GiB; None if /proc/meminfo is unavailable (non-Linux)."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fo:
            for line in fo:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / 1024 / 1024, 2)
    except OSError:
        pass
    return None


def _rustc_version():
    try:
        completed = subprocess.run(
            ["rustc", "--version"], capture_output=True, text=True, timeout=10
        )
        return completed.stdout.strip() if completed.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def collect_environment_info(binary_path, threads):
    """Record the hardware/software context a benchmark report cannot be trusted without."""
    import numba

    binary_path = str(binary_path)
    return {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "numba_version": numba.__version__,
        "cpu_model": _read_cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "ram_total_gib": _read_ram_gb(),
        "rustc_version": _rustc_version(),
        "gclda_binary_path": binary_path,
        # A proxy, not a guarantee: the standard cargo layout puts optimized builds under
        # target/release/ and unoptimized debug builds under target/debug/. This inspects the
        # path only -- it cannot detect a release binary someone manually relocated or renamed.
        "gclda_binary_looks_like_release_build": "target/release" in binary_path.replace(
            "\\", "/"
        ),
        "requested_threads": threads,
        "effective_threads_hint": (
            threads if threads and threads > 0 else os.cpu_count()
        ),
        "threads_hint_note": (
            "threads=0 means 'rayon picks one thread per logical CPU'; "
            "effective_threads_hint is os.cpu_count() in that case, not a value read back "
            "from the Rust process, since gclda-train does not report it."
        ),
    }


_USR_BIN_TIME = "/usr/bin/time"
_HAVE_USR_BIN_TIME = os.path.isfile(_USR_BIN_TIME)


def _run_timed_subprocess(cmd, report_path):
    """Run ``cmd``, returning (returncode, stdout, stderr, wall_seconds, peak_rss_kb, method).

    Prefers ``/usr/bin/time -v`` (Linux-only), which reports a clean peak RSS scoped to
    exactly this one child process. Falls back to ``resource.getrusage(RUSAGE_CHILDREN)``,
    whose ``ru_maxrss`` is a *cumulative* high-water mark across every child this driver
    process has spawned so far (POSIX semantics) -- a later, smaller-peak run cannot lower it
    back down, so a value obtained this way is only a ceiling on that run's true peak, not an
    exact per-run measurement. Callers must propagate ``method`` into the report so this
    ambiguity is never silently dropped.
    """
    if _HAVE_USR_BIN_TIME:
        full_cmd = [_USR_BIN_TIME, "-v", "-o", str(report_path), "--"] + list(cmd)
        t0 = time.perf_counter()
        completed = subprocess.run(full_cmd, capture_output=True, text=True)
        wall = time.perf_counter() - t0
        peak_kb = None
        if os.path.isfile(report_path):
            with open(report_path, encoding="utf-8") as fo:
                report_text = fo.read()
            match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", report_text)
            if match:
                peak_kb = int(match.group(1))
        return (
            completed.returncode,
            completed.stdout,
            completed.stderr,
            wall,
            peak_kb,
            "usr_bin_time_v_per_run",
        )

    ru_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    t0 = time.perf_counter()
    completed = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    ru_after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    peak_kb = max(ru_before, ru_after)
    return (
        completed.returncode,
        completed.stdout,
        completed.stderr,
        wall,
        peak_kb,
        "rusage_children_cumulative_ceiling",
    )


def run_python_once(counts_path, coords_path, mask_path, out_dir, params, report_path):
    """Train once with the Python implementation in a fresh subprocess.

    Returns a dict with ``phase_times`` (matching ``GCLDAModel.phase_times_``),
    ``construct_seconds``, ``fit_wall_clock_seconds`` (both measured inside the child, around
    just the named step), plus ``process_wall_clock_seconds``, ``peak_rss_kb``, and
    ``rss_method`` (measured by the parent around the whole subprocess).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args_path = out_dir / "_args.json"
    payload = dict(params)
    payload.update(
        counts_path=str(counts_path),
        coordinates_path=str(coords_path),
        mask_path=str(mask_path),
        out_dir=str(out_dir),
    )
    with open(args_path, "w", encoding="utf-8") as fo:
        json.dump(payload, fo)

    cmd = [sys.executable, "-c", CHILD_SCRIPT, str(args_path)]
    report_path = out_dir / "time_report.txt" if report_path is None else report_path
    rc, out, err, wall, peak_kb, rss_method = _run_timed_subprocess(cmd, report_path)
    if rc != 0:
        raise RuntimeError(f"Python GCLDA child failed (exit {rc}):\n{err}")

    stdout_lines = [line for line in out.splitlines() if line.strip()]
    if not stdout_lines:
        raise RuntimeError(f"Python GCLDA child produced no stdout. stderr:\n{err}")
    result = json.loads(stdout_lines[-1])
    result.update(
        process_wall_clock_seconds=wall,
        peak_rss_kb=peak_kb,
        rss_method=rss_method,
    )
    return result


def run_rust_once(binary, counts_path, coords_path, mask_path, out_dir, params, report_path):
    """Train once with the Rust ``gclda-train`` binary. See run_python_once for return shape."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(binary),
        "--counts", str(counts_path),
        "--coordinates", str(coords_path),
        "--mask", str(mask_path),
        "--out-dir", str(out_dir),
        "--n-topics", str(params["n_topics"]),
        "--n-regions", str(params["n_regions"]),
        "--symmetric", "true" if params["symmetric"] else "false",
        "--alpha", str(params["alpha"]),
        "--beta", str(params["beta"]),
        "--gamma", str(params["gamma"]),
        "--delta", str(params["delta"]),
        "--dobs", str(params["dobs"]),
        "--roi-size", str(params["roi_size"]),
        "--seed-init", str(params["seed_init"]),
        "--n-iters", str(params["n_iters"]),
        "--loglikely-freq", str(params["loglikely_freq"]),
        "--output-dtype", str(params.get("output_dtype", "f64")),
        "--threads", str(params.get("threads", 0)),
        "--peak-block-size", str(params.get("peak_block_size", 8192)),
    ]
    report_path = out_dir / "time_report.txt" if report_path is None else report_path
    rc, out, err, wall, peak_kb, rss_method = _run_timed_subprocess(cmd, report_path)
    if rc != 0:
        raise RuntimeError(f"gclda-train failed (exit {rc}):\n{err}")

    with open(out_dir / "model.json", encoding="utf-8") as fo:
        meta = json.load(fo)
    return {
        "phase_times": meta["phase_times"],
        "process_wall_clock_seconds": wall,
        "peak_rss_kb": peak_kb,
        "rss_method": rss_method,
    }


def compare_p_matrices(py_dir, rs_dir):
    """Bitwise-compare the four probability matrices. Return None if identical, else a message
    naming the diverging matrix, its dtype/shape, and the first differing flat index -- never a
    bare pass/fail, so a caller can act on *why* it failed.
    """
    py_dir, rs_dir = Path(py_dir), Path(rs_dir)
    for name in P_MATRIX_FILES:
        py_path, rs_path = py_dir / name, rs_dir / name
        if not py_path.is_file():
            return f"{name}: missing from Python output directory {py_dir}"
        if not rs_path.is_file():
            return f"{name}: missing from Rust output directory {rs_dir}"

        py_arr = np.ascontiguousarray(np.load(py_path))
        rs_arr = np.ascontiguousarray(np.load(rs_path))

        if py_arr.shape != rs_arr.shape:
            return f"{name}: shape mismatch python={py_arr.shape} rust={rs_arr.shape}"
        if py_arr.dtype != rs_arr.dtype:
            return f"{name}: dtype mismatch python={py_arr.dtype} rust={rs_arr.dtype}"

        uint_dtype = {4: np.uint32, 8: np.uint64}.get(py_arr.dtype.itemsize)
        if uint_dtype is None:
            return f"{name}: unexpected dtype {py_arr.dtype} (cannot bit-compare)"
        py_bits = py_arr.ravel().view(uint_dtype)
        rs_bits = rs_arr.ravel().view(uint_dtype)
        if not np.array_equal(py_bits, rs_bits):
            bad = int(np.flatnonzero(py_bits != rs_bits)[0])
            return (
                f"{name} NOT bit-identical: first divergence at flat index {bad} "
                f"(python={py_arr.ravel()[bad]!r}, rust={rs_arr.ravel()[bad]!r}, "
                f"shape={py_arr.shape}, dtype={py_arr.dtype})"
            )
    return None


def load_neurosynth_corpus(seed=0, timeout=15, data_dir=None):
    """Load a real corpus from the cached Neurosynth dataset for the "neurosynth" scale.

    Returns ``(count_df, coordinates_df, meta)`` on success, or ``(None, None, meta)`` with
    ``meta["available"] = False`` if the dataset is not cached and could not be fetched (e.g.
    no network) -- callers must skip the "neurosynth" scale in that case, never silently
    substitute synthetic data under the "neurosynth" label.

    Neurosynth ships tf-idf term weights, not raw word counts, and GCLDA needs non-negative
    integer counts. ``count_df`` here is ``round(tfidf * 100)``, clipped at zero -- a
    non-negative integer approximation chosen only to reproduce a realistic vocabulary size
    and sparsity pattern for *benchmark timing*. It is explicitly NOT a scientifically
    meaningful GCLDA training corpus, and ``meta["counts_are_scaled_tfidf"]`` records this so
    it cannot be missed downstream.
    """
    import socket

    from nimare.extract import fetch_neurosynth

    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    try:
        datasets = fetch_neurosynth(
            data_dir=data_dir,
            version="7",
            return_type="dataset",
            source="abstract",
            vocab="terms",
            type="tfidf",
        )
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure means "skip"
        return None, None, {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        socket.setdefaulttimeout(old_timeout)

    if not datasets:
        return None, None, {"available": False, "reason": "fetch_neurosynth returned no datasets"}

    dataset = datasets[0]
    annotations = dataset.annotations.copy()
    annotations["id"] = annotations["id"].astype(str)
    id_cols = [c for c in ("id", "study_id", "contrast_id") if c in annotations.columns]
    term_cols = [c for c in annotations.columns if c not in id_cols]
    if not term_cols:
        return None, None, {"available": False, "reason": "fetched dataset has no term columns"}

    term_df = annotations.set_index("id")[term_cols]
    # Collapse any duplicate ids (e.g. multiple contrasts per study) rather than silently
    # dropping rows, which pandas set_index alone would leave in place but downstream
    # GCLDAModel's id->index mapping would resolve arbitrarily.
    term_df = term_df.groupby(level=0).sum()

    scale_factor = 100.0
    counts = np.rint(term_df.to_numpy(dtype=np.float64) * scale_factor).astype(np.int64)
    np.clip(counts, 0, None, out=counts)

    rng = np.random.default_rng(seed)
    row_sums = counts.sum(axis=1)
    empty_rows = np.flatnonzero(row_sums == 0)
    if empty_rows.size:
        filler_terms = rng.integers(0, len(term_cols), size=empty_rows.size)
        filler_counts = rng.integers(1, 4, size=empty_rows.size)
        counts[empty_rows, filler_terms] = filler_counts

    count_df = pd.DataFrame(counts, index=term_df.index.astype(str), columns=term_cols)

    coordinates_df = dataset.coordinates[["id", "x", "y", "z"]].copy()
    coordinates_df["id"] = coordinates_df["id"].astype(str)

    meta = {
        "available": True,
        "source": "neurosynth_v7_abstract_terms_tfidf",
        "n_docs": int(count_df.shape[0]),
        "n_terms": int(count_df.shape[1]),
        "n_peaks": int(len(coordinates_df)),
        "counts_are_scaled_tfidf": True,
        "tfidf_scale_factor": scale_factor,
        "caveat": (
            "Neurosynth ships tf-idf term weights, not raw word counts. count_df is "
            "round(tfidf * 100), clipped at zero -- a non-negative integer approximation "
            "used only to reproduce realistic vocabulary size/sparsity for benchmark timing. "
            "This is not a scientifically meaningful GCLDA training corpus."
        ),
    }
    return count_df, coordinates_df, meta


def _stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "n": len(values),
        "all": values,
    }


def summarize(runs, side):
    """Aggregate a list of per-repeat result dicts into min/median/max, never a lone sample."""
    summary = {
        "process_wall_clock_seconds": _stats([r["process_wall_clock_seconds"] for r in runs]),
        "peak_rss_kb": _stats([r["peak_rss_kb"] for r in runs]),
    }
    for phase in PHASE_KEYS:
        summary[f"phase_{phase}_seconds"] = _stats([r["phase_times"][phase] for r in runs])
    if side == "python":
        summary["fit_wall_clock_seconds"] = _stats(
            [r["fit_wall_clock_seconds"] for r in runs]
        )
        summary["construct_seconds"] = _stats([r["construct_seconds"] for r in runs])
    return summary


def print_table(report):
    """Print a plain-text min/median/max table -- raw numbers, not just a ratio."""
    print()
    print("=" * 78)
    status = report["status"]
    print(f"GCLDA benchmark -- scale={report['config']['scale']} status={status}")
    print("=" * 78)

    if status != "ok":
        print(report.get("message", "(no message)"))
        print("=" * 78)
        return

    py, rs = report["summary"]["python"], report["summary"]["rust"]
    rows = [("process_wall_clock_seconds", "process_wall_clock_seconds")]
    rows += [("fit_wall_clock_seconds (python only)", "fit_wall_clock_seconds")]
    rows += [("construct_seconds (python only)", "construct_seconds")]
    rows += [(f"phase_{p}_seconds", f"phase_{p}_seconds") for p in PHASE_KEYS]
    rows += [("peak_rss_kb", "peak_rss_kb")]

    header = f"{'metric':32s} {'python (min/med/max)':30s} {'rust (min/med/max)':30s}"
    print(header)
    print("-" * len(header))
    for label, key in rows:
        py_s, rs_s = py.get(key), rs.get(key)

        def fmt(s):
            if s is None:
                return "n/a"
            return f"{s['min']:.4g}/{s['median']:.4g}/{s['max']:.4g}"

        print(f"{label:32s} {fmt(py_s):30s} {fmt(rs_s):30s}")
    print("-" * len(header))
    print(f"n_repeats={report['config']['repeats']}  jit_warmup_seconds="
          f"{report['jit_warmup_seconds']:.3g}  "
          f"rss_method(python)={report['python_runs'][0]['rss_method']}  "
          f"rss_method(rust)={report['rust_runs'][0]['rss_method']}")
    print()
    print("Equality check: PASSED -- all four p_* matrices bit-identical (verified before")
    print("any timing above was recorded).")
    print()
    print("Caveats (see report JSON 'caveats' for full text):")
    for i, caveat in enumerate(CAVEATS, 1):
        print(f"  {i}. {caveat[:100]}{'...' if len(caveat) > 100 else ''}")
    print("=" * 78)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark GCLDA Python vs Rust training, verifying output equality "
        "before reporting any timing."
    )
    parser.add_argument("--scale", choices=["tiny", "small", "neurosynth"], required=True)
    parser.add_argument("--n-iters", type=int, default=20)
    parser.add_argument("--n-topics", type=int, default=20)
    parser.add_argument("--n-regions", type=int, default=2)
    parser.add_argument(
        "--symmetric", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--loglikely-freq", type=int, default=None)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--peak-block-size",
        type=int,
        default=8192,
        help="Peaks per parallel PDF-evaluation block in the Rust trainer.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0, help="Corpus generation seed.")
    parser.add_argument("--seed-init", type=int, default=1, help="Model sampler seed.")
    parser.add_argument("--binary", default=None, help="Path to gclda-train. Auto-detected.")
    parser.add_argument("--mask", default=None, help="Mask NIfTI path. Defaults to MNI152 2mm.")
    parser.add_argument(
        "--stage-dir",
        default=str(Path(os.environ.get("TMPDIR", "/tmp")) / "gclda_bench_inputs"),
        help="Directory for staged input TSVs and per-run output dirs (persists after exit).",
    )
    parser.add_argument(
        "--neurosynth-timeout",
        type=float,
        default=15.0,
        help="Socket timeout (seconds) for the Neurosynth fetch attempt.",
    )
    parser.add_argument(
        "--neurosynth-data-dir",
        default=None,
        help="Directory holding the cached Neurosynth v7 download. Defaults to NiMARE's "
        "own cache location (~/.nimare). Set this when the cache lives elsewhere, so the "
        "benchmark uses the real corpus instead of skipping the scale.",
    )
    parser.add_argument("--out", default=None, help="Write the full JSON report here.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    binary = Path(args.binary) if args.binary else DEFAULT_BINARY
    if not binary.is_file():
        raise SystemExit(
            f"gclda-train binary not found at {binary}. Build it with "
            "`cd rust/gclda && cargo build --release`, or pass --binary."
        )
    mask_path = args.mask or DEFAULT_MASK
    if not os.path.isfile(mask_path):
        raise SystemExit(f"Mask not found at {mask_path}.")

    stage_dir = Path(args.stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    env_info = collect_environment_info(binary, args.threads)

    if args.scale == "neurosynth":
        count_df, coordinates_df, corpus_meta = load_neurosynth_corpus(
            seed=args.seed,
            timeout=args.neurosynth_timeout,
            data_dir=args.neurosynth_data_dir,
        )
        if count_df is None:
            message = (
                "SKIP: Neurosynth data unavailable "
                f"({corpus_meta.get('reason', 'unknown reason')}). "
                "Not substituting synthetic data under the 'neurosynth' label."
            )
            report = {
                "status": "skipped",
                "message": message,
                "config": vars(args),
                "corpus": corpus_meta,
                "environment": env_info,
            }
            print_table(report)
            if args.out:
                with open(args.out, "w", encoding="utf-8") as fo:
                    json.dump(report, fo, indent=2, default=str)
            return
    else:
        preset = SYNTHETIC_SCALES[args.scale]
        count_df, coordinates_df = make_synthetic_corpus(seed=args.seed, **preset)
        corpus_meta = {
            "available": True,
            "source": f"synthetic:{args.scale}",
            "seed": args.seed,
            **preset,
        }

    loglikely_freq = args.loglikely_freq
    if loglikely_freq is None:
        loglikely_freq = max(1, min(10, args.n_iters))

    train_params = dict(DEFAULT_TRAIN_PARAMS)
    train_params.update(
        n_topics=args.n_topics,
        n_regions=args.n_regions,
        symmetric=args.symmetric,
        seed_init=args.seed_init,
        n_iters=args.n_iters,
        loglikely_freq=loglikely_freq,
        threads=args.threads,
        peak_block_size=args.peak_block_size,
    )

    counts_path, coords_path = export_gclda_tsvs(
        count_df, coordinates_df, stage_dir / "inputs"
    )

    # --- numba JIT warm-up (untimed as a "phase", reported separately) -------------------
    # @njit(cache=True) kernels in nimare/annotate/gclda.py cache compiled code to disk, so
    # this pays the one-time compile cost (normally amortized across an environment's whole
    # lifetime, not per subprocess) up front, keeping it out of the timed repeats below.
    print("Warming up numba JIT (untimed) ...", flush=True)
    warmup_counts, warmup_coords = make_synthetic_corpus(n_docs=4, n_terms=4, n_peaks=8, seed=0)
    warmup_counts_path, warmup_coords_path = export_gclda_tsvs(
        warmup_counts, warmup_coords, stage_dir / "warmup_inputs"
    )
    warmup_params = dict(DEFAULT_TRAIN_PARAMS)
    warmup_params.update(n_topics=2, n_regions=2, symmetric=True, n_iters=1, loglikely_freq=1)
    t0 = time.perf_counter()
    run_python_once(
        warmup_counts_path,
        warmup_coords_path,
        mask_path,
        stage_dir / "py_warmup",
        warmup_params,
        report_path=stage_dir / "warmup_time_report.txt",
    )
    jit_warmup_seconds = time.perf_counter() - t0

    # --- repeats, first of which doubles as the equality check --------------------------
    py_check_dir = stage_dir / "py_out"
    rs_check_dir = stage_dir / "rs_out"
    py_runs, rs_runs = [], []
    for i in range(args.repeats):
        print(f"Run {i + 1}/{args.repeats}: python ...", flush=True)
        py_runs.append(
            run_python_once(
                counts_path,
                coords_path,
                mask_path,
                py_check_dir,
                train_params,
                report_path=stage_dir / f"py_time_report_{i}.txt",
            )
        )
        print(f"Run {i + 1}/{args.repeats}: rust ...", flush=True)
        rs_runs.append(
            run_rust_once(
                binary,
                counts_path,
                coords_path,
                mask_path,
                rs_check_dir,
                train_params,
                report_path=stage_dir / f"rs_time_report_{i}.txt",
            )
        )
        if i == 0:
            mismatch = compare_p_matrices(py_check_dir, rs_check_dir)
            if mismatch is not None:
                message = (
                    "EQUALITY CHECK FAILED -- refusing to report any timing.\n"
                    f"  {mismatch}"
                )
                report = {
                    "status": "equality_check_failed",
                    "message": message,
                    "mismatch": mismatch,
                    "config": vars(args),
                    "corpus": corpus_meta,
                    "params": train_params,
                    "environment": env_info,
                }
                print_table(report)
                if args.out:
                    with open(args.out, "w", encoding="utf-8") as fo:
                        json.dump(report, fo, indent=2, default=str)
                sys.exit(1)

    summary = {"python": summarize(py_runs, "python"), "rust": summarize(rs_runs, "rust")}

    report = {
        "status": "ok",
        "config": vars(args),
        "corpus": corpus_meta,
        "params": train_params,
        "environment": env_info,
        "jit_warmup_seconds": jit_warmup_seconds,
        "equality_check": "passed: all four p_* matrices bit-identical (checked on repeat 1)",
        "caveats": CAVEATS,
        "python_runs": py_runs,
        "rust_runs": rs_runs,
        "summary": summary,
    }

    print_table(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fo:
            json.dump(report, fo, indent=2, default=str)
        print(f"Full report written to {args.out}")


if __name__ == "__main__":
    main()
