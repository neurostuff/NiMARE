"""Deterministic synthetic corpus generator for GCLDA benchmarking.

Produces a word-count matrix and a peak-coordinates table with the same
shapes :class:`nimare.annotate.gclda.GCLDAModel` and the Rust
``gclda-train`` trainer both expect, at whatever scale the caller asks for,
without needing a real (large, slow-to-fetch) corpus. This is used by
``benchmarks/bench_gclda_rust.py`` for the ``tiny`` and ``small`` benchmark
scales; the ``neurosynth`` scale uses real coordinates instead (see that
script), because a fabricated corpus would not exercise realistic spatial
clustering.
"""

import numpy as np
import pandas as pd


def make_synthetic_corpus(n_docs, n_terms, n_peaks, seed=0):
    """Generate a deterministic synthetic word-count + coordinates corpus.

    Parameters
    ----------
    n_docs : :obj:`int`
        Number of documents (studies).
    n_terms : :obj:`int`
        Vocabulary size.
    n_peaks : :obj:`int`
        Number of peak coordinates to generate, distributed across the
        ``n_docs`` documents (not necessarily one per document -- some
        documents may receive zero peaks, others several).
    seed : :obj:`int`, optional
        Seed for the deterministic random number generator. The same seed
        with the same ``n_docs``/``n_terms``/``n_peaks`` always produces
        byte-identical output. Default is 0.

    Returns
    -------
    count_df : :obj:`pandas.DataFrame`
        ``(n_docs, n_terms)`` non-negative integer word-count matrix, indexed
        by document ID. Every row has at least one nonzero count -- a
        document with zero tokens is a degenerate input GCLDA cannot usefully
        train on, not a realistic benchmark case.
    coordinates_df : :obj:`pandas.DataFrame`
        ``n_peaks`` rows with ``id``, ``x``, ``y``, ``z`` columns. ``id``
        values are always a subset of ``count_df.index``.
    """
    rng = np.random.default_rng(seed)

    doc_ids = [f"synth-{i:06d}" for i in range(n_docs)]
    terms = [f"term_{j:05d}" for j in range(n_terms)]

    # Sparse, right-skewed counts, similar in shape to real abstract word
    # counts (most words absent or rare, a few repeated).
    counts = rng.poisson(lam=0.8, size=(n_docs, n_terms)).astype(np.int64)

    # A document with an all-zero row is a degenerate input (no tokens to
    # sample topics from), not a realistic benchmark case -- force at least
    # one nonzero count per document.
    row_sums = counts.sum(axis=1)
    empty_rows = np.flatnonzero(row_sums == 0)
    if empty_rows.size:
        filler_terms = rng.integers(0, n_terms, size=empty_rows.size)
        filler_counts = rng.integers(1, 4, size=empty_rows.size)
        counts[empty_rows, filler_terms] = filler_counts

    count_df = pd.DataFrame(counts, index=doc_ids, columns=terms)

    # Peaks are distributed across documents (not necessarily uniformly --
    # some documents may get more than others, some none at all), with
    # coordinates spread over a realistic MNI-ish bounding box.
    doc_for_peak = rng.integers(0, n_docs, size=n_peaks)
    coordinates_df = pd.DataFrame(
        {
            "id": [doc_ids[d] for d in doc_for_peak],
            "x": np.round(rng.uniform(-70.0, 70.0, size=n_peaks), 1),
            "y": np.round(rng.uniform(-105.0, 70.0, size=n_peaks), 1),
            "z": np.round(rng.uniform(-45.0, 78.0, size=n_peaks), 1),
        }
    )

    return count_df, coordinates_df
