"""Small shared helpers for :mod:`nimare.ml`."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import sparse


def _preview(values, limit=8):
    """Render a short, bounded preview of a collection for an error message."""
    values = list(values)
    shown = ", ".join(repr(value) for value in values[:limit])
    if len(values) > limit:
        shown += f", ... ({len(values)} total)"
    return shown or "none"


def _missing_mask(values, kind):
    """Return a boolean mask of values that are absent rather than merely zero."""
    if kind == "numeric":
        return ~np.isfinite(np.asarray(values, dtype=float))

    def absent(value):
        if value is None:
            return True
        if isinstance(value, float) and np.isnan(value):
            return True
        if isinstance(value, str) and not value.strip():
            return True
        return False

    return np.array([absent(value) for value in values], dtype=bool)


def _jsonable(value):
    """Return a JSON-serialisable stand-in for a provenance value."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _to_dense(block):
    """Return a dense view of a feature block."""
    return block.toarray() if sparse.issparse(block) else block


def _check_lengths(n_rows, rows, columns):
    """Refuse blocks that describe different analyses, or name the wrong columns."""
    for name, value in rows.items():
        if value is None:
            continue
        length = value.shape[0] if hasattr(value, "shape") else len(value)
        if length != n_rows:
            raise ValueError(
                f"{name} covers {length} analyses, but map_features has {n_rows} rows."
            )

    for name, (value, expected) in columns.items():
        if value is not None and expected is not None and len(value) != expected:
            raise ValueError(f"{name} names {len(value)} columns, but there are {expected}.")


def _hstack_blocks(blocks):
    """Stack descriptor blocks side by side, staying sparse if any of them is."""
    if any(sparse.issparse(block) for block in blocks):
        return sparse.hstack([_as_sparse(block) for block in blocks], format="csr")
    return np.hstack(blocks)


def _hstack(left, right):
    """Stack two feature blocks, keeping the result sparse if either part is."""
    if right is None:
        return left
    if sparse.issparse(left) or sparse.issparse(right):
        return sparse.hstack([_as_sparse(left), _as_sparse(right)], format="csr")
    return np.hstack([left, right])


def _as_sparse(block):
    """Return a CSR view of a feature block."""
    return block if sparse.issparse(block) else sparse.csr_matrix(block)


def _take_rows(value, rows):
    """Take rows from matrix-like, frame-like or sequence-like data."""
    if value is None:
        return None
    if sparse.issparse(value):
        return value[rows]
    if hasattr(value, "iloc"):
        return value.iloc[rows].copy()
    return np.asarray(value)[rows]
