"""Small shared helpers for :mod:`nimare.ml`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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


class _FeatureNames(Sequence):
    """A bundle's column names, built on access rather than up front."""

    def __init__(self, n_voxels, descriptor_names):
        self.n_voxels = int(n_voxels)
        self.descriptor_names = list(descriptor_names)

    def __len__(self):
        return self.n_voxels + len(self.descriptor_names)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        position = int(index)
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError(index)
        if position < self.n_voxels:
            return f"voxel_{position}"
        return self.descriptor_names[position - self.n_voxels]

    def __contains__(self, value):
        return self._position(value) is not None

    def index(self, value, start=0, stop=None):
        """Return where ``value`` sits, without naming every column to find it."""
        position = self._position(value)
        stop = len(self) if stop is None else stop
        if position is None or not start <= position < stop:
            raise ValueError(f"{value!r} is not in the feature names")
        return position

    def _position(self, value):
        """Return the column ``value`` names, or None, by reading the name."""
        if not isinstance(value, str):
            return None
        if value in self.descriptor_names:
            return self.n_voxels + self.descriptor_names.index(value)
        if not value.startswith("voxel_"):
            return None
        try:
            position = int(value[len("voxel_") :])
        except ValueError:
            return None
        return position if 0 <= position < self.n_voxels else None

    def __repr__(self):
        return f"<{len(self)} feature names>"


class _NamesAt(Sequence):
    """The names of selected columns, likewise built on access."""

    def __init__(self, names, indices):
        self.names = names
        self.indices = np.atleast_1d(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [str(self.names[position]) for position in self.indices[index]]
        return str(self.names[self.indices[index]])

    def __repr__(self):
        return f"<{len(self)} column names>"
