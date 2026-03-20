"""Studyset/Dataset normalization helpers."""

from __future__ import annotations

from nimare.utils import load_nimads

_TABLE_ATTRS = ("ids", "coordinates", "images", "metadata", "annotations", "texts")


def _snapshot_dataset_tables(dataset, copy_tables=False):
    """Capture Dataset tables for Studyset caching and quickload construction."""
    table_cache = {
        "space": dataset.space,
        "masker": dataset.masker,
        "basepath": dataset.basepath,
    }
    for attr in _TABLE_ATTRS:
        value = getattr(dataset, attr)
        table_cache[attr] = value.copy() if copy_tables else value
    return table_cache


def normalize_collection(dataset):
    """Normalize an input collection to a Dataset or Studyset object."""
    from nimare.dataset import Dataset
    from nimare.nimads import Studyset

    if isinstance(dataset, (Dataset, Studyset)):
        return dataset

    if isinstance(dataset, (dict, str)):
        return load_nimads(dataset)

    raise ValueError(
        "Input must be a Dataset, Studyset, dict, or path to a NIMADS studyset JSON, "
        f"not {type(dataset)}."
    )
