"""Coerce assorted inputs into a :class:`~nimare.studyset.studyset.Studyset`."""

from __future__ import annotations

__all__ = ["normalize_collection"]


def normalize_collection(collection):
    """Return a Studyset for whatever was passed in.

    A :class:`~nimare.dataset.Dataset` is converted here, at the boundary, so
    that nothing downstream has to know about it. ``Dataset`` is deprecated; the
    conversion is correctness-first, not speed-first.
    """
    from nimare.dataset import Dataset
    from nimare.studyset.studyset import Studyset

    if isinstance(collection, Studyset):
        return collection
    if isinstance(collection, Dataset):
        return Studyset.from_dataset(collection)
    if isinstance(collection, (dict, str)):
        return Studyset(collection)
    raise ValueError(
        "Input must be a Studyset, Dataset, dict, or path to a NIMADS studyset "
        f"JSON, not {type(collection)}."
    )
