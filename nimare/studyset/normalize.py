"""Coerce assorted inputs into a :class:`~nimare.studyset.studyset.Studyset`.

This is the boundary every algorithm entry point goes through, which is why
the module is not underscore-prefixed: :func:`normalize_collection` is part
of the public surface even though what it mostly exists to absorb -- the
deprecated :class:`~nimare.dataset.Dataset` -- is not.
"""

from __future__ import annotations

__all__ = ["normalize_collection"]


def normalize_collection(collection):
    """Return a Studyset for whatever was passed in.

    A :class:`~nimare.dataset.Dataset` is converted here, at the boundary, so
    that nothing downstream has to know about it. ``Dataset`` is deprecated; the
    conversion is correctness-first, not speed-first.

    Passing a ``Dataset`` emits a :class:`FutureWarning`. Nearly every algorithm
    entry point warns from here; the exception is
    :meth:`~nimare.meta.kernel.KernelTransformer.transform`, which reads a ``Dataset``
    without normalising it and so raises the notice itself.
    """
    from nimare.dataset import Dataset, _warn_dataset_input
    from nimare.studyset.studyset import Studyset

    if isinstance(collection, Studyset):
        return collection
    if isinstance(collection, Dataset):
        _warn_dataset_input()
        return Studyset.from_dataset(collection)
    if isinstance(collection, (dict, str)):
        return Studyset(collection)
    raise ValueError(
        "Input must be a Studyset, Dataset, dict, or path to a NIMADS studyset "
        f"JSON, not {type(collection)}."
    )
