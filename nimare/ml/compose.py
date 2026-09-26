"""A :func:`~sklearn.compose.make_column_transformer` that knows the bundle's blocks."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, _name_estimators
from sklearn.preprocessing import FunctionTransformer

from nimare.ml._helpers import _preview, _to_dense
from nimare.ml.reduce import _resolve_map_reducer

BLOCKS = ("maps", "descriptors")


def _handles_sparse(transformer):
    """Report whether ``transformer`` can be fitted on a sparse block.

    Asked by fitting a clone on a tiny sparse probe, because the answer is a
    property of the arguments rather than of the class: ``StandardScaler()``
    refuses sparse input while ``StandardScaler(with_mean=False)`` does not,
    and no estimator tag separates them. Anything that fails the probe is
    handed dense columns, which always works.
    """
    probe = sparse.csr_matrix(np.array([[1.0], [0.0], [2.0], [3.0]]))
    try:
        clone(transformer).fit(probe)
    except Exception:
        return False
    return True


def _named(names, func=None):
    """Return a step that reports ``names`` as its output names.

    A ColumnTransformer selects columns by position, because the feature
    matrix is an array rather than a frame, so without this a column comes out
    of ``get_feature_names_out`` as ``x228483`` and a fitted coefficient cannot
    be read back to the feature it weighs. ``func`` densifies on the way
    through when the transformer downstream needs it; naming and densifying
    are one step so that nothing in the chain lacks
    ``get_feature_names_out``.
    """
    return FunctionTransformer(
        func,
        accept_sparse=True,
        feature_names_out=lambda _, __: np.asarray(names, dtype=object),
    )


def _columns_of(bunch, columns):
    """Return the column spec ``columns`` means, and the names of those columns.

    A string is a column name, as it is to a
    :class:`~sklearn.compose.ColumnTransformer` reading a frame. Here the names
    are the two blocks, ``"maps"`` and ``"descriptors"``, and the descriptors'
    own field names; a block name wins if a descriptor shares it.
    """
    wanted = [columns] if isinstance(columns, str) else columns
    if _all_strings(wanted):
        columns = _by_name(bunch, wanted)
    indices = np.arange(len(bunch.feature_names))[columns]
    return columns, [str(bunch.feature_names[index]) for index in np.atleast_1d(indices)]


def _all_strings(columns):
    """Report whether ``columns`` is a name or a sequence of names."""
    return (
        isinstance(columns, (list, tuple))
        and bool(columns)
        and all(isinstance(name, str) for name in columns)
    )


def _by_name(bunch, wanted):
    """Return the column indices the names in ``wanted`` select."""
    descriptors = list(bunch.descriptor_names)
    offset = bunch.descriptor_columns.start
    indices = []
    for name in wanted:
        if name in BLOCKS:
            block = bunch.map_columns if name == "maps" else bunch.descriptor_columns
            indices.extend(range(block.start, block.stop))
        elif name in descriptors:
            indices.append(offset + descriptors.index(name))
        else:
            raise ValueError(
                f"{name!r} names neither a block nor a descriptor of this bundle. The "
                f"blocks are {', '.join(BLOCKS)} and the descriptors are "
                f"{_preview(descriptors)}; anything else must be a column spec, such as "
                "bunch.map_columns."
            )
    return indices


def _resolve(bunch, transformer):
    """Return the transformer a step will really use.

    An atlas becomes an :class:`~nimare.ml.MaskerTransformer` bound to the
    bundle's masker, which is also what the step is then named after, since
    ``nifti1image`` would say nothing about what the step does.
    """
    if isinstance(transformer, str):
        if transformer not in ("passthrough", "drop"):
            raise ValueError(
                f"{transformer!r} is not a transformer. A string may only be "
                "'passthrough' or 'drop', as it may be for a ColumnTransformer."
            )
        return transformer
    return _resolve_map_reducer(transformer, masker=bunch.get("masker"))


def _step(transformer, columns):
    """Return the transformer to use for one block, wrapped where it has to be."""
    if transformer == "drop":
        return transformer
    if transformer == "passthrough":
        # an identity rather than the string, so the columns keep whatever
        # sparsity they arrived with and are still named
        return _named(columns)
    # a sparse-safe transformer keeps the block sparse, which matters when it
    # is a pattern selection thousands of labels wide
    densify = None if _handles_sparse(transformer) else _to_dense
    return Pipeline([("name", _named(columns, densify)), ("transform", transformer)])


def _check_claims(bunch, specs, remainder):
    """Refuse to silently drop descriptor columns nobody asked about."""
    descriptors = bunch.descriptor_columns
    if descriptors.stop <= descriptors.start or remainder != "drop":
        return
    claimed = set()
    for columns in specs:
        claimed.update(np.arange(len(bunch.feature_names))[columns].tolist())
    unclaimed = [
        index for index in range(descriptors.start, descriptors.stop) if index not in claimed
    ]
    if unclaimed:
        names = [str(bunch.feature_names[index]) for index in unclaimed]
        raise ValueError(
            f"No transformer covers {len(unclaimed)} descriptor column(s): "
            f"{_preview(names)}. A ColumnTransformer drops what nobody claims, so name "
            "them with a ('descriptors') spec, or pass remainder='passthrough' to keep "
            "them as they are."
        )


def _read_pairs(bunch, transformers):
    """Return the transformers and the column specs a call's pairs name.

    The transformers resolve first, so that an unusable one is reported as
    such rather than as whatever its columns did or did not cover.
    """
    for pair in transformers:
        if not (isinstance(pair, tuple) and len(pair) == 2):
            raise ValueError(
                f"{pair!r} is not a (transformer, columns) pair. "
                "make_nimare_column_transformer takes them the way "
                "sklearn.compose.make_column_transformer does."
            )

    used = [_resolve(bunch, transformer) for transformer, _ in transformers]
    resolved = [_columns_of(bunch, columns) for _, columns in transformers]
    return used, resolved


def make_nimare_column_transformer(
    bunch,
    *transformers,
    remainder="drop",
    sparse_threshold=1.0,
    n_jobs=None,
    verbose=False,
    verbose_feature_names_out=True,
):
    """Construct a ColumnTransformer over the blocks of ``bunch``.

    :func:`~sklearn.compose.make_column_transformer` with three things added
    that a bundle knows and scikit-learn cannot work out on its own: the column
    spans of the two blocks, the masker an atlas reducer needs, and the names
    of the columns each transformer is given.

    Everything else is scikit-learn's, including the shape of ``transformers``
    and the automatic step names. Where this function does not cover a case,
    write ``ColumnTransformer`` out with ``bunch.map_columns`` and
    ``bunch.descriptor_columns``, which are ordinary slices.

    Parameters
    ----------
    bunch : :class:`sklearn.utils.Bunch`
        A bundle from :meth:`~nimare.studyset.Studyset.to_bunch`.
    *transformers : :obj:`tuple`
        ``(transformer, columns)`` pairs, as
        :func:`~sklearn.compose.make_column_transformer` takes them.

        ``columns`` may be a name, or a list of names, as it may be for a
        ColumnTransformer reading a frame: ``"maps"`` and ``"descriptors"``
        name the two blocks, and a descriptor may be named by its own field
        name. It may equally be anything a
        :class:`~sklearn.compose.ColumnTransformer` accepts: a slice, indices,
        a mask or a callable.

        ``transformer`` may be a scikit-learn transformer, ``"passthrough"``,
        ``"drop"``, or any atlas :class:`~nimare.ml.MaskerTransformer` accepts,
        which is built against the bundle's masker.
    remainder : {"drop", "passthrough"} or estimator, default="drop"
        What happens to columns no transformer claims, as in scikit-learn.
        Leaving descriptor columns unclaimed under ``"drop"`` raises rather
        than discarding them quietly.
    sparse_threshold : :obj:`float`, default=1.0
        Scikit-learn defaults this to 0.3, which would densify an unreduced map
        block -- about 1.6 GB at 228,000 columns -- so the default here keeps
        the result sparse whenever any block is.
    n_jobs : :obj:`int`, optional
        Passed to :class:`~sklearn.compose.ColumnTransformer`.
    verbose : :obj:`bool`, default=False
        Passed to :class:`~sklearn.compose.ColumnTransformer`.
    verbose_feature_names_out : :obj:`bool`, default=True
        Passed to :class:`~sklearn.compose.ColumnTransformer`.

    Returns
    -------
    :class:`~sklearn.compose.ColumnTransformer`
        Unfitted, with the steps named after their transformers.

    Raises
    ------
    :obj:`ValueError`
        If a pair is malformed, if a block name is not one of the bundle's, or
        if descriptor columns would be dropped without being named.

    See Also
    --------
    sklearn.compose.make_column_transformer : The function this follows.

    Examples
    --------
    >>> preprocessor = make_nimare_column_transformer(  # doctest: +SKIP
    ...     bunch,
    ...     (TruncatedSVD(n_components=50), "maps"),
    ...     (SimpleImputer(strategy="median"), "descriptors"),
    ... )
    """
    used, resolved = _read_pairs(bunch, transformers)
    _check_claims(bunch, [columns for columns, _ in resolved], remainder)

    steps = [
        (_step(transformer, names), columns)
        for transformer, (columns, names) in zip(used, resolved)
    ]
    # named after the transformer rather than the wrapper built around it, so
    # the step names are the ones scikit-learn would have chosen
    names = [name for name, _ in _name_estimators(used)] if used else []
    return ColumnTransformer(
        [(name, step, columns) for name, (step, columns) in zip(names, steps)],
        n_jobs=n_jobs,
        remainder=remainder,
        sparse_threshold=sparse_threshold,
        verbose=verbose,
        verbose_feature_names_out=verbose_feature_names_out,
    )
