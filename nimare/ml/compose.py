"""A :func:`~sklearn.compose.make_column_transformer` that knows the bunch's blocks."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, _name_estimators
from sklearn.preprocessing import FunctionTransformer

from nimare.ml._helpers import _NamesAt, _preview, _to_dense
from nimare.ml.reduce import _resolve_voxel_transformer

BLOCKS = ("voxels", "descriptors")

_PROBE = sparse.csr_matrix(np.array([[1.0], [0.0], [2.0], [3.0]]))

try:  # scikit-learn >= 1.6
    from sklearn.utils import get_tags
except ImportError:  # pragma: no cover - exercised on the minimum version
    get_tags = None


def _handles_sparse(transformer):
    """Report whether ``transformer`` reads sparse input.

    Asks scikit-learn's own tags where they exist. They answer per instance, so
    ``StandardScaler(with_mean=False)`` is separated from ``StandardScaler()``,
    and they compose through a :class:`~sklearn.pipeline.Pipeline`. On older
    scikit-learn, fit a clone on a sparse probe and see whether making the
    probe dense is what fixes a failure.
    """
    if get_tags is not None:
        try:
            return bool(get_tags(transformer).input_tags.sparse)
        except Exception:
            pass
    if not _fails(transformer, _PROBE):
        return True
    return _fails(transformer, _PROBE.toarray())


def _fails(transformer, probe):
    """Report whether fitting a clone on ``probe`` raises."""
    try:
        clone(transformer).fit(probe)
    except Exception:
        return True
    return False


def _named(names, func=None):
    """Return a step that reports ``names`` as its output names, densifying with ``func``."""
    return FunctionTransformer(
        func,
        accept_sparse=True,
        feature_names_out=lambda _, __: np.asarray(names, dtype=object),
    )


def _columns_of(bunch, columns):
    """Return the column spec ``columns`` means, and the names of those columns."""
    wanted = [columns] if isinstance(columns, str) else columns
    if _all_strings(wanted):
        columns = _by_name(bunch, wanted)
    return columns, _NamesAt(bunch.feature_names, _positions(columns, len(bunch.feature_names)))


def _positions(columns, n_columns):
    """Return the column positions a spec covers, without naming them.

    A slice is turned into its own range rather than used to index one, because
    the voxel block spans the whole image grid.
    """
    if isinstance(columns, slice):
        return np.arange(*columns.indices(n_columns))
    values = np.asarray(columns)
    if values.dtype == bool:
        return np.flatnonzero(values)
    if np.issubdtype(values.dtype, np.integer):
        return np.atleast_1d(values)
    return np.atleast_1d(np.arange(n_columns)[columns])


def _all_strings(columns):
    """Report whether ``columns`` is a name or a sequence of names."""
    return (
        isinstance(columns, (list, tuple))
        and bool(columns)
        and all(isinstance(name, str) for name in columns)
    )


def _by_name(bunch, wanted):
    """Return the columns the names in ``wanted`` select.

    One name is its own slice, so naming the voxel block costs nothing; several
    are concatenated in the order they were given.
    """
    descriptors = list(bunch.descriptor_names)
    offset = bunch.descriptor_columns.start
    spans = []
    for name in wanted:
        if name in BLOCKS:
            spans.append(_block_span(bunch, name))
        elif name in descriptors:
            position = offset + descriptors.index(name)
            spans.append(slice(position, position + 1))
        else:
            raise ValueError(
                f"{name!r} names neither a block nor a descriptor of this bunch. The "
                f"blocks are {', '.join(BLOCKS)} and the descriptors are "
                f"{_preview(descriptors)}; anything else must be a column spec, such as "
                "bunch.voxel_columns."
            )
    if len(spans) == 1:
        return spans[0]
    return np.concatenate([np.arange(span.start, span.stop) for span in spans])


def _resolve(bunch, transformer):
    """Return the transformer a step will really use, with an atlas resolved."""
    if isinstance(transformer, str):
        if transformer not in ("passthrough", "drop"):
            raise ValueError(
                f"{transformer!r} is not a transformer. A string may only be "
                "'passthrough' or 'drop', as it may be for a ColumnTransformer."
            )
        return transformer
    return _resolve_voxel_transformer(transformer, masker=bunch.get("masker"))


def _step(transformer, columns, coded, given):
    """Return the transformer to use for one block, wrapped where it has to be.

    ``given`` is the transformer as the caller wrote it, which is what the
    sparse probe must see rather than the one told which categories to expect.
    """
    if transformer == "drop":
        return transformer
    if transformer == "passthrough":
        # an identity rather than the string, so the columns stay named
        return _named(columns)
    densify = None if _handles_sparse(given) else _to_dense
    steps = [("name", _named(columns, densify)), ("transform", transformer)]
    if coded:
        steps.append(("label", _relabel(coded)))
    return Pipeline(steps)


def _coded_columns(bunch, columns):
    """Return the coded categoricals in one spec, and the columns that are not.

    Read from positions rather than names: a voxel spec covers the whole grid.
    """
    positions = _positions(columns, len(bunch.feature_names))
    categories = bunch.get("descriptor_categories") or {}
    if not categories:
        return {}, positions

    span = bunch.descriptor_columns
    names = list(bunch.descriptor_names)
    inside = positions[(positions >= span.start) & (positions < span.stop)]

    found, coded_positions = {}, []
    for position in inside:
        name = names[int(position) - span.start]
        if name in categories:
            found[name] = categories[name]
            coded_positions.append(int(position))

    others = positions[~np.isin(positions, coded_positions)] if found else positions
    return found, others


def _relabel(coded):
    """Return a step that puts category labels back into the output names."""
    replacements = {
        f"{name}_{float(position)}": f"{name}_{label}"
        for name, labels in coded.items()
        for position, label in enumerate(labels)
    }

    def named(_, input_features):
        return np.asarray(
            [replacements.get(str(name), str(name)) for name in input_features], dtype=object
        )

    return FunctionTransformer(accept_sparse=True, feature_names_out=named)


def _with_categories(transformer, coded):
    """Tell an encoder which categories to expect, when it would guess."""
    if not coded or isinstance(transformer, str):
        return transformer
    if "categories" not in getattr(transformer, "get_params", dict)():
        return transformer
    if not isinstance(transformer.categories, str):
        return transformer
    expected = [np.arange(len(labels), dtype=float) for labels in coded.values()]
    return clone(transformer).set_params(categories=expected)


def _check_categoricals(bunch, used, coded):
    """Refuse to hand a raw category code to a model."""
    for transformer, (labels, others) in zip(used, coded):
        if not labels or transformer == "drop":
            continue

        if len(others):
            raise ValueError(
                f"{_preview(sorted(labels))} is a coded categorical descriptor, and this "
                f"spec also covers {len(others)} column(s) that are not: "
                f"{_preview(_NamesAt(bunch.feature_names, others[:8]))}. A code stands for "
                "a label rather than a quantity, so give the categorical columns a spec of "
                f"their own with an encoder, such as (OneHotEncoder(), {sorted(labels)[0]!r})."
            )
        if transformer == "passthrough":
            raise ValueError(
                f"{_preview(sorted(labels))} is a coded categorical descriptor and cannot "
                "be passed through: its values are positions in "
                "bunch.descriptor_categories, so a model would read the third category as "
                "three times the first. Name an encoder, such as OneHotEncoder() or "
                "TargetEncoder(), or ('drop', ...) if it is not wanted."
            )


def _check_claims(bunch, specs, remainder):
    """Refuse to silently drop any block nobody asked about."""
    if remainder != "drop":
        return
    # a mask rather than a set of indices: the voxel block spans the whole grid
    claimed = np.zeros(len(bunch.feature_names), dtype=bool)
    for columns in specs:
        claimed[columns] = True

    for block in BLOCKS:
        span = _block_span(bunch, block)
        if span.stop <= span.start:
            continue
        unclaimed = span.start + np.flatnonzero(~claimed[span])
        if not len(unclaimed):
            continue
        names = _NamesAt(bunch.feature_names, unclaimed[:8])
        raise ValueError(
            f"No transformer covers {len(unclaimed)} of the {block} columns: "
            f"{_preview(names)}. A ColumnTransformer drops what nobody claims, so name "
            f"them with a {block!r} spec, pass ('drop', {block!r}) if that is meant, or "
            "remainder='passthrough' to keep them as they are."
        )


def _block_span(bunch, block):
    """Return the column slice one block name covers."""
    return bunch.voxel_columns if block == "voxels" else bunch.descriptor_columns


def _read_pairs(bunch, transformers):
    """Return the transformers and the column specs a call's pairs name."""
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

    :func:`~sklearn.compose.make_column_transformer` with the bunch filled in:
    the column spans of the two blocks, the masker an atlas needs, the column
    names each transformer is given, and the categories a categorical
    descriptor code stands for. Everything else is scikit-learn's, including
    the shape of ``transformers`` and the automatic step names.

    For a case this does not cover, write ``ColumnTransformer`` out with
    ``bunch.voxel_columns`` and ``bunch.descriptor_columns``, which are
    ordinary slices.

    Parameters
    ----------
    bunch : :class:`sklearn.utils.Bunch`
        A bunch from :meth:`~nimare.studyset.Studyset.to_bunch`.
    *transformers : :obj:`tuple`
        ``(transformer, columns)`` pairs, as
        :func:`~sklearn.compose.make_column_transformer` takes them.

        ``columns`` may be a name, or a list of names, as it may be for a
        ColumnTransformer reading a frame: ``"voxels"`` and ``"descriptors"``
        name the two blocks, and a descriptor may be named by its own field
        name. It may equally be anything a
        :class:`~sklearn.compose.ColumnTransformer` accepts: a slice, indices,
        a mask or a callable.

        ``transformer`` may be a scikit-learn transformer, ``"passthrough"``,
        ``"drop"``, or any atlas :class:`~nimare.ml.MaskerTransformer` accepts,
        which is built against the bunch's masker.
    remainder : {"drop", "passthrough"} or estimator, default="drop"
        What happens to columns no transformer claims, as in scikit-learn.
        Claim both blocks under ``"drop"``; say ``("drop", "voxels")`` to drop
        one on purpose.
    sparse_threshold : :obj:`float`, default=1.0
        Scikit-learn defaults this to 0.3, which would densify an unreduced
        voxel block -- about 6.5 GB at 902,629 columns -- so the default here
        keeps the result sparse whenever any block is.
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
        If a pair is malformed, if a block name is not one of the bunch's, if
        either block would be dropped without being named, or if a coded
        categorical descriptor would reach a model unencoded.

    See Also
    --------
    sklearn.compose.make_column_transformer : The function this follows.

    Examples
    --------
    >>> preprocessor = make_nimare_column_transformer(  # doctest: +SKIP
    ...     bunch,
    ...     (MAKernel(MKDAKernel(r=10), source_masker=bunch.masker), "voxels"),
    ...     (SimpleImputer(strategy="median"), "descriptors"),
    ... )
    """
    used, resolved = _read_pairs(bunch, transformers)
    _check_claims(bunch, [columns for columns, _ in resolved], remainder)
    coded = [_coded_columns(bunch, columns) for columns, _ in resolved]
    _check_categoricals(bunch, used, coded)
    given = used
    used = [_with_categories(transformer, labels) for transformer, (labels, _) in zip(used, coded)]

    steps = [
        (_step(transformer, names, labels, written), columns)
        for transformer, written, (columns, names), (labels, _) in zip(
            used, given, resolved, coded
        )
    ]
    # named after the transformer rather than the wrapper, as sklearn would
    names = [name for name, _ in _name_estimators(used)] if used else []
    return ColumnTransformer(
        [(name, step, columns) for name, (step, columns) in zip(names, steps)],
        n_jobs=n_jobs,
        remainder=remainder,
        sparse_threshold=sparse_threshold,
        verbose=verbose,
        verbose_feature_names_out=verbose_feature_names_out,
    )
