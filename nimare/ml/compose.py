"""Building the scikit-learn step that treats map and descriptor columns apart."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import sparse
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from nimare.ml._helpers import _preview, _to_dense
from nimare.ml.reduce import _resolve_map_reducer


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


def _dense_step(transformer, names):
    """Wrap a descriptor transformer so that it is handed dense columns.

    The block arrives sparse because it sits beside the voxels in one matrix,
    and ``StandardScaler`` refuses to centre sparse data at all.

    The wrapper also restores the descriptor names. A ColumnTransformer selects
    columns by position, because the feature matrix is an array rather than a
    frame, so without this a descriptor comes out of ``get_feature_names_out``
    as ``x228483`` and a fitted coefficient cannot be read back to the field it
    belongs to.
    """
    named = FunctionTransformer(
        accept_sparse=True,
        feature_names_out=lambda _, __: np.asarray(names, dtype=object),
    )
    if isinstance(transformer, str):
        # identity, so the column keeps whatever sparsity it arrived with, but
        # named rather than positional
        return named
    if _handles_sparse(transformer):
        # a sparse-safe transformer keeps the block sparse, which matters when
        # the descriptors are a pattern selection thousands of labels wide
        return Pipeline([("name", named), ("transform", transformer)])
    return Pipeline(
        [
            (
                "to_dense",
                FunctionTransformer(
                    _to_dense,
                    accept_sparse=True,
                    feature_names_out=lambda _, __: np.asarray(names, dtype=object),
                ),
            ),
            ("transform", transformer),
        ]
    )


def _is_passthrough(transformer):
    """Report whether a transformer slot was left at its default."""
    return isinstance(transformer, str) and transformer == "passthrough"


def _step_name(name, position, used):
    """Return a ColumnTransformer step name, which may not contain ``__``.

    Annotation fields often do -- ``Neurosynth_TFIDF__pain`` -- so the name is
    softened, and disambiguated by position if softening made it collide.
    """
    safe = name.replace("__", "_") or f"descriptor_{position}"
    if safe in used:
        safe = f"{safe}_{position}"
    used.add(safe)
    return safe


def _descriptor_step(bunch, descriptor_transformer):
    """Return the step applied to the descriptor block."""
    names = list(bunch.descriptor_names)
    if not isinstance(descriptor_transformer, Mapping):
        return _dense_step(descriptor_transformer, names)

    unknown = [name for name in descriptor_transformer if name not in names]
    if unknown:
        raise ValueError(
            f"No descriptor called {_preview(unknown)}. This bundle has {_preview(names)}."
        )

    # Column indices here are relative to the descriptor block, and every
    # descriptor gets a step of its own so the columns come out in the order they
    # went in rather than in the order the mapping happened to name them.
    used = set()
    steps = []
    for position, name in enumerate(names):
        steps.append(
            (
                _step_name(name, position, used),
                _dense_step(descriptor_transformer.get(name, "passthrough"), [name]),
                [position],
            )
        )

    return ColumnTransformer(steps, sparse_threshold=1.0, verbose_feature_names_out=False)


def make_nimare_column_transformer(
    bunch,
    map_reducer,
    descriptor_transformer="passthrough",
    **reducer_params,
):
    """Apply a reducer to the map columns of ``bunch`` and something else to the rest.

    A :class:`~sklearn.compose.ColumnTransformer` with the column boundary
    filled in from the bundle, the bundle's masker bound into an atlas reducer,
    and ``sparse_threshold=1.0`` so a wide sparse map block is never quietly
    densified. With no descriptor columns to keep the reducer away from, the
    reducer is returned as it is.

    Parameters
    ----------
    bunch : :class:`sklearn.utils.Bunch`
        A bundle from :meth:`~nimare.studyset.Studyset.to_bunch`.
    map_reducer : estimator, :obj:`type`, atlas or None
        A scikit-learn transformer, a transformer class built here from
        ``**reducer_params``, any atlas :class:`~nimare.ml.AtlasAggregator`
        accepts, or None to leave the map columns alone.
    descriptor_transformer : estimator, :obj:`str` or :obj:`dict`, default="passthrough"
        One transformer for every descriptor column, or a mapping from
        descriptor name to transformer. Descriptors the mapping does not name
        are passed through, and the column order is the one they came in with.
        Transformers are handed their columns dense.
    **reducer_params
        Passed to ``map_reducer`` when it is a class.

    Returns
    -------
    estimator
        A :class:`~sklearn.compose.ColumnTransformer` when the bundle has
        descriptor columns, and the reducer itself when it has not.

    Raises
    ------
    :obj:`ValueError`
        If ``descriptor_transformer`` names something that is not a descriptor,
        if it is given for a bundle with no descriptor columns, or if
        parameters are passed alongside a built transformer.

    See Also
    --------
    sklearn.compose.make_column_transformer : The scikit-learn shorthand this
        follows. Write that one out with ``bunch.map_columns`` and
        ``bunch.descriptor_columns`` for anything this does not cover.

    Examples
    --------
    >>> pipeline = make_pipeline(  # doctest: +SKIP
    ...     make_nimare_column_transformer(bunch, TruncatedSVD(n_components=50)),
    ...     LogisticRegression(),
    ... )
    """
    if map_reducer is None or _is_passthrough(map_reducer):
        reducer = "passthrough"
    else:
        reducer = _resolve_map_reducer(map_reducer, masker=bunch.masker, **reducer_params)

    descriptors = bunch.descriptor_columns
    if descriptors.stop <= descriptors.start:
        if not _is_passthrough(descriptor_transformer):
            raise ValueError(
                "This bundle has no descriptor columns, so there is nothing for "
                "descriptor_transformer to act on."
            )
        return reducer

    return ColumnTransformer(
        [
            ("maps", reducer, bunch.map_columns),
            ("descriptors", _descriptor_step(bunch, descriptor_transformer), descriptors),
        ],
        sparse_threshold=1.0,
    )
