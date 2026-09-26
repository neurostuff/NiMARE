"""Building the scikit-learn step that treats map and descriptor columns apart."""

from __future__ import annotations

from collections.abc import Mapping

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from nimare.ml._helpers import _preview, _to_dense
from nimare.ml.reduce import _resolve_map_reducer


def _dense_step(transformer):
    """Wrap a descriptor transformer so that it is handed dense columns.

    The block arrives sparse because it sits beside the voxels in one matrix,
    and ``StandardScaler`` refuses to centre sparse data at all.
    """
    if isinstance(transformer, str):
        return transformer
    return Pipeline(
        [
            ("to_dense", FunctionTransformer(_to_dense, accept_sparse=True)),
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
    if not isinstance(descriptor_transformer, Mapping):
        return _dense_step(descriptor_transformer)

    names = list(bunch.descriptor_names)
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
                _dense_step(descriptor_transformer.get(name, "passthrough")),
                [position],
            )
        )

    return ColumnTransformer(steps, sparse_threshold=1.0)


def make_preprocessor(
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

    Examples
    --------
    >>> pipeline = make_pipeline(  # doctest: +SKIP
    ...     make_preprocessor(bunch, TruncatedSVD(n_components=50)),
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
