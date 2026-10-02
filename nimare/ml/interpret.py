"""Reading a fitted model back to the brain."""

from __future__ import annotations

import numpy as np
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker, NiftiMasker
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils import _safe_indexing

from nimare.ml._helpers import _to_dense
from nimare.ml._peaks import mask_source
from nimare.ml.kernel import MAKernel
from nimare.ml.reduce import MaskerTransformer

KINDS = ("weights", "pattern")

ATLAS_READINGS = ("voxel", "region")

# NiftiMasker options that make its transform depend on the other rows or on time,
# so that it has no fixed linear part to transpose; standardize_confounds is left out
# because it only acts on confounds, which a MaskerTransformer never passes
_CLEANING = ("standardize", "detrend", "low_pass", "high_pass")


def coefficient_image(estimator, bunch, coef=None, kind="weights", atlas="voxel", X=None):
    """Return a fitted model's voxel weights, or its activation pattern, as an image.

    With ``kind="weights"``, walks the fitted steps backwards, undoing each
    reduction, until the weights are one per voxel again, and unmasks them.
    Every step between the model and the voxels has to be read back for a
    weight to name a place in the brain, and a step that cannot be stops the
    walk and says so.

    With ``kind="pattern"``, returns the activation pattern of
    :footcite:t:`haufe2014interpretation` instead: how each voxel covaries with
    the model's scores. Weights say what a model uses, including voxels that
    only cancel noise elsewhere; a pattern says where the data differ along what
    the model reads, and is what a weight map is usually mistaken for.

    Parameters
    ----------
    estimator : :class:`~sklearn.pipeline.Pipeline` or estimator
        A fitted pipeline ending in a model with ``coef_``, or the model
        itself when the features are already voxels.
    bunch : :class:`sklearn.utils.Bunch`
        The bunch the model was fitted on, for its ``masker`` and the span of
        its voxel columns.
    coef : array_like, optional
        Weights to project instead of the model's own, by default None. One per
        feature the final step was given, which is what a permutation
        importance returns. With ``kind="pattern"`` they are read as a linear
        readout, ``scores = features @ coef.T``.
    kind : {"weights", "pattern"}, default="weights"
        What to return. ``"weights"`` gives one weight per voxel such that the
        model's scores are the voxel maps times the weights, plus a constant.
        ``"pattern"`` gives ``cov(voxels, scores) @ pinv(cov(scores))``,
        computed from ``X``; it needs no step to be undone, so it works through
        any pipeline whose final model is linear in its features.
    atlas : {"voxel", "region"}, default="voxel"
        How weights are read back through a :class:`~nimare.ml.MaskerTransformer`
        that reduced voxels to regions. ``"voxel"`` gives each voxel its own
        weight, the transpose of the reduction: a region's weight divided by its
        size for an averaging labels atlas, and corrected for the overlap of
        the maps for a probabilistic atlas. ``"region"`` paints each region's
        weight over its voxels, which shows the regions but is not a per-voxel
        weight. Ignored with ``kind="pattern"``.
    X : array_like or sparse matrix, optional
        The rows to compute a pattern from, as the pipeline takes them, by
        default None, which uses ``bunch.data``. Pass the voxel block, for
        instance, when that is what the pipeline was fitted on. Only used with
        ``kind="pattern"``.

    Returns
    -------
    :class:`~nibabel.nifti1.Nifti1Image`
        One weight per voxel, in the bunch masker's space. A model with one
        set of weights gives a 3D image, several give a 4D one.

    Raises
    ------
    :obj:`ValueError`
        If ``kind`` or ``atlas`` is not recognised, if no weights can be found,
        if a step cannot be undone, or if the weights or the voxel data do not
        end up one per voxel.

    Notes
    -----
    A model that scores ``w @ (A x + c)`` scores ``(A.T @ w) @ x`` plus a
    constant, so a weight moves back through a linear step by the transpose of
    its linear part, and the step's offset belongs to the intercept. That is not
    what ``inverse_transform`` computes: it maps a point back, ``A^-1 (z - c)``,
    which agrees with the transpose only for an orthogonal step with no offset.
    Through a :class:`~sklearn.preprocessing.StandardScaler` it would give
    ``w * scale + mean`` where the weight is ``w / scale``. The steps read back
    are therefore the ones whose linear part is known: the scikit-learn scalers
    (:class:`~sklearn.preprocessing.StandardScaler`,
    :class:`~sklearn.preprocessing.MaxAbsScaler`,
    :class:`~sklearn.preprocessing.MinMaxScaler`,
    :class:`~sklearn.preprocessing.RobustScaler`),
    :class:`~sklearn.decomposition.PCA`,
    :class:`~sklearn.decomposition.TruncatedSVD`, feature selectors, and a
    :class:`~nimare.ml.MaskerTransformer`, which is undone through the atlas it
    applied. Any other step is refused.

    Through an atlas the same rule applies. A labels atlas that averages has
    ``A[r, v] = 1 / n_r`` over the ``n_r`` voxels of region ``r``, so each voxel
    carries ``w_r / n_r``; one that sums carries ``w_r``. A probabilistic atlas
    fits each map by least squares, ``A = (M M.T)^-1 M``, so the voxel weights
    are ``M.T (M M.T)^-1 w``. A :class:`~nilearn.maskers.NiftiMasker` that only
    smooths is its own transpose. Strategies and signal cleaning that are not
    linear in the voxels are refused.

    A pattern is a covariance in voxel space, so it moves through a step the
    way data do, not the way weights do. Rather than walk the steps for it, the
    pattern is computed where the voxels are: the output of the
    :class:`~nimare.ml.MAKernel`, or ``X`` itself when there is none. The
    scores are the final features times the weights, without the intercept,
    which a covariance does not see.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    nimare.ml.MaskerTransformer : Undone through the atlas it applied.

    Examples
    --------
    >>> image = coefficient_image(pipeline, bunch)  # doctest: +SKIP
    >>> plotting.plot_stat_map(image)  # doctest: +SKIP
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}.")
    if atlas not in ATLAS_READINGS:
        raise ValueError(f"atlas must be one of {ATLAS_READINGS}, not {atlas!r}.")

    steps = list(estimator) if isinstance(estimator, Pipeline) else [estimator]
    model = steps[-1] if coef is None else None
    if kind == "pattern" and model is not None and getattr(model, "coef_", None) is None:
        raise ValueError(
            f"{type(model).__name__} has no 'coef_', and a pattern needs the linear "
            "readout a model scores its features with. Pass that readout as 'coef'."
        )
    weights = _weights(model, coef)

    masker = mask_source(bunch.masker)
    n_voxels = int(np.sum(np.asarray(masker.mask_img.dataobj) > 0))

    if kind == "pattern":
        weights = _pattern(steps, weights, bunch.data if X is None else X, n_voxels)
    else:
        for step in reversed(steps[:-1]):
            weights = _undo(step, weights, bunch, atlas)

    if weights.shape[1] != n_voxels:
        raise ValueError(
            f"The weights came back {weights.shape[1]} wide, but the masker has "
            f"{n_voxels} voxels. Every step between the model and the voxels has to "
            "be undone for a weight to name a place in the brain."
        )

    image = masker.inverse_transform(weights)
    if weights.shape[0] == 1:
        # one set of weights is a map, not a stack of one
        return image.__class__(np.asarray(image.dataobj)[..., 0], image.affine, image.header)
    return image


def _weights(model, coef):
    """Return the weights to project, one row per set."""
    if coef is None:
        coef = getattr(model, "coef_", None)
        if coef is None:
            coef = getattr(model, "feature_importances_", None)
        if coef is None:
            raise ValueError(
                f"{type(model).__name__} has neither 'coef_' nor 'feature_importances_', "
                "so it does not say what weight it gave each feature. Pass the weights "
                "as 'coef', from permutation_importance for instance."
            )
    return np.atleast_2d(np.asarray(coef, dtype=float))


def _undo(step, weights, bunch, atlas="voxel"):
    """Return ``weights`` as they were before ``step`` transformed the features."""
    if isinstance(step, Pipeline):
        for inner in reversed(list(step)):
            weights = _undo(inner, weights, bunch, atlas)
        return weights

    if isinstance(step, ColumnTransformer):
        return _undo(*_voxel_branch(step, weights, bunch), bunch, atlas)

    if isinstance(step, MAKernel):
        # a kernel is where the voxels are: its input is peaks, and a weight
        # over peaks is not a weight over the brain
        return weights

    if isinstance(step, FunctionTransformer):
        # its inverse is the identity unless it was given one, which would pass
        # the weights through as if the function had not been applied
        if step.func not in (None, _to_dense) and step.inverse_func is None:
            raise ValueError(
                "A FunctionTransformer with a 'func' but no 'inverse_func' cannot be "
                "undone: its inverse is the identity, which would report the weights "
                "after it as if they were the weights before it. Give it an "
                "'inverse_func', or project weights you have already undone."
            )
        return weights

    if isinstance(step, MaskerTransformer):
        return _through_atlas(step, weights, atlas)

    pulled = _pull_back(step, weights)
    if pulled is not None:
        return pulled

    raise ValueError(
        f"{type(step).__name__} is not a step whose weights can be read back to the "
        "features before it. A weight moves back through a linear step by the transpose "
        "of that step, which 'inverse_transform' does not compute, so only steps with a "
        "known linear part are undone: StandardScaler, MaxAbsScaler, MinMaxScaler, "
        "RobustScaler, PCA, TruncatedSVD, feature selectors and MaskerTransformer. Drop "
        "it from the pipeline you pass here, or project weights you have already undone."
    )


def _pull_back(step, weights):
    """Return ``weights`` over a linear step's inputs, or None for any other step.

    The transpose of the step's linear part, without its offset: see the Notes of
    :func:`coefficient_image`.
    """
    if isinstance(step, (StandardScaler, MaxAbsScaler, RobustScaler)):
        # x' = (x - offset) / scale; scale_ is None when scaling was switched off
        return weights if step.scale_ is None else weights / step.scale_
    if isinstance(step, MinMaxScaler):
        # x' = x * scale_ + min_
        return weights * step.scale_
    if isinstance(step, PCA):
        # z = (x - mean_) @ components_.T, divided by sqrt(explained_variance_) if whitened
        if step.whiten:
            weights = weights / np.sqrt(step.explained_variance_)
        return weights @ step.components_
    if isinstance(step, TruncatedSVD):
        return weights @ step.components_
    if isinstance(step, SelectorMixin):
        # a selection's transpose puts each weight back in its column and zeros the rest,
        # which is what a selector's inverse_transform already does
        return np.asarray(step.inverse_transform(weights))
    return None


def _through_atlas(step, weights, atlas):
    """Return ``weights`` over the voxels a MaskerTransformer reduced.

    Every reading is the masker's own ``inverse_transform`` of adjusted weights,
    since that is ``w @ M`` for the region (or map) matrix ``M`` it fitted.
    """
    if atlas == "region":
        return np.asarray(step.inverse_transform(weights))

    reducer = step.masker_
    if isinstance(reducer, NiftiLabelsMasker):
        strategy = getattr(reducer, "strategy", "mean")
        if strategy == "sum":
            return np.asarray(step.inverse_transform(weights))
        if strategy != "mean":
            raise ValueError(
                f"A labels atlas with strategy={strategy!r} is not linear in the voxels, "
                "so its regions' weights have no per-voxel reading. Use strategy='mean' "
                "or 'sum', or atlas='region' to paint the region weights."
            )
        sizes = _region_matrix(step).sum(axis=1)
        safe = np.where(sizes > 0, sizes, 1.0)
        return np.asarray(step.inverse_transform(np.where(sizes > 0, weights / safe, 0.0)))

    if isinstance(reducer, NiftiMapsMasker):
        maps = _region_matrix(step)
        gram = maps @ maps.T
        return np.asarray(step.inverse_transform(weights @ np.linalg.pinv(gram)))

    if isinstance(reducer, NiftiMasker):
        cleaning = [name for name in _CLEANING if getattr(reducer, name, None)]
        if cleaning:
            raise ValueError(
                f"A NiftiMasker that applies {', '.join(cleaning)} depends on the other "
                "rows, so it has no fixed linear part to read weights back through."
            )
        if getattr(reducer, "smoothing_fwhm", None):
            # a Gaussian is symmetric, so the smoothing is its own transpose
            return np.asarray(step.transform(weights))
        return weights

    raise ValueError(
        f"Weights cannot be read back per voxel through a {type(reducer).__name__}. "
        "Use atlas='region' to paint the region weights."
    )


def _region_matrix(step):
    """Return the regions-by-voxels matrix a fitted MaskerTransformer reduces with.

    Read from the masker's ``inverse_transform`` of the identity, so it is the
    matrix the masker itself uses, in the space its features come in.
    """
    n_regions = step._n_features_out()
    return np.asarray(step.inverse_transform(np.eye(n_regions)), dtype=float)


def _pattern(steps, weights, X, n_voxels):
    """Return the activation pattern of a linear readout, one row per score."""
    features = X
    for step in steps[:-1]:
        features = step.transform(features)
    features = features.toarray() if sparse.issparse(features) else np.asarray(features)
    scores = features @ weights.T
    scores = scores - scores.mean(axis=0)

    voxels = _voxel_data(steps[:-1], X)
    if voxels.shape[1] != n_voxels:
        raise ValueError(
            f"The voxel data came out {voxels.shape[1]} wide, but the masker has "
            f"{n_voxels} voxels. A pattern is computed after the MAKernel, or from X "
            "itself when the pipeline has none; pass X in the masker's voxel space."
        )

    n = scores.shape[0]
    # the scores are centred, so the voxels need not be: cov(x, s) = x.T s / (n - 1)
    cross = np.asarray(voxels.T @ scores) / (n - 1)
    return (cross @ np.linalg.pinv(scores.T @ scores / (n - 1))).T


def _voxel_data(steps, X):
    """Return ``X`` at the voxel stage: after the last MAKernel, or as given.

    Follows a ColumnTransformer into the branch with a kernel, as the backwards
    walk follows the branch covering the voxels.
    """
    voxels, current = X, X
    for step in steps:
        if isinstance(step, MAKernel):
            voxels = current = step.transform(current)
            continue
        if isinstance(step, (Pipeline, ColumnTransformer)):
            inner = _kernel_branch(step, current)
            if inner is not None:
                return inner
        current = step.transform(current)
    return voxels


def _kernel_branch(step, X):
    """Return the voxel-stage data inside a composite step with a kernel, or None."""
    if isinstance(step, Pipeline):
        if not any(isinstance(inner, MAKernel) for inner in _flatten(step)):
            return None
        return _voxel_data(list(step), X)
    for _, transformer, columns in step.transformers_:
        if isinstance(transformer, str):
            continue
        if any(isinstance(inner, MAKernel) for inner in _flatten(transformer)):
            branch = list(transformer) if isinstance(transformer, Pipeline) else [transformer]
            return _voxel_data(branch, _safe_indexing(X, columns, axis=1))
    return None


def _flatten(step):
    """Yield a step and every step nested inside it."""
    yield step
    if isinstance(step, Pipeline):
        for inner in step:
            yield from _flatten(inner)
    elif isinstance(step, ColumnTransformer):
        for _, transformer, _ in step.transformers_:
            yield from _flatten(transformer)


def _voxel_branch(column_transformer, weights, bunch):
    """Return ``(branch, its share of the weights)`` for the voxel columns."""
    span = bunch.voxel_columns
    for name, transformer, columns in column_transformer.transformers_:
        covered = np.atleast_1d(np.arange(len(bunch.feature_names))[columns])
        if not len(covered) or covered.min() >= span.stop or covered.max() < span.start:
            continue
        output = column_transformer.output_indices_.get(name)
        if output is None or output.stop <= output.start:
            raise ValueError(
                f"The {name!r} step covers the voxel columns but contributes none to "
                "the output, so there are no weights over the voxels to read back."
            )
        return transformer, weights[:, output]

    raise ValueError(
        "No step of this ColumnTransformer covers the voxel columns, so none of the "
        "weights are over the brain."
    )
