"""Reading a fitted model back to the brain."""

from __future__ import annotations

import numpy as np
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

from nimare.ml._helpers import _to_dense
from nimare.ml._peaks import mask_source
from nimare.ml.kernel import MAKernel
from nimare.ml.reduce import MaskerTransformer


def coefficient_image(estimator, bunch, coef=None):
    """Return a fitted model's voxel weights as an image.

    Walks the fitted steps backwards, undoing each reduction, until the weights
    are one per voxel again, and unmasks them. Every step between the model and
    the voxels has to be read back for a weight to name a place in the brain,
    and a step that cannot be stops the walk and says so.

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
        importance returns.

    Returns
    -------
    :class:`~nibabel.nifti1.Nifti1Image`
        One weight per voxel, in the bunch masker's space. A model with one
        set of weights gives a 3D image, several give a 4D one.

    Raises
    ------
    :obj:`ValueError`
        If no weights can be found, if a step cannot be undone, or if the
        weights do not end up one per voxel.

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

    See Also
    --------
    nimare.ml.MaskerTransformer : Undone through the atlas it applied.

    Examples
    --------
    >>> image = coefficient_image(pipeline, bunch)  # doctest: +SKIP
    >>> plotting.plot_stat_map(image)  # doctest: +SKIP
    """
    steps = list(estimator) if isinstance(estimator, Pipeline) else [estimator]
    weights = _weights(steps[-1] if coef is None else None, coef)

    for step in reversed(steps[:-1]):
        weights = _undo(step, weights, bunch)

    masker = mask_source(bunch.masker)
    n_voxels = int(np.sum(np.asarray(masker.mask_img.dataobj) > 0))
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


def _undo(step, weights, bunch):
    """Return ``weights`` as they were before ``step`` transformed the features."""
    if isinstance(step, Pipeline):
        for inner in reversed(list(step)):
            weights = _undo(inner, weights, bunch)
        return weights

    if isinstance(step, ColumnTransformer):
        return _undo(*_voxel_branch(step, weights, bunch), bunch)

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
        return np.asarray(step.inverse_transform(weights))

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
