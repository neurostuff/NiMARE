"""Reading a fitted model back to the brain."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from nimare.ml._helpers import _to_dense
from nimare.ml._peaks import mask_source
from nimare.ml.kernel import MAKernel


def coefficient_image(estimator, bunch, coef=None):
    """Return a fitted model's voxel weights as an image.

    Walks the fitted steps backwards, undoing each reduction, until the weights
    are one per voxel again, and unmasks them. Every step between the model and
    the voxels needs an inverse for a weight to name a place in the brain, and
    a step that has none stops the walk and says so.

    Parameters
    ----------
    estimator : :class:`~sklearn.pipeline.Pipeline` or estimator
        A fitted pipeline ending in a model with ``coef_``, or the model
        itself when the features are already voxels.
    bunch : :class:`sklearn.utils.Bunch`
        The bundle the model was fitted on, for its ``masker`` and the span of
        its voxel columns.
    coef : array_like, optional
        Weights to project instead of the model's own, by default None. One per
        feature the final step was given, which is what a permutation
        importance returns.

    Returns
    -------
    :class:`~nibabel.nifti1.Nifti1Image`
        One weight per voxel, in the bundle masker's space. A model with one
        set of weights gives a 3D image, several give a 4D one.

    Raises
    ------
    :obj:`ValueError`
        If no weights can be found, if a step cannot be undone, or if the
        weights do not end up one per voxel.

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

    if hasattr(step, "inverse_transform"):
        return np.asarray(step.inverse_transform(weights))

    raise ValueError(
        f"{type(step).__name__} has no 'inverse_transform', so the weights after it "
        "cannot be read back to the features before it. Drop it from the pipeline you "
        "pass here, or project weights you have already undone."
    )


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
