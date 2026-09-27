"""Machine-learning helpers for Studyset features.

:meth:`~nimare.studyset.Studyset.to_bunch` turns a
:class:`~nimare.nimads.Studyset` into the arrays a scikit-learn workflow
expects. This module holds what a Studyset cannot answer on its own:
:func:`describe_fields` reports which of its fields are worth modelling,
:class:`MAKernel` turns peak columns into modeled activation maps,
:func:`make_nimare_column_transformer` routes each block to its own
transformer, and :class:`MaskerTransformer` reduces voxels over an atlas.
:func:`study_folds` keeps a study out of two folds at once,
:func:`coefficient_image` reads a fitted model back to the brain, and every
other transformation is an ordinary scikit-learn one.

See :doc:`the machine learning documentation </machine_learning>`.
"""

from nimare.ml.compose import make_nimare_column_transformer
from nimare.ml.extract import describe_fields
from nimare.ml.interpret import coefficient_image
from nimare.ml.kernel import MAKernel, clear_map_cache
from nimare.ml.reduce import MaskerTransformer
from nimare.ml.split import study_folds

__all__ = [
    "MAKernel",
    "MaskerTransformer",
    "clear_map_cache",
    "coefficient_image",
    "describe_fields",
    "make_nimare_column_transformer",
    "study_folds",
]
