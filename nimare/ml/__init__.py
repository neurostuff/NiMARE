"""Machine-learning helpers for Studyset features.

:meth:`~nimare.studyset.Studyset.to_bunch` turns a
:class:`~nimare.nimads.Studyset` into the arrays a scikit-learn workflow
expects. This module holds what a Studyset cannot answer on its own:
:func:`describe_fields` reports which of its fields are worth modelling,
:class:`MAKernel` turns peak columns into modeled activation maps,
:func:`make_nimare_column_transformer` routes each block to its own
transformer, and :class:`MaskerTransformer` reduces voxels over an atlas.
Every other transformation is an ordinary scikit-learn one.

See :doc:`the machine learning documentation </machine_learning>`.
"""

from nimare.ml.compose import make_nimare_column_transformer
from nimare.ml.extract import describe_fields
from nimare.ml.kernel import MAKernel
from nimare.ml.reduce import MaskerTransformer

__all__ = [
    "MAKernel",
    "MaskerTransformer",
    "describe_fields",
    "make_nimare_column_transformer",
]
