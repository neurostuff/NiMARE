"""Machine-learning helpers for modeled activation (MA) features.

:meth:`~nimare.studyset.Studyset.to_bunch` turns a
:class:`~nimare.nimads.Studyset` into the arrays a scikit-learn workflow
expects: a sparse analysis-by-voxel feature matrix, an optional target, and the
study labels that keep analyses from one study out of two different partitions.
This module holds what a Studyset cannot answer on its own:
:func:`describe_fields` reports which of its fields are worth modelling,
:func:`make_nimare_column_transformer` keeps a reducer off the descriptor columns, and
:class:`MaskerTransformer` reduces voxels over an atlas. Every other reduction is
an ordinary scikit-learn transformer.

See :doc:`the machine learning documentation </machine_learning>` for what the
module does and does not take responsibility for.
"""

from nimare.ml.compose import make_nimare_column_transformer
from nimare.ml.extract import describe_fields
from nimare.ml.reduce import MaskerTransformer

__all__ = [
    "MaskerTransformer",
    "describe_fields",
    "make_nimare_column_transformer",
]
