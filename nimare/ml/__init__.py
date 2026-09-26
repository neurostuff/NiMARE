"""Machine-learning helpers for modeled activation (MA) features.

:meth:`~nimare.studyset.Studyset.to_bunch` turns a
:class:`~nimare.nimads.Studyset` into the arrays a scikit-learn workflow
expects: a sparse analysis-by-voxel feature matrix, an optional target, and the
study labels that keep analyses from one study out of two different partitions.
This module holds what a Studyset cannot answer on its own:
:func:`describe_fields` reports which of its fields are worth modelling,
:func:`make_preprocessor` keeps a reducer off the descriptor columns, and
:class:`AtlasAggregator` reduces voxels over an atlas. Every other reduction is
an ordinary scikit-learn transformer.

See :doc:`the machine learning documentation </machine_learning>` for what the
module does and does not take responsibility for.
"""

from nimare.ml.extract import describe_fields
from nimare.ml.preprocess import make_preprocessor
from nimare.ml.reduce import AtlasAggregator

__all__ = [
    "AtlasAggregator",
    "describe_fields",
    "make_preprocessor",
]
