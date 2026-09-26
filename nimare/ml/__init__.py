"""Machine-learning helpers for modeled activation (MA) features.

Converts a :class:`~nimare.nimads.Studyset` into the arrays a scikit-learn
workflow expects: a sparse analysis-by-voxel feature matrix, an optional
target, and the study labels that keep analyses from one study out of two
different partitions. :meth:`FeatureSet.from_studyset` builds the container and
:class:`AtlasAggregator` reduces its voxels over an atlas; every other
reduction is an ordinary scikit-learn transformer.

See :doc:`the machine learning documentation </machine_learning>` for what the
module does and does not take responsibility for.
"""

from nimare.ml.features import FeatureSet
from nimare.ml.reduce import AtlasAggregator

__all__ = [
    "AtlasAggregator",
    "FeatureSet",
]
