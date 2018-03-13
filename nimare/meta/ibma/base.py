"""
Image-based meta-analysis estimators
"""
from __future__ import division

import numpy as np
from sklearn.preprocessing import normalize
from scipy import stats

from ..base import MetaEstimator


class IBMAEstimator(MetaEstimator):
    """Base class for image-based meta-analysis methods.
    """
    pass
