"""

.. _metas_cbma:

===================
The Estimator class
===================

An introduction to the Estimator class.
"""
import os

from nimare.dataset import Dataset
from nimare.utils import get_resource_path

###############################################################################
# Load Dataset
# --------------------------------------------------
dset_file = os.path.join(get_resource_path(), "nidm_pain_dset.json")
dset = Dataset(dset_file)
