"""

.. _nimads_object:

========================
Using NIMADS with NiMARE
========================

How to use the NeuroImaging Meta-Analysis Data Structure
`(NIMADS) <https://neurostuff.github.io/NIMADS/>`_ with NiMARE.
"""
import os.path as op

from nimare.io import convert_nimads_to_dataset
from nimare.nimads import Studyset
from nimare.tests.utils import get_test_data_path

###############################################################################
# Access Data
# -----------------------------------------------------------------------------

nimads_studyset = op.join(get_test_data_path(), "nimads_studyset.json")
nimads_annotation = op.join(get_test_data_path(), "nimads_annotation.json")

###############################################################################
# Load Data
# -----------------------------------------------------------------------------
# Load the json files into a NiMADS Studyset object.

studyset = Studyset(nimads_studyset, nimads_annotation)


###############################################################################
# Convert to NiMARE Dataset
# -----------------------------------------------------------------------------
# Convert the NiMADS Studyset object to a NiMARE Dataset object.
# Then you can run NiMARE analyses on the Dataset object.

nimare_dset = studyset.to_dataset()
nimare_dset.coordinates.head()

###############################################################################
# Directly to NiMARE Dataset
# -----------------------------------------------------------------------------
# Alternatively, you can convert the NiMADS json files directly to a NiMARE Dataset object
# if you wish to skip using the nimads studyset object directly.

nimare_dset_2 = convert_nimads_to_dataset(nimads_studyset, nimads_annotation)
nimare_dset_2.coordinates.head()
