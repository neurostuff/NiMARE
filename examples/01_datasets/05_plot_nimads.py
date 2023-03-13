"""

.. _nimads_object:

========================
Using NIMADS with NiMARE
========================

How to use the NeuroImaging Meta-Analysis Data Structure `(NIMADS) <https://neurostuff.github.io/NIMADS/>`_
with NiMARE.
"""

import os.path as op
from nimare.nimads import Studyset
from nimare.utils import get_test_data_path
from nimare.io import convert_nimads_to_dataset



###############################################################################
# Access Data
# -----------------------------------------------------------------------------
nimads_studyset = op.join(get_test_data_path(), "nimads_studyset.json")
nimads_annotation = op.join(get_test_data_path(), "nimads_annotation.json")

###############################################################################
# Load Data
# -----------------------------------------------------------------------------
studyset = Studyset(nimads_studyset, nimads_annotation)
