"""

.. _nimads_object:

========================
Using NIMADS with NiMARE
========================

How to use the NeuroImaging Meta-Analysis Data Structure
`(NIMADS) <https://neurostuff.github.io/NIMADS/>`_ with NiMARE.
"""
from requests import request

from nimare.io import convert_nimads_to_dataset
from nimare.nimads import Studyset

###############################################################################
# Download Data from NeuroStore
# -----------------------------------------------------------------------------


def download_file(url):
    """Download a file from NeuroStore."""
    response = request("GET", url)
    return response.json()


nimads_studyset = download_file(
    "https://neurostore.org/api/studysets/Cv2LLUqG76W9?nested=true"
)
nimads_annotation = download_file(
    "https://neurostore.org/api/annotations/76PyNqoTNEsE"
)


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
