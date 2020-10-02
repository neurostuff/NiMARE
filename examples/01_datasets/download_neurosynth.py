# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _datasets2:

=============================================
 Download and convert the Neurosynth database
=============================================

Download and convert the Neurosynth database (with abstracts) for analysis with
NiMARE.

.. note::
    This will likely change as we work to shift database querying to a remote
    database, rather than handling it locally with NiMARE.

"""
###############################################################################
# Start with the necessary imports
# --------------------------------
import os

from neurosynth.base.dataset import download

import nimare

###############################################################################
# Download Neurosynth
# --------------------------------
out_dir = os.path.abspath("../example_data/")
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

if not os.path.isfile(os.path.join(out_dir, "database.txt")):
    download(out_dir, unpack=True)

###############################################################################
# Convert Neurosynth database to NiMARE dataset file
# --------------------------------------------------
dset = nimare.io.convert_neurosynth_to_dataset(
    os.path.join(out_dir, "database.txt"), os.path.join(out_dir, "features.txt")
)
dset.save(os.path.join(out_dir, "neurosynth_dataset.pkl.gz"))

###############################################################################
# Add article abstracts to dataset
# --------------------------------
dset = nimare.extract.download_abstracts(dset, "tsalo006@fiu.edu")
dset.save(os.path.join(out_dir, "neurosynth_nimare_with_abstracts.pkl.gz"))
