# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _datasets1:

===================================================
 Downloading and converting the Neurosynth database
===================================================

Download and convert the Neurosynth database for analysis with NiMARE.

..note::
    This will likely change as we work to shift database querying to a remote
    database, rather than handling it locally with NiMARE.

"""
###############################################################################
# Start with the necessary imports
# --------------------------------
import os.path as op
from os import mkdir
from neurosynth.base.dataset import download

from nimare.io import convert_neurosynth_to_dataset
from nimare.dataset import Dataset

###############################################################################
# Download Neurosynth
# --------------------------------
out_dir = op.abspath('../example_data/')
if not op.isdir(out_dir):
    mkdir(out_dir)

if not op.isfile(op.join(out_dir, 'database.txt')):
    download(out_dir, unpack=True)

###############################################################################
# Convert Neurosynth database to NiMARE dataset file
# --------------------------------------------------
dset = convert_neurosynth_to_dataset(op.join(out_dir, 'database.txt'),
                                     op.join(out_dir, 'features.txt'))
gz_file = op.join(out_dir, 'neurosynth_dataset.pkl.gz')
dset.save(gz_file)
