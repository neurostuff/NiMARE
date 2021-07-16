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
from pprint import pprint

import nimare

###############################################################################
# Download Neurosynth
# --------------------------------
out_dir = os.path.abspath("../example_data/")
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

files = nimare.extract.fetch_neurosynth(
    path=out_dir,
    version="7",
    overwrite=False,
    source="abstract",
    vocab="terms",
)
pprint(files)
first_database = files[0]

###############################################################################
# Convert Neurosynth database to NiMARE dataset file
# --------------------------------------------------
dset = nimare.io.convert_neurosynth_to_dataset(
    database_file=first_database["database"],
    annotations_files=first_database["features"],
)
dset.save(os.path.join(out_dir, "neurosynth_dataset.pkl.gz"))
print(dset)

###############################################################################
# Add article abstracts to dataset
# --------------------------------
dset = nimare.extract.download_abstracts(dset, "tsalo006@fiu.edu")
dset.save(os.path.join(out_dir, "neurosynth_nimare_with_abstracts.pkl.gz"))
