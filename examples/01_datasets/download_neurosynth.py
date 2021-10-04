# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _datasets2:

================================================
 Download the Neurosynth or NeuroQuery databases
================================================

Download and convert the Neurosynth database (with abstracts) for analysis with NiMARE.

.. warning::
    In August 2021, the Neurosynth database was reorganized according to a new file format.
    As such, the ``fetch_neurosynth`` function for NiMARE versions before 0.0.10 will not work
    with its default parameters.
    In order to download the Neurosynth database in its older format using NiMARE <= 0.0.9,
    do the following::
        nimare.extract.fetch_neurosynth(
            url=(
                "https://github.com/neurosynth/neurosynth-data/blob/"
                "e8f27c4a9a44dbfbc0750366166ad2ba34ac72d6/current_data.tar.gz?raw=true"
            ),
        )

For information about where these files will be downloaded to on your machine,
see :ref:`Fetching resources from the internet <fetching tools>`.

"""
###############################################################################
# Start with the necessary imports
# --------------------------------
import os
from pprint import pprint

import nimare

###############################################################################
# Download Neurosynth
# -------------------
# Neurosynth's data files are stored at https://github.com/neurosynth/neurosynth-data.
out_dir = os.path.abspath("../example_data/")
os.makedirs(out_dir, exist_ok=True)

files = nimare.extract.fetch_neurosynth(
    data_dir=out_dir,
    version="7",
    overwrite=False,
    source="abstract",
    vocab="terms",
)
# Note that the files are saved to a new folder within "out_dir" named "neurosynth".
pprint(files)
neurosynth_db = files[0]

###############################################################################
# Convert Neurosynth database to NiMARE dataset file
# --------------------------------------------------
neurosynth_dset = nimare.io.convert_neurosynth_to_dataset(
    coordinates_file=neurosynth_db["coordinates"],
    metadata_file=neurosynth_db["metadata"],
    annotations_files=neurosynth_db["features"],
)
neurosynth_dset.save(os.path.join(out_dir, "neurosynth_dataset.pkl.gz"))
print(neurosynth_dset)

###############################################################################
# Add article abstracts to dataset
# --------------------------------
# This is only possible because Neurosynth uses PMIDs as study IDs.
#
# Make sure you replace the example email address with your own.
neurosynth_dset = nimare.extract.download_abstracts(neurosynth_dset, "example@example.edu")
neurosynth_dset.save(os.path.join(out_dir, "neurosynth_dataset_with_abstracts.pkl.gz"))

###############################################################################
# Do the same with NeuroQuery
# ---------------------------
# NeuroQuery's data files are stored at https://github.com/neuroquery/neuroquery_data.
files = nimare.extract.fetch_neuroquery(
    data_dir=out_dir,
    version="1",
    overwrite=False,
    source="combined",
    vocab="neuroquery7547",
    type="tfidf",
)
# Note that the files are saved to a new folder within "out_dir" named "neuroquery".
pprint(files)
neuroquery_db = files[0]

# Note that the conversion function says "neurosynth".
# This is just for backwards compatibility.
neuroquery_dset = nimare.io.convert_neurosynth_to_dataset(
    coordinates_file=neuroquery_db["coordinates"],
    metadata_file=neuroquery_db["metadata"],
    annotations_files=neuroquery_db["features"],
)
neuroquery_dset.save(os.path.join(out_dir, "neuroquery_dataset.pkl.gz"))
print(neuroquery_dset)

# NeuroQuery also uses PMIDs as study IDs.
neuroquery_dset = nimare.extract.download_abstracts(neuroquery_dset, "example@example.edu")
neuroquery_dset.save(os.path.join(out_dir, "neuroquery_dataset_with_abstracts.pkl.gz"))
