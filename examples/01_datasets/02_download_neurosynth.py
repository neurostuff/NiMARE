"""

.. _datasets_databases:

=========================
Neurosynth and NeuroQuery
=========================

Neurosynth and NeuroQuery are the two largest publicly-available coordinate-based databases.
NiMARE includes functions for downloading releases of each database and converting the databases
to NiMARE Datasets.

In this example, we download and convert the Neurosynth and NeuroQuery databases for analysis with
NiMARE.

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
see :doc:`../../fetching`.
"""
###############################################################################
# Start with the necessary imports
# -----------------------------------------------------------------------------
import os
from pprint import pprint

from nimare.extract import download_abstracts, fetch_neuroquery, fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset

###############################################################################
# Download Neurosynth
# -----------------------------------------------------------------------------
# Neurosynth's data files are stored at https://github.com/neurosynth/neurosynth-data.
out_dir = os.path.abspath("../example_data/")
os.makedirs(out_dir, exist_ok=True)

files = fetch_neurosynth(
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
# -----------------------------------------------------------------------------
neurosynth_dset = convert_neurosynth_to_dataset(
    coordinates_file=neurosynth_db["coordinates"],
    metadata_file=neurosynth_db["metadata"],
    annotations_files=neurosynth_db["features"],
)
neurosynth_dset.save(os.path.join(out_dir, "neurosynth_dataset.pkl.gz"))
print(neurosynth_dset)

###############################################################################
# Add article abstracts to dataset
# -----------------------------------------------------------------------------
# This is only possible because Neurosynth uses PMIDs as study IDs.
#
# Make sure you replace the example email address with your own.
neurosynth_dset = download_abstracts(neurosynth_dset, "example@example.edu")
neurosynth_dset.save(os.path.join(out_dir, "neurosynth_dataset_with_abstracts.pkl.gz"))

###############################################################################
# Do the same with NeuroQuery
# -----------------------------------------------------------------------------
# NeuroQuery's data files are stored at https://github.com/neuroquery/neuroquery_data.
files = fetch_neuroquery(
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
neuroquery_dset = convert_neurosynth_to_dataset(
    coordinates_file=neuroquery_db["coordinates"],
    metadata_file=neuroquery_db["metadata"],
    annotations_files=neuroquery_db["features"],
)
neuroquery_dset.save(os.path.join(out_dir, "neuroquery_dataset.pkl.gz"))
print(neuroquery_dset)

# NeuroQuery also uses PMIDs as study IDs.
neuroquery_dset = download_abstracts(neuroquery_dset, "example@example.edu")
neuroquery_dset.save(os.path.join(out_dir, "neuroquery_dataset_with_abstracts.pkl.gz"))
