"""

.. _datasets_databases:

=========================
Neurosynth and NeuroQuery
=========================

Neurosynth and NeuroQuery are the two largest publicly-available coordinate-based databases.
NiMARE includes functions for downloading releases of each database and converting the databases
to NiMARE collections for analysis.

In this example, we download and convert the Neurosynth and NeuroQuery databases for analysis with
NiMARE.

For most Neurosynth term-based workflows, including the decoding examples in NiMARE, you should
download only the abstract-derived term annotations by passing ``source="abstract"`` and
``vocab="terms"``. Leaving these selectors unset downloads every available annotation set for the
release.

The selector keywords determine which annotation files are downloaded:

+--------------+---------------------------------------------------------------+
| Keyword      | Meaning                                                       |
+==============+===============================================================+
| ``source``   | Text source used to generate the annotations. Neurosynth      |
|              | currently provides ``"abstract"``.                            |
+--------------+---------------------------------------------------------------+
| ``vocab``    | Annotation vocabulary. ``"terms"`` selects term-level tf-idf |
|              | features, while ``"LDA50"``, ``"LDA100"``, ``"LDA200"``, and  |
|              | ``"LDA400"`` select topic-model vocabularies.                 |
+--------------+---------------------------------------------------------------+
| ``type``     | Feature representation. ``"tfidf"`` is used for ``"terms"``, |
|              | while ``"weight"`` is used for the LDA vocabularies.          |
+--------------+---------------------------------------------------------------+

Only the combinations below are valid for Neurosynth:

+--------------+------------+------------+-------------------------------+
| version      | source     | vocab      | type                          |
+==============+============+============+===============================+
| ``3``-``5``  | abstract   | terms      | tfidf                         |
+--------------+------------+------------+-------------------------------+
| ``6``-``7``  | abstract   | terms      | tfidf                         |
+--------------+------------+------------+-------------------------------+
| ``6``-``7``  | abstract   | LDA50      | weight                        |
+--------------+------------+------------+-------------------------------+
| ``6``-``7``  | abstract   | LDA100     | weight                        |
+--------------+------------+------------+-------------------------------+
| ``6``-``7``  | abstract   | LDA200     | weight                        |
+--------------+------------+------------+-------------------------------+
| ``6``-``7``  | abstract   | LDA400     | weight                        |
+--------------+------------+------------+-------------------------------+

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

# biopython is unnecessary here, but is required by download_abstracts.
# We import it here only to document the dependency and cause an early failure if it's missing.
import Bio  # pip install biopython

###############################################################################
# Download Neurosynth
# -----------------------------------------------------------------------------
# Neurosynth's data files are stored at https://github.com/neurosynth/neurosynth-data.
# For term-based workflows, use the abstract-derived term annotations instead of downloading every
# annotation set in the release.
out_dir = os.path.abspath("../example_data/")
os.makedirs(out_dir, exist_ok=True)

files = fetch_neurosynth(
    data_dir=out_dir,
    version="7",
    overwrite=False,
    source="abstract",
    vocab="terms",
    return_type="files",
)
# Note that the files are saved to a new folder within "out_dir" named "neurosynth".
pprint(files)

###############################################################################
# Download Neurosynth directly as a Studyset
# -----------------------------------------------------------------------------
# Studysets are now the default return type. For the legacy Dataset return type,
# pass ``return_type="dataset"``.
neurosynth_studyset = fetch_neurosynth(
    data_dir=out_dir,
    version="7",
    overwrite=False,
    source="abstract",
    vocab="terms",
)[0]

###############################################################################
# Add article abstracts to Studyset
# -----------------------------------------------------------------------------
# This is only possible because Neurosynth uses PMIDs as study IDs.
#
# Make sure you replace the example email address with your own.
neurosynth_studyset = download_abstracts(neurosynth_studyset, "example@example.edu")
neurosynth_studyset.to_nimads(os.path.join(out_dir, "neurosynth_studyset.json"))
print(neurosynth_studyset)

###############################################################################
# Do the same with NeuroQuery
# -----------------------------------------------------------------------------
# NeuroQuery's data files are stored at https://github.com/neuroquery/neuroquery_data.
files = fetch_neuroquery(
    data_dir=out_dir,
    version="1",
    overwrite=False,
    source="combined",
    vocab="neuroquery6308",
    type="tfidf",
    return_type="files",
)
# Note that the files are saved to a new folder within "out_dir" named "neuroquery".
pprint(files)

neuroquery_studyset = fetch_neuroquery(
    data_dir=out_dir,
    version="1",
    overwrite=False,
    source="combined",
    vocab="neuroquery6308",
    type="tfidf",
)[0]

# NeuroQuery also uses PMIDs as study IDs.
neuroquery_studyset = download_abstracts(neuroquery_studyset, "example@example.edu")
neuroquery_studyset.to_nimads(os.path.join(out_dir, "neuroquery_studyset.json"))
print(neuroquery_studyset)
