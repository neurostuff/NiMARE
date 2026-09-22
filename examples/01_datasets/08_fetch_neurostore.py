"""

.. _datasets_neurostore:

===============================
Download NeuroStore coordinates
===============================

`NeuroStore <https://neurostore.org>`_ is the maintained, continuously-updated database of
neuroimaging coordinates behind `Neurosynth Compose <https://compose.neurosynth.org>`_, and it is
where new NiMARE analyses should get their coordinate data.

NeuroStore publishes periodic snapshots of the whole database as
`studyset releases <https://neurostore.org/api/neurostore-studyset-releases/>`_: archives of
parquet tables that load straight into a :class:`~nimare.nimads.Studyset`
(see :ref:`parquet_studyset` for the file layout). :func:`~nimare.extract.fetch_neurostore`
downloads, checksums, and extracts one of those releases for you.

.. note::
    Reading the parquet tables requires ``pyarrow``, which NiMARE ships as an extra::

        pip install nimare[parquet]

.. warning::
    :func:`~nimare.extract.fetch_neurosynth` (see :ref:`datasets_databases`) downloads a frozen
    snapshot of Neurosynth, whose coordinates were extracted in 2018. It is deprecated and will be
    removed in NiMARE 1.0.0. Keep using it only to reproduce published Neurosynth analyses, or to
    get the Neurosynth term annotations that NiMARE's Neurosynth-based decoders need.

For information about where these files will be downloaded to on your machine,
see :doc:`../../fetching`.
"""

###############################################################################
# Start with the necessary imports
# -----------------------------------------------------------------------------
import os

from nimare.extract import fetch_neurostore, fetch_neurostore_releases

###############################################################################
# See which releases are available
# -----------------------------------------------------------------------------
# NeuroStore publishes dated releases (one per month) plus a rolling ``nightly`` build.
# Each entry reports when it was built and how much data it holds.
for release in fetch_neurostore_releases():
    print(
        f"{release['version']:>10} ({release['release_type']}): "
        f"{release['study_count']} studies, built {release['built_at']}"
    )

###############################################################################
# Download the most recent release
# -----------------------------------------------------------------------------
# ``version="latest"`` (the default) takes the most recent dated release. Pass
# ``version="nightly"`` for the rolling build, or an explicit version such as
# ``version="2026-09"`` to pin an analysis to a specific snapshot.
#
# The release is cached on disk, so re-running this is cheap; pass ``overwrite=True``
# to force a fresh download.
out_dir = os.path.abspath("../example_data/")
os.makedirs(out_dir, exist_ok=True)

studyset = fetch_neurostore(data_dir=out_dir)
print(studyset)
print(f"Studyset ID: {studyset.id}")
print(f"Number of studies: {len(studyset.study_ids)}")
print(f"Number of analyses: {len(studyset.ids)}")

###############################################################################
# Inspect the coordinates
# -----------------------------------------------------------------------------
# The Studyset exposes the standard table views, so the release is ready for
# filtering, annotation, and meta-analysis.
print(studyset.coordinates.head())

###############################################################################
# Keep the extracted files instead
# -----------------------------------------------------------------------------
# ``return_type="files"`` skips loading and returns the path to the extracted parquet
# directory, which can be loaded later with ``Studyset(path)``.
release_dir = fetch_neurostore(data_dir=out_dir, return_type="files")
print(release_dir)
print(sorted(os.listdir(release_dir)))

###############################################################################
# Download a single studyset instead of the whole database
# -----------------------------------------------------------------------------
# To fetch one curated studyset (for example, one assembled in Neurosynth Compose)
# rather than a full release, use :func:`~nimare.io.fetch_neurostore_studyset` with the
# studyset's NeuroStore ID.
