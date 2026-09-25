.. include:: links.rst

.. _fetching tools:

Fetching resources from the internet
====================================
:mod:`~nimare.extract`

NiMARE's ``extract`` module contains a number of functions for downloading resources
(e.g., ontologies, images, and datasets) from the internet.

.. topic:: Where do downloaded resources end up?

    The fetching functions in NiMARE use the same approach as ``nilearn``.
    Namely, data fetched using NiMARE's functions will be downloaded to the disk.
    These files will be saved to one of the following directories:

    - the folder specified by ``data_dir`` parameter in the fetching function
    - the global environment variable ``NIMARE_SHARED_DATA``
    - the user environment variable ``NIMARE_DATA``
    - the ``.nimare`` folder in the user home folder

    The two different environment variables (``NIMARE_SHARED_DATA`` and ``NIMARE_DATA``) are provided for multi-user systems,
    to distinguish a global dataset repository that may be read-only at the user-level.
    Note that you can copy that folder to another user's computers to avoid the initial dataset download on the first fetching call.

    You can check in which directory NiMARE will store the data with the function :func:`~nimare.extract.utils.get_data_dirs`.

Where should coordinate data come from?
---------------------------------------

Use :func:`~nimare.extract.fetch_neurostore` to download coordinate data.
It fetches a `NeuroStore studyset release <https://neurostore.org/api/neurostore-studyset-releases/>`_,
which is rebuilt from the live NeuroStore database and returns a :class:`~nimare.nimads.Studyset`::

    from nimare.extract import fetch_neurostore, fetch_neurostore_releases

    # See what is available (dated monthly releases plus a rolling nightly build).
    for release in fetch_neurostore_releases():
        print(release["version"], release["study_count"])

    studyset = fetch_neurostore()  # the most recent dated release

.. warning::

    :func:`~nimare.extract.fetch_neurosynth` and :func:`~nimare.extract.fetch_neuroquery`
    download frozen snapshots of the Neurosynth and NeuroQuery databases. Neurosynth's
    coordinates were extracted in 2018, and its data files were last repackaged in 2021.
    ``fetch_neurosynth`` is deprecated and will be removed in NiMARE 1.0.0; it is kept
    for reproducing published Neurosynth analyses and for the term annotations that
    Neurosynth-based :doc:`decoding <decoding>` needs. New coordinate-based analyses
    should use :func:`~nimare.extract.fetch_neurostore` instead.
