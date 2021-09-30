.. _fetching tools:

.. include:: ../links.rst

Fetching resources from the internet
====================================
:mod:`nimare.extract`

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

    You can check in which directory NiMARE will store the data with the function :func:`nimare.extract.utils.get_data_dirs`.
