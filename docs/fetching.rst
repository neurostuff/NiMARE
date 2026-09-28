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

Searching NeuroStore for the studies you want
---------------------------------------------

A release is the whole database. To assemble a studyset from a search instead --
a term, a year range, a region of the brain -- use
:func:`~nimare.extract.search_neurostore_studyset`, which queries the live
NeuroStore API through ``neurostore-sdk``::

    from nimare.extract import search_neurostore_studyset

    studyset = search_neurostore_studyset(
        "working memory",          # substring search over title, abstract, authors, journal
        year_min=2010,
        require_coordinates=True,
        max_results=100,
    )

Any query parameter of the `base-studies endpoint
<https://neurostore.org/api/base-studies/>`_ can be passed as a keyword:
``semantic_search``, ``authors``, ``publication``, ``doi``, ``pmid``,
``data_type``, ``level``, ``is_oa``, ``feature_filter``, or the spatial query
``x``/``y``/``z``/``radius``.

.. _neurostore version heuristics:

Which version of each study?
````````````````````````````

NeuroStore stores one *base study* per paper, and several *versions* of each:
one per source that extracted it (``neurosynth``, ``neuroquery``, ``llm``, ...)
plus any that users curated by hand. They disagree -- different analyses,
different coordinates, different metadata -- and a studyset holds exactly one
study per paper, so which version to take is a choice you make explicitly.

``heuristic`` makes it. It is one name, or several applied in order as
tie-breaks::

    # A hand-curated version if one exists, else an LLM-extracted one,
    # else whichever was updated most recently.
    studyset = search_neurostore_studyset(
        "working memory", heuristic=("user", "llm", "last_updated")
    )

    # Prefer the classic automated corpora, Neurosynth before NeuroQuery.
    studyset = search_neurostore_studyset("insula", heuristic=("neurosynth", "neuroquery"))

    # The same chain as a single string, for a command line or a config file.
    studyset = search_neurostore_studyset("insula", heuristic="user,llm,last_updated")

The names are in :data:`~nimare.extract.VERSION_HEURISTICS`; the default is
:data:`~nimare.extract.DEFAULT_VERSION_HEURISTIC`. A source that is not named
there can be written as ``"source:<name>"``, and any callable
``(version, base_study) -> sort key`` (lower wins) is a heuristic too, so a
project-specific rule needs no change to NiMARE::

    studyset = search_neurostore_studyset(
        "insula", heuristic=[lambda version: 0 if version["username"] == "me" else 1]
    )

Every chain is completed with a deterministic tie-break, so the same query and
the same heuristic always select the same versions. Each study in the resulting
studyset records which base study and version it came from, under the
``neurostore_base_study_id``, ``neurostore_version_id``, ``neurostore_source``
and ``neurostore_username`` metadata keys.

Searching and selecting can also be done separately -- to inspect what matched
before downloading anything, or to select twice from one search::

    from nimare.extract import search_base_studies, studyset_from_base_studies

    base_studies = search_base_studies("working memory", max_results=100)
    print(len(base_studies), "papers")

    curated = studyset_from_base_studies(base_studies, heuristic="user")
    automated = studyset_from_base_studies(base_studies, heuristic=("neurosynth", "neuroquery"))

Only the selected versions are downloaded, one request each. Pass ``n_jobs`` to
download them concurrently, or ``nested=True`` to have the search embed every
version's coordinates in one pass -- fewer requests, a much larger payload.

.. note::

    :func:`~nimare.extract.search_neurostore_studyset` queries the live
    database, so its results change as NeuroStore does. Save the studyset you
    analyzed (:meth:`~nimare.nimads.Studyset.to_parquet` or
    :meth:`~nimare.nimads.Studyset.to_nimads`) if the analysis needs to be
    reproducible, or use a dated release from
    :func:`~nimare.extract.fetch_neurostore`.
