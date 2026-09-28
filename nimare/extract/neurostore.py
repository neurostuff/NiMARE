"""Search NeuroStore base studies and assemble them into a local studyset.

A NeuroStore *base study* is one paper. Each base study holds several *versions*
of that paper -- one per source that extracted it (``neurosynth``, ``neuroquery``,
``llm``, ...) plus any a user curated by hand -- and the versions disagree: they
carry different analyses, different coordinates and different metadata. A
studyset holds exactly one study per paper, so searching NeuroStore is two
decisions rather than one: *which papers*, and *which version of each paper*.

The first is a query (:func:`search_base_studies`), the second a heuristic
(:func:`select_base_study_version`). Heuristics are named, ordered and
composable: ``("user", "llm", "last_updated")`` reads as "a hand-curated version
if there is one, otherwise an LLM-extracted one, otherwise whichever was updated
most recently". Every chain is completed with a deterministic tie-break, so the
same query and the same heuristic always pick the same versions.

:func:`search_neurostore_studyset` does both in one call.
"""

from __future__ import annotations

import inspect
import logging
from datetime import datetime, timezone

from tqdm.auto import tqdm

LGR = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_VERSION_HEURISTIC",
    "VERSION_HEURISTICS",
    "base_studies_to_nimads_dict",
    "prefer_source",
    "register_version_heuristic",
    "resolve_version_heuristic",
    "search_base_studies",
    "search_neurostore_studyset",
    "select_base_study_version",
    "studyset_from_base_studies",
]

#: Largest ``page_size`` the NeuroStore API accepts.
MAX_PAGE_SIZE = 29999

#: Metadata keys :func:`base_studies_to_nimads_dict` records on every study it emits.
PROVENANCE_KEYS = (
    "neurostore_base_study_id",
    "neurostore_version_id",
    "neurostore_source",
    "neurostore_username",
)

# Study attributes a version may be missing while its base study has them: base
# studies are enriched (DOI lookup, PubMed) independently of their versions.
_INHERITED_ATTRS = (
    "name",
    "description",
    "publication",
    "doi",
    "pmid",
    "pmcid",
    "authors",
    "year",
    "level",
)


# ---------------------------------------------------------------------------
# the API client
# ---------------------------------------------------------------------------
def _store_api(api=None, host=None):
    """Return a NeuroStore ``StoreApi``, and the ``ApiClient`` used to serialize.

    Parameters
    ----------
    api : :obj:`neurostore_sdk.StoreApi`, optional
        An existing client to use, so that a caller can supply credentials or a
        cassette-backed double. Built against the default host when omitted.
    host : :obj:`str`, optional
        Base URL of a NeuroStore API to talk to instead of the SDK default,
        e.g. a staging deployment. Ignored when ``api`` is given.
    """
    from neurostore_sdk import ApiClient, Configuration, StoreApi

    if api is not None:
        return api, ApiClient()
    if host is not None:
        client = ApiClient(Configuration(host=host))
        return StoreApi(client), client
    return StoreApi(), ApiClient()


def _search_params():
    """Return the query parameters ``base_studies_get`` accepts."""
    from neurostore_sdk import StoreApi

    return frozenset(
        name
        for name in inspect.signature(StoreApi.base_studies_get).parameters
        if name != "self" and not name.startswith("_")
    )


# ---------------------------------------------------------------------------
# heuristics
# ---------------------------------------------------------------------------
def _timestamp(value):
    """Return an ISO-8601 string as a POSIX timestamp, or None if unparsable."""
    if isinstance(value, datetime):
        stamp = value
    elif isinstance(value, str) and value:
        text = value.strip().replace("Z", "+00:00")
        try:
            stamp = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.timestamp()


def _time_key(value, newest_first):
    """Return a sort key over a timestamp; a missing timestamp always sorts last."""
    stamp = _timestamp(value)
    if stamp is None:
        return (1, 0.0)
    return (0, -stamp if newest_first else stamp)


def prefer_source(*sources):
    """Return a heuristic that prefers versions from the named sources, in order.

    Parameters
    ----------
    *sources : :obj:`str`
        Source names as NeuroStore records them, most preferred first, e.g.
        ``prefer_source("llm", "neurosynth")``. Matching is case-insensitive.
        A version from an unnamed source sorts after all of them.

    Returns
    -------
    heuristic : :obj:`callable`
        Takes ``(version, base_study)`` and returns a sort key, lower being
        preferred -- the shape :func:`select_base_study_version` expects.

    Examples
    --------
    >>> heuristic = prefer_source("llm")
    >>> heuristic({"source": "llm"}, None) < heuristic({"source": "neurosynth"}, None)
    True
    """
    order = {str(source).lower(): rank for rank, source in enumerate(sources)}

    def heuristic(version, base_study=None):
        source = version.get("source")
        source = str(source).lower() if source else ""
        return order.get(source, len(order))

    heuristic.__name__ = "prefer_source_" + "_".join(str(s) for s in sources)
    heuristic.__doc__ = f"Prefer versions whose source is one of {sources!r}, in that order."
    return heuristic


def _prefer_user(version, base_study=None):
    """Prefer a version someone owns, i.e. one curated in NeuroStore by hand."""
    return 0 if version.get("user") else 1


def _prefer_coordinates(version, base_study=None):
    """Prefer a version that actually has coordinates."""
    return 0 if version.get("has_coordinates") else 1


def _prefer_images(version, base_study=None):
    """Prefer a version that has images."""
    return 0 if version.get("has_images") else 1


def _last_updated(version, base_study=None):
    """Prefer the version updated most recently."""
    return _time_key(version.get("updated_at"), newest_first=True)


def _first_updated(version, base_study=None):
    """Prefer the version updated longest ago."""
    return _time_key(version.get("updated_at"), newest_first=False)


def _last_created(version, base_study=None):
    """Prefer the version created most recently."""
    return _time_key(version.get("created_at"), newest_first=True)


def _first_created(version, base_study=None):
    """Prefer the version created longest ago."""
    return _time_key(version.get("created_at"), newest_first=False)


#: The named heuristics, usable as strings anywhere a heuristic is accepted.
#:
#: ============================  ==================================================
#: name                          prefers
#: ============================  ==================================================
#: ``"user"``                    versions owned by a user, i.e. curated by hand
#: ``"llm"``                     versions extracted by the LLM pipeline
#: ``"neurosynth"``              versions from the Neurosynth corpus
#: ``"neuroquery"``              versions from the NeuroQuery corpus
#: ``"neurostore"``              versions entered through NeuroStore itself
#: ``"neurovault"``              versions ingested from NeuroVault
#: ``"pubget"``, ``"ace"``       versions from those text-extraction pipelines
#: ``"coordinates"``             versions that have coordinates
#: ``"images"``                  versions that have images
#: ``"last_updated"``            the most recently updated version (``"newest"``)
#: ``"first_updated"``           the least recently updated version
#: ``"last_created"``            the most recently created version
#: ``"first_created"``           the oldest version (``"oldest"``)
#: ============================  ==================================================
#:
#: Anything not listed can be written as ``"source:<name>"``, or as
#: ``"source:<name>|<name>"`` for several sources in preference order.
VERSION_HEURISTICS = {
    "user": _prefer_user,
    "llm": prefer_source("llm"),
    "neurosynth": prefer_source("neurosynth"),
    "neuroquery": prefer_source("neuroquery"),
    "neurostore": prefer_source("neurostore"),
    "neurovault": prefer_source("neurovault"),
    "pubget": prefer_source("pubget"),
    "ace": prefer_source("ace"),
    "coordinates": _prefer_coordinates,
    "images": _prefer_images,
    "last_updated": _last_updated,
    "newest": _last_updated,
    "first_updated": _first_updated,
    "last_created": _last_created,
    "first_created": _first_created,
    "oldest": _first_created,
}

#: Heuristic chain used when none is given: a version that has coordinates, then
#: one a user curated, then the most recently updated.
DEFAULT_VERSION_HEURISTIC = ("coordinates", "user", "last_updated")


def register_version_heuristic(name, heuristic):
    """Register a heuristic under a name, so it can be used as a string.

    Parameters
    ----------
    name : :obj:`str`
        Name to register under. Re-registering an existing name is an error, so
        that a plugin cannot silently change what a built-in name means.
    heuristic : :obj:`callable`
        Takes ``(version, base_study)`` -- or just ``(version,)`` -- and returns
        a sort key, lower being preferred.

    Returns
    -------
    heuristic : :obj:`callable`
        The heuristic, so this can be used as a decorator.
    """
    key = str(name)
    if key in VERSION_HEURISTICS:
        raise ValueError(f"A version heuristic named '{key}' is already registered.")
    if not callable(heuristic):
        raise TypeError(f"A version heuristic must be callable, not {type(heuristic)}.")
    VERSION_HEURISTICS[key] = heuristic
    return heuristic


def _adapt(heuristic):
    """Allow a heuristic to take only the version, and ignore the base study."""
    try:
        parameters = inspect.signature(heuristic).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return heuristic
    if any(p.kind is p.VAR_POSITIONAL for p in parameters.values()):
        return heuristic
    positional = [
        p for p in parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) >= 2:
        return heuristic

    def adapted(version, base_study=None):
        return heuristic(version)

    return adapted


def _named_heuristic(name):
    """Look up one heuristic name, including the ``source:`` escape hatch."""
    key = name.strip()
    if key in VERSION_HEURISTICS:
        return VERSION_HEURISTICS[key]
    if key.lower().startswith("source:"):
        sources = [part.strip() for part in key.split(":", 1)[1].split("|") if part.strip()]
        if not sources:
            raise ValueError(f"'{name}' names no source. Write it as 'source:neurosynth'.")
        return prefer_source(*sources)
    known = ", ".join(sorted(VERSION_HEURISTICS))
    raise ValueError(
        f"Unknown version heuristic '{name}'. Known heuristics are: {known}. "
        "An unlisted source can be written as 'source:<name>'."
    )


def resolve_version_heuristic(heuristic=None):
    """Resolve a heuristic specification into a tuple of callables.

    Parameters
    ----------
    heuristic : :obj:`str`, :obj:`callable`, sequence, or None
        One heuristic, or several to apply in order as tie-breaks. A string may
        chain several names with commas (``"user,llm,last_updated"``), which is
        what makes a heuristic expressible on a command line. ``None`` selects
        :data:`DEFAULT_VERSION_HEURISTIC`.

    Returns
    -------
    heuristics : :obj:`tuple` of :obj:`callable`
        The chain, with a deterministic tie-break appended: the most recently
        updated version, then the lowest version ID. Two runs of the same query
        therefore select the same versions.
    """
    if heuristic is None:
        heuristic = DEFAULT_VERSION_HEURISTIC
    if callable(heuristic) or isinstance(heuristic, str):
        heuristic = [heuristic]

    chain = []
    for item in heuristic:
        if callable(item):
            chain.append(_adapt(item))
        elif isinstance(item, str):
            for name in item.split(","):
                if name.strip():
                    chain.append(_adapt(_named_heuristic(name)))
        else:
            raise TypeError(
                f"A version heuristic must be a string or a callable, not {type(item)}."
            )
    chain.append(_last_updated)
    chain.append(lambda version, base_study=None: str(version.get("id") or ""))
    return tuple(chain)


# ---------------------------------------------------------------------------
# versions
# ---------------------------------------------------------------------------
def _version_dicts(base_study):
    """Return a base study's versions as dictionaries.

    NeuroStore returns versions as bare IDs unless ``info`` or ``nested`` was
    requested. Those carry nothing to choose on, so they are an error here rather
    than a silently arbitrary pick.
    """
    versions = base_study.get("versions") or []
    dicts = [version for version in versions if isinstance(version, dict)]
    if versions and not dicts:
        raise ValueError(
            f"Base study '{base_study.get('id')}' lists its versions as IDs, which carry "
            "nothing to choose between them. Search with info=True (the default) or "
            "nested=True."
        )
    return dicts


def _has_embedded_analyses(version):
    """Return whether a version already carries its analyses, foci and all.

    A nested search embeds ``analyses`` as objects; a version with no analyses
    embeds an empty list, which is still an answer and needs no download. An
    ``info`` search omits the key, or carries bare analysis IDs.
    """
    analyses = version.get("analyses")
    if not isinstance(analyses, list):
        return False
    return all(isinstance(analysis, dict) for analysis in analyses)


def select_base_study_version(base_study, heuristic=None, version_filter=None):
    """Choose which version of a base study belongs in a studyset.

    .. versionadded:: 0.22.0

    Parameters
    ----------
    base_study : :obj:`dict`
        A base study as NeuroStore returns it, with its ``versions`` embedded --
        that is, from a search made with ``info=True`` or ``nested=True``.
    heuristic : :obj:`str`, :obj:`callable`, sequence, or None, optional
        What to prefer, resolved by :func:`resolve_version_heuristic`. Default is
        :data:`DEFAULT_VERSION_HEURISTIC`.
    version_filter : :obj:`callable`, optional
        Takes a version and returns whether it may be selected at all, e.g.
        ``lambda v: v.get("has_coordinates")``. A base study whose every version
        is filtered out has no selection.

    Returns
    -------
    version : :obj:`dict` or None
        The selected version, or ``None`` when the base study has no version to
        select from.

    Examples
    --------
    >>> base_study = {
    ...     "id": "abc",
    ...     "versions": [
    ...         {"id": "v1", "source": "neurosynth", "updated_at": "2020-01-01T00:00:00+00:00"},
    ...         {"id": "v2", "source": None, "user": "someone",
    ...          "updated_at": "2019-01-01T00:00:00+00:00"},
    ...     ],
    ... }
    >>> select_base_study_version(base_study, heuristic="user")["id"]
    'v2'
    >>> select_base_study_version(base_study, heuristic="last_updated")["id"]
    'v1'
    """
    versions = _version_dicts(base_study)
    if version_filter is not None:
        versions = [version for version in versions if version_filter(version)]
    if not versions:
        return None
    chain = resolve_version_heuristic(heuristic)
    return min(versions, key=lambda version: tuple(f(version, base_study) for f in chain))


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------
def search_base_studies(
    query=None,
    *,
    page_size=100,
    max_results=None,
    nested=False,
    api=None,
    host=None,
    timeout=None,
    **filters,
):
    """Search NeuroStore base studies.

    .. versionadded:: 0.22.0

    Parameters
    ----------
    query : :obj:`str`, optional
        Substring search across the indexed study fields -- title, abstract,
        authors, journal. This is the plain string search of the NeuroStore UI.
        Omit it to search by filters alone.
    page_size : :obj:`int`, default=100
        Results requested per API call. Only affects how many round trips the
        search takes, not what it returns.
    max_results : :obj:`int`, optional
        Stop after this many base studies. Default is to return every match,
        which for a broad query can be tens of thousands.
    nested : :obj:`bool`, default=False
        Request each version's analyses, foci and images inline. Turns the whole
        search into one pass, at a much larger payload per study; leave it off
        and :func:`studyset_from_base_studies` downloads only the versions it
        selects. Incompatible with the ``info`` filter.
    api : :obj:`neurostore_sdk.StoreApi`, optional
        Client to search with, for credentials or a test double.
    host : :obj:`str`, optional
        A NeuroStore API to query instead of the SDK default.
    timeout : :obj:`float` or :obj:`tuple`, optional
        Request timeout, in seconds, passed to the SDK.
    **filters
        Any other query parameter of the base-studies endpoint, e.g.
        ``semantic_search``, ``year_min``, ``year_max``, ``authors``,
        ``publication``, ``pmid``, ``doi``, ``level``, ``data_type``,
        ``is_oa``, ``feature_filter``, ``sort``, ``desc``, or the spatial query
        ``x``/``y``/``z``/``radius``. An unknown parameter is an error rather
        than a silently ignored filter.

    Returns
    -------
    base_studies : :obj:`list` of :obj:`dict`
        One dictionary per matching base study, each with its ``versions``
        embedded, ready for :func:`select_base_study_version`.

    Examples
    --------
    >>> from nimare.extract import search_base_studies
    >>> base_studies = search_base_studies(
    ...     "working memory", year_min=2010, max_results=50
    ... )  # doctest: +SKIP

    See Also
    --------
    search_neurostore_studyset : Search and build a studyset in one call.
    """
    from neurostore_sdk import ApiException

    if max_results is not None and max_results <= 0:
        return []
    if not 1 <= page_size <= MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {MAX_PAGE_SIZE}, not {page_size}.")

    unknown = sorted(set(filters) - _search_params())
    if unknown:
        raise TypeError(
            f"Unknown NeuroStore base-study search parameter(s): {', '.join(unknown)}."
        )
    reserved = sorted({"page", "page_size", "nested", "paginate", "flat"} & set(filters))
    if reserved:
        raise TypeError(
            f"{', '.join(reserved)} cannot be passed as a filter, because this function "
            "drives pagination itself and needs each base study's versions embedded. "
            "page_size and nested are its own keywords."
        )
    if query is not None and "search" in filters:
        raise TypeError("Pass the substring search as 'query', not as 'search'.")
    if nested and filters.get("info"):
        raise ValueError("nested and info cannot both be requested; they are incompatible.")

    params = dict(filters)
    if query is not None:
        params["search"] = query
    if nested:
        params["nested"] = True
    else:
        params.setdefault("info", True)
    if timeout is not None:
        params["_request_timeout"] = timeout
    if max_results is not None:
        page_size = min(page_size, max_results)

    store_api, client = _store_api(api=api, host=host)

    base_studies = []
    total = None
    page = 1
    while True:
        try:
            response = store_api.base_studies_get(page=page, page_size=page_size, **params)
        except ApiException as exc:
            raise ValueError(
                f"Failed to search NeuroStore base studies (page {page} of the query "
                f"{params!r})."
            ) from exc

        payload = client.sanitize_for_serialization(response)
        results = payload.get("results") or []
        if total is None:
            total = (payload.get("metadata") or {}).get("total_count")
            LGR.info(
                f"NeuroStore base-study search matched "
                f"{total if total is not None else 'an unknown number of'} studies."
            )
        base_studies.extend(results)

        if max_results is not None and len(base_studies) >= max_results:
            return base_studies[:max_results]
        if not results or len(results) < page_size:
            return base_studies
        if total is not None and len(base_studies) >= total:
            return base_studies
        page += 1


# ---------------------------------------------------------------------------
# studyset assembly
# ---------------------------------------------------------------------------
def _fetch_version(store_api, client, version_id, timeout=None):
    """Download one study version from NeuroStore, analyses and foci included."""
    from neurostore_sdk import ApiException

    kwargs = {"nested": True}
    if timeout is not None:
        kwargs["_request_timeout"] = timeout
    try:
        response = store_api.studies_id_get(version_id, **kwargs)
    except ApiException as exc:
        raise ValueError(f"Failed to download NeuroStore study version '{version_id}'.") from exc
    return client.sanitize_for_serialization(response)


def _is_missing(value):
    """Return whether a study attribute carries nothing usable."""
    return value is None or (isinstance(value, str) and not value.strip())


def _study_payload(version, base_study, inherit=True, provenance=True):
    """Return the NIMADS study to emit for a selected version."""
    study = dict(version)
    study.pop("versions", None)

    if inherit:
        for name in _INHERITED_ATTRS:
            if _is_missing(study.get(name)) and not _is_missing(base_study.get(name)):
                study[name] = base_study[name]

    if provenance:
        metadata = dict(study.get("metadata") or {})
        metadata["neurostore_base_study_id"] = base_study.get("id")
        metadata["neurostore_version_id"] = version.get("id")
        metadata["neurostore_source"] = version.get("source")
        metadata["neurostore_username"] = version.get("username")
        study["metadata"] = metadata

    return study


def base_studies_to_nimads_dict(
    base_studies,
    *,
    heuristic=None,
    version_filter=None,
    require_coordinates=False,
    studyset_id="neurostore-search",
    name="NeuroStore base-study search",
    inherit_base_study_metadata=True,
    add_provenance=True,
    api=None,
    host=None,
    timeout=None,
    n_jobs=1,
    verbose=True,
):
    """Select one version per base study and return them as a NIMADS studyset.

    .. versionadded:: 0.22.0

    Parameters
    ----------
    base_studies : :obj:`list` of :obj:`dict`
        Base studies from :func:`search_base_studies`, with versions embedded.
    heuristic : :obj:`str`, :obj:`callable`, sequence, or None, optional
        Which version of each base study to take. See
        :func:`resolve_version_heuristic`; default is
        :data:`DEFAULT_VERSION_HEURISTIC`.
    version_filter : :obj:`callable`, optional
        Restricts which versions may be selected. See
        :func:`select_base_study_version`.
    require_coordinates : :obj:`bool`, default=False
        Drop base studies whose selected version has no coordinates, instead of
        contributing a study with no foci. Combines with ``version_filter``.
    studyset_id, name : :obj:`str`, optional
        Identifier and label for the studyset being built.
    inherit_base_study_metadata : :obj:`bool`, default=True
        Fill a version's missing bibliographic fields from its base study. Base
        studies are enriched (DOI, PubMed) independently of their versions, so a
        version often has the coordinates while the base study has the citation.
    add_provenance : :obj:`bool`, default=True
        Record which base study and version each study came from, under the
        :data:`PROVENANCE_KEYS` metadata keys. What made a studyset is then part
        of the studyset, rather than of the script that built it.
    api : :obj:`neurostore_sdk.StoreApi`, optional
        Client to download with.
    host : :obj:`str`, optional
        A NeuroStore API to query instead of the SDK default.
    timeout : :obj:`float` or :obj:`tuple`, optional
        Request timeout, in seconds.
    n_jobs : :obj:`int`, default=1
        Number of concurrent downloads of the selected versions. Only used for
        versions whose analyses were not already embedded by a nested search.
    verbose : :obj:`bool`, default=True
        Show a progress bar while downloading.

    Returns
    -------
    studyset : :obj:`dict`
        A NIMADS studyset document: ``{"id": ..., "name": ..., "studies": [...]}``.

    See Also
    --------
    studyset_from_base_studies : The same thing as a :class:`~nimare.nimads.Studyset`.
    """
    keeps = [version_filter] if version_filter is not None else []
    if require_coordinates:
        keeps.append(lambda version: bool(version.get("has_coordinates")))
    combined = None if not keeps else (lambda version: all(keep(version) for keep in keeps))

    selected = []
    for base_study in base_studies:
        version = select_base_study_version(
            base_study, heuristic=heuristic, version_filter=combined
        )
        if version is None:
            LGR.debug(
                f"No version of base study '{base_study.get('id')}' satisfies the selection; "
                "skipping it."
            )
            continue
        selected.append((base_study, version))

    dropped = len(base_studies) - len(selected)
    if dropped:
        LGR.info(f"Skipped {dropped} of {len(base_studies)} base studies with no usable version.")

    to_fetch = [
        index for index, (_, version) in enumerate(selected) if not _has_embedded_analyses(version)
    ]
    if to_fetch:
        store_api, client = _store_api(api=api, host=host)
        ids = [selected[index][1].get("id") for index in to_fetch]
        progress = tqdm(
            ids,
            desc="Downloading NeuroStore study versions",
            disable=not verbose,
            total=len(ids),
        )
        if n_jobs == 1:
            fetched = [_fetch_version(store_api, client, vid, timeout) for vid in progress]
        else:
            from joblib import Parallel, delayed

            fetched = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_fetch_version)(store_api, client, vid, timeout) for vid in progress
            )
        for index, payload in zip(to_fetch, fetched):
            base_study, version = selected[index]
            # The summary and the full payload describe the same version, so
            # prefer whichever of the two actually carries a value.
            merged = dict(version)
            for key, value in payload.items():
                if value is not None or key not in merged:
                    merged[key] = value
            merged["analyses"] = payload.get("analyses") or []
            selected[index] = (base_study, merged)

    studies = [
        _study_payload(
            version,
            base_study,
            inherit=inherit_base_study_metadata,
            provenance=add_provenance,
        )
        for base_study, version in selected
    ]
    return {"id": studyset_id, "name": name, "studies": studies}


def studyset_from_base_studies(base_studies, *, target="mni152_2mm", mask=None, **kwargs):
    """Select one version per base study and build a local studyset.

    .. versionadded:: 0.22.0

    Parameters
    ----------
    base_studies : :obj:`list` of :obj:`dict`
        Base studies from :func:`search_base_studies`, with versions embedded.
    target : :obj:`str` or None, default="mni152_2mm"
        Space to report coordinates in.
    mask : Niimg-like or :class:`~nilearn.maskers.NiftiMasker`, optional
        Masker for execution.
    **kwargs
        Passed to :func:`base_studies_to_nimads_dict`, which documents the
        selection: ``heuristic``, ``version_filter``, ``require_coordinates``,
        ``studyset_id``, ``name``, ``inherit_base_study_metadata``,
        ``add_provenance``, ``api``, ``host``, ``timeout``, ``n_jobs``,
        ``verbose``.

    Returns
    -------
    studyset : :obj:`nimare.nimads.Studyset`
        One study per base study that had a usable version.
    """
    from nimare.nimads import Studyset

    payload = base_studies_to_nimads_dict(base_studies, **kwargs)
    return Studyset(payload, target=target, mask=mask)


def search_neurostore_studyset(
    query=None,
    *,
    heuristic=None,
    page_size=100,
    max_results=None,
    nested=False,
    target="mni152_2mm",
    mask=None,
    version_filter=None,
    require_coordinates=False,
    studyset_id="neurostore-search",
    name=None,
    inherit_base_study_metadata=True,
    add_provenance=True,
    api=None,
    host=None,
    timeout=None,
    n_jobs=1,
    verbose=True,
    **filters,
):
    """Search NeuroStore and build a local studyset from the results.

    .. versionadded:: 0.22.0

    Searches the base studies (one record per paper), picks one version of each
    according to ``heuristic``, downloads those versions and returns them as a
    :class:`~nimare.nimads.Studyset`.

    Parameters
    ----------
    query : :obj:`str`, optional
        Substring search across title, abstract, authors and journal.
    heuristic : :obj:`str`, :obj:`callable`, sequence, or None, optional
        Which version of each paper to take: e.g. ``"last_updated"``,
        ``"user"``, ``"llm"``, ``("neurosynth", "neuroquery")``, or
        ``"user,llm,last_updated"``. See :data:`VERSION_HEURISTICS` for the
        names and :func:`resolve_version_heuristic` for how a chain is read.
        Default is :data:`DEFAULT_VERSION_HEURISTIC`.
    page_size, max_results, nested, api, host, timeout
        Passed to :func:`search_base_studies`.
    target, mask
        Passed to :class:`~nimare.nimads.Studyset`.
    version_filter, require_coordinates, studyset_id, name, \
inherit_base_study_metadata, add_provenance, n_jobs, verbose
        Passed to :func:`base_studies_to_nimads_dict`.
    **filters
        Any other base-studies query parameter, e.g. ``year_min``,
        ``data_type``, ``semantic_search``.

    Returns
    -------
    studyset : :obj:`nimare.nimads.Studyset`
        One study per matching paper.

    Examples
    --------
    >>> from nimare.extract import search_neurostore_studyset
    >>> studyset = search_neurostore_studyset(
    ...     "working memory",
    ...     heuristic=("user", "llm", "last_updated"),
    ...     year_min=2010,
    ...     require_coordinates=True,
    ...     max_results=25,
    ... )  # doctest: +SKIP

    See Also
    --------
    search_base_studies : The search on its own.
    studyset_from_base_studies : The version selection on its own.
    nimare.extract.fetch_neurostore : Download a whole NeuroStore release.
    nimare.io.fetch_neurostore_studyset : Download a studyset someone curated.
    """
    base_studies = search_base_studies(
        query,
        page_size=page_size,
        max_results=max_results,
        nested=nested,
        api=api,
        host=host,
        timeout=timeout,
        **filters,
    )
    if name is None:
        name = f"NeuroStore search: {query}" if query else "NeuroStore base-study search"
    return studyset_from_base_studies(
        base_studies,
        target=target,
        mask=mask,
        heuristic=heuristic,
        version_filter=version_filter,
        require_coordinates=require_coordinates,
        studyset_id=studyset_id,
        name=name,
        inherit_base_study_metadata=inherit_base_study_metadata,
        add_provenance=add_provenance,
        api=api,
        host=host,
        timeout=timeout,
        n_jobs=n_jobs,
        verbose=verbose,
    )
