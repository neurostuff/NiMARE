"""Test nimare.extract.neurostore (NeuroStore base-study search)."""

import os

import pytest

from nimare.extract import neurostore as ns
from nimare.nimads import Studyset


def _version(vid, **kwargs):
    """Return a version summary of the shape an ``info`` search returns."""
    version = {
        "id": vid,
        "source": None,
        "user": None,
        "username": None,
        "created_at": "2020-01-01T00:00:00+00:00",
        "updated_at": "2020-01-01T00:00:00+00:00",
        "has_coordinates": True,
        "has_images": False,
    }
    version.update(kwargs)
    return version


def _base_study(versions, **kwargs):
    """Return a base study carrying the given versions."""
    base_study = {"id": "base-1", "name": "A paper", "year": 2015, "versions": list(versions)}
    base_study.update(kwargs)
    return base_study


class FakeStoreApi:
    """A StoreApi double: serves canned pages and canned nested studies."""

    def __init__(self, pages=(), studies=None, total=None):
        self.pages = list(pages)
        self.studies = studies or {}
        self.total = total
        self.calls = []
        self.fetched = []

    def base_studies_get(self, **kwargs):
        """Return the next canned page, recording how it was asked for."""
        self.calls.append(kwargs)
        page = kwargs.get("page", 1)
        results = self.pages[page - 1] if page - 1 < len(self.pages) else []
        total = self.total if self.total is not None else sum(len(p) for p in self.pages)
        return {"metadata": {"total_count": total}, "results": results}

    def studies_id_get(self, id, **kwargs):
        """Return the canned nested study for a version ID."""
        self.fetched.append(id)
        return self.studies[id]


# ---------------------------------------------------------------------------
# heuristics
# ---------------------------------------------------------------------------
def test_prefer_source_orders_by_preference():
    """prefer_source ranks the named sources first, in the order given."""
    heuristic = ns.prefer_source("llm", "neurosynth")

    assert heuristic({"source": "llm"}) < heuristic({"source": "neurosynth"})
    assert heuristic({"source": "neurosynth"}) < heuristic({"source": "neuroquery"})
    assert heuristic({"source": "LLM"}) == heuristic({"source": "llm"})
    assert heuristic({"source": None}) == heuristic({"source": "anything-else"})


@pytest.mark.parametrize(
    "heuristic,expected",
    [
        ("last_updated", "recent"),
        ("newest", "recent"),
        ("first_updated", "stale"),
        ("last_created", "recent"),
        ("first_created", "stale"),
        ("oldest", "stale"),
        ("user", "curated"),
        ("llm", "llm"),
        ("neurosynth", "stale"),
        ("source:llm", "llm"),
        ("source:llm|neuroquery", "llm"),
        ("source:neuroquery|llm", "recent"),
        (["user", "llm"], "curated"),
        ("user,llm", "curated"),
        (["llm", "user"], "llm"),
    ],
)
def test_select_base_study_version_heuristics(heuristic, expected):
    """Each heuristic selects the version it names, and chains break ties in order."""
    versions = [
        _version(
            "stale",
            source="neurosynth",
            created_at="2018-01-01T00:00:00+00:00",
            updated_at="2018-01-01T00:00:00+00:00",
        ),
        _version(
            "llm",
            source="llm",
            created_at="2021-01-01T00:00:00+00:00",
            updated_at="2021-01-01T00:00:00+00:00",
        ),
        _version(
            "recent",
            source="neuroquery",
            created_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:00:00+00:00",
        ),
        _version(
            "curated",
            source=None,
            user="auth0|1",
            created_at="2019-01-01T00:00:00+00:00",
            updated_at="2019-01-01T00:00:00+00:00",
        ),
    ]
    selected = ns.select_base_study_version(_base_study(versions), heuristic=heuristic)

    assert selected["id"] == expected


def test_default_heuristic_prefers_coordinates_then_user():
    """The default drops a version with no coordinates before honoring ownership."""
    versions = [
        _version("owned-but-empty", user="auth0|1", has_coordinates=False),
        _version("has-foci", source="llm"),
    ]

    assert ns.select_base_study_version(_base_study(versions))["id"] == "has-foci"


def test_heuristic_accepts_a_callable_of_one_argument():
    """A user heuristic may take only the version, or the version and base study."""
    versions = [_version("a", year=2000), _version("b", year=2020)]
    by_year_desc = ns.select_base_study_version(
        _base_study(versions), heuristic=lambda version: -(version.get("year") or 0)
    )
    with_base = ns.select_base_study_version(
        _base_study(versions),
        heuristic=lambda version, base_study: abs((version.get("year") or 0) - base_study["year"]),
    )

    assert by_year_desc["id"] == "b"
    assert with_base["id"] == "b"


def test_heuristic_chain_is_deterministic_without_timestamps():
    """Two versions that tie on everything still select the same way every time."""
    versions = [_version("b", created_at=None, updated_at=None), _version("a", updated_at=None)]

    assert ns.select_base_study_version(_base_study(versions), heuristic="llm")["id"] == "a"


def test_missing_timestamps_sort_last():
    """A version with no timestamp is a last resort, not a most-recent one."""
    versions = [
        _version("dated", updated_at="2015-01-01T00:00:00Z"),
        _version("undated", updated_at=None),
    ]

    assert ns.select_base_study_version(_base_study(versions), heuristic="newest")["id"] == "dated"


def test_naive_and_zulu_timestamps_parse():
    """Compare the several timestamp spellings NeuroStore emits."""
    assert ns._timestamp("2020-01-01T00:00:00Z") == ns._timestamp("2020-01-01T00:00:00+00:00")
    assert ns._timestamp("2020-01-01T00:00:00") == ns._timestamp("2020-01-01T00:00:00+00:00")
    assert ns._timestamp("not a date") is None
    assert ns._timestamp(None) is None


def test_version_filter_can_leave_nothing_to_select():
    """A base study whose every version is filtered out has no selection."""
    base_study = _base_study([_version("a", has_images=False)])

    assert (
        ns.select_base_study_version(base_study, version_filter=lambda v: v["has_images"]) is None
    )
    assert ns.select_base_study_version(_base_study([])) is None


def test_versions_as_ids_are_an_error():
    """Versions returned as bare IDs carry nothing to choose on."""
    with pytest.raises(ValueError, match="lists its versions as IDs"):
        ns.select_base_study_version(_base_study(["v1", "v2"]))


def test_unknown_heuristic_names_are_reported_with_the_known_ones():
    """A typo in a heuristic name is an error that lists the alternatives."""
    with pytest.raises(ValueError, match="Unknown version heuristic 'neurosynthh'"):
        ns.resolve_version_heuristic("neurosynthh")
    with pytest.raises(ValueError, match="names no source"):
        ns.resolve_version_heuristic("source:")
    with pytest.raises(TypeError, match="must be a string or a callable"):
        ns.resolve_version_heuristic([3])


def test_register_version_heuristic():
    """A registered heuristic is usable by name, and cannot shadow a built-in."""
    ns.register_version_heuristic(
        "test_prefer_b", lambda version: 0 if version["id"] == "b" else 1
    )
    try:
        versions = [_version("a"), _version("b")]
        assert (
            ns.select_base_study_version(_base_study(versions), heuristic="test_prefer_b")["id"]
            == "b"
        )
        with pytest.raises(ValueError, match="already registered"):
            ns.register_version_heuristic("test_prefer_b", lambda version: 0)
        with pytest.raises(TypeError, match="must be callable"):
            ns.register_version_heuristic("test_not_callable", "nope")
    finally:
        ns.VERSION_HEURISTICS.pop("test_prefer_b", None)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------
def test_search_base_studies_paginates_and_passes_the_query():
    """The search walks pages until the results run out."""
    api = FakeStoreApi(
        pages=[[_base_study([_version("a")], id="s1")] * 2, [_base_study([], id="s2")]]
    )

    results = ns.search_base_studies("working memory", page_size=2, api=api)

    assert len(results) == 3
    assert [call["page"] for call in api.calls] == [1, 2]
    assert api.calls[0]["search"] == "working memory"
    assert api.calls[0]["info"] is True


def test_search_base_studies_stops_at_max_results():
    """max_results caps the results and the number of requests."""
    api = FakeStoreApi(pages=[[{"id": f"s{i}", "versions": []} for i in range(5)]] * 3, total=15)

    results = ns.search_base_studies(max_results=3, page_size=5, api=api)

    assert [study["id"] for study in results] == ["s0", "s1", "s2"]
    assert len(api.calls) == 1
    assert api.calls[0]["page_size"] == 3


def test_search_base_studies_stops_at_the_reported_total():
    """A server that keeps returning full pages does not loop forever."""
    api = FakeStoreApi(pages=[[{"id": "s1", "versions": []}]] * 4, total=2)

    assert len(ns.search_base_studies(page_size=1, api=api)) == 2


def test_search_base_studies_nested_requests_nested_payloads():
    """A nested search asks for embedded analyses instead of version summaries."""
    api = FakeStoreApi(pages=[[{"id": "s1", "versions": []}]])

    ns.search_base_studies(nested=True, api=api)

    assert api.calls[0]["nested"] is True
    assert "info" not in api.calls[0]


def test_search_base_studies_forwards_filters():
    """Filters reach the endpoint under their own names."""
    api = FakeStoreApi(pages=[[{"id": "s1", "versions": []}]])

    ns.search_base_studies(year_min=2010, data_type="coordinates", sort="year", desc=True, api=api)

    assert api.calls[0]["year_min"] == 2010
    assert api.calls[0]["data_type"] == "coordinates"
    assert api.calls[0]["sort"] == "year"
    assert api.calls[0]["desc"] is True


def test_search_base_studies_rejects_bad_parameters():
    """A mistyped filter is an error rather than a silently ignored one."""
    api = FakeStoreApi(pages=[[]])

    with pytest.raises(TypeError, match="Unknown NeuroStore base-study search parameter"):
        ns.search_base_studies(yer_min=2010, api=api)
    with pytest.raises(TypeError, match="drives pagination itself"):
        ns.search_base_studies(page=3, api=api)
    with pytest.raises(TypeError, match="drives pagination itself"):
        ns.search_base_studies(flat=True, api=api)
    with pytest.raises(TypeError, match="drives pagination itself"):
        ns.search_base_studies(paginate=False, api=api)
    with pytest.raises(TypeError, match="Pass the substring search as 'query'"):
        ns.search_base_studies("memory", search="memory", api=api)
    with pytest.raises(ValueError, match="incompatible"):
        ns.search_base_studies(nested=True, info=True, api=api)
    with pytest.raises(ValueError, match="page_size must be between"):
        ns.search_base_studies(page_size=0, api=api)
    assert ns.search_base_studies(max_results=0, api=api) == []


def test_search_base_studies_wraps_api_errors():
    """A failed search reports the query that failed."""
    import neurostore_sdk

    class FailingApi:
        def base_studies_get(self, **kwargs):
            raise neurostore_sdk.ApiException(status=500, reason="Server Error")

    with pytest.raises(ValueError, match="Failed to search NeuroStore base studies"):
        ns.search_base_studies("memory", api=FailingApi())


# ---------------------------------------------------------------------------
# studyset assembly
# ---------------------------------------------------------------------------
def _nested_study(vid, n_points=2, **kwargs):
    """Return a nested study payload of the shape ``studies_id_get`` returns."""
    study = {
        "id": vid,
        "name": None,
        "doi": None,
        "pmid": None,
        "authors": None,
        "year": None,
        "publication": None,
        "description": None,
        "metadata": {"sample_sizes": [20]},
        "analyses": [
            {
                "id": f"{vid}-analysis",
                "name": "contrast",
                "conditions": [],
                "images": [],
                "points": [
                    {
                        "id": f"{vid}-point-{i}",
                        "coordinates": [float(i), 0.0, 0.0],
                        "space": "MNI",
                        "kind": "unknown",
                        "values": [],
                    }
                    for i in range(n_points)
                ],
            }
        ],
    }
    study.update(kwargs)
    return study


def test_base_studies_to_nimads_dict_downloads_only_selected_versions():
    """Only the selected version of each base study is downloaded."""
    base_studies = [
        _base_study(
            [
                _version("wanted", source="llm"),
                _version("unwanted", source="neurosynth"),
            ],
            id="base-1",
        )
    ]
    api = FakeStoreApi(studies={"wanted": _nested_study("wanted")})

    payload = ns.base_studies_to_nimads_dict(base_studies, heuristic="llm", api=api, verbose=False)

    assert api.fetched == ["wanted"]
    assert [study["id"] for study in payload["studies"]] == ["wanted"]
    assert len(payload["studies"][0]["analyses"][0]["points"]) == 2


def test_base_studies_to_nimads_dict_records_provenance_and_inherits_metadata():
    """Each study says where it came from, and borrows the citation it lacks."""
    base_studies = [
        _base_study(
            [_version("v1", source="llm", username="Someone")],
            id="base-1",
            name="Working memory in the insula",
            doi="10.1234/abc",
            pmid="12345",
            year=2015,
        )
    ]
    api = FakeStoreApi(studies={"v1": _nested_study("v1")})

    payload = ns.base_studies_to_nimads_dict(base_studies, heuristic="llm", api=api, verbose=False)
    study = payload["studies"][0]

    assert study["name"] == "Working memory in the insula"
    assert study["doi"] == "10.1234/abc"
    assert study["pmid"] == "12345"
    assert study["year"] == 2015
    assert study["metadata"]["neurostore_base_study_id"] == "base-1"
    assert study["metadata"]["neurostore_version_id"] == "v1"
    assert study["metadata"]["neurostore_source"] == "llm"
    assert study["metadata"]["neurostore_username"] == "Someone"
    # The version's own metadata survives alongside the provenance.
    assert study["metadata"]["sample_sizes"] == [20]


def test_base_studies_to_nimads_dict_can_skip_provenance_and_inheritance():
    """Both annotations of the payload are opt-out."""
    base_studies = [_base_study([_version("v1")], id="base-1", name="A paper")]
    api = FakeStoreApi(studies={"v1": _nested_study("v1")})

    payload = ns.base_studies_to_nimads_dict(
        base_studies,
        api=api,
        verbose=False,
        add_provenance=False,
        inherit_base_study_metadata=False,
    )
    study = payload["studies"][0]

    assert study["name"] is None
    assert not any(key in study["metadata"] for key in ns.PROVENANCE_KEYS)


def test_base_studies_to_nimads_dict_uses_embedded_analyses():
    """A nested search needs no follow-up downloads."""
    embedded = _nested_study("v1")
    embedded.update(_version("v1", source="llm"))
    embedded["analyses"] = _nested_study("v1")["analyses"]
    api = FakeStoreApi()

    payload = ns.base_studies_to_nimads_dict(
        [_base_study([embedded], id="base-1")], api=api, verbose=False
    )

    assert api.fetched == []
    assert len(payload["studies"][0]["analyses"][0]["points"]) == 2


def test_base_studies_to_nimads_dict_treats_no_analyses_as_embedded():
    """A nested version with no analyses is an answer, not a missing download."""
    version = _version("v1", source="llm", has_coordinates=False)
    version["analyses"] = []
    api = FakeStoreApi()

    payload = ns.base_studies_to_nimads_dict(
        [_base_study([version], id="base-1")], api=api, verbose=False
    )

    assert api.fetched == []
    assert payload["studies"][0]["analyses"] == []


def test_base_studies_to_nimads_dict_require_coordinates_skips_studies():
    """require_coordinates drops papers that would contribute no foci."""
    base_studies = [
        _base_study([_version("empty", has_coordinates=False)], id="base-1"),
        _base_study([_version("full", has_coordinates=True)], id="base-2"),
    ]
    api = FakeStoreApi(studies={"full": _nested_study("full")})

    payload = ns.base_studies_to_nimads_dict(
        base_studies, require_coordinates=True, api=api, verbose=False
    )

    assert [study["id"] for study in payload["studies"]] == ["full"]
    assert api.fetched == ["full"]


def test_base_studies_to_nimads_dict_combines_filters():
    """version_filter and require_coordinates both have to pass."""
    base_studies = [
        _base_study(
            [
                _version("no-foci", source="llm", has_coordinates=False),
                _version("wrong-source", source="neurosynth"),
            ],
            id="base-1",
        )
    ]
    api = FakeStoreApi()

    payload = ns.base_studies_to_nimads_dict(
        base_studies,
        require_coordinates=True,
        version_filter=lambda version: version.get("source") == "llm",
        api=api,
        verbose=False,
    )

    assert payload["studies"] == []


def test_base_studies_to_nimads_dict_names_the_studyset():
    """The studyset carries the id and name it was asked for."""
    api = FakeStoreApi(studies={"v1": _nested_study("v1")})

    payload = ns.base_studies_to_nimads_dict(
        [_base_study([_version("v1")], id="base-1")],
        studyset_id="my-id",
        name="my name",
        api=api,
        verbose=False,
    )

    assert payload["id"] == "my-id"
    assert payload["name"] == "my name"


def test_base_studies_to_nimads_dict_parallel_downloads_keep_order():
    """Concurrent downloads are reassembled against the studies that asked for them."""
    base_studies = [_base_study([_version(f"v{i}")], id=f"base-{i}") for i in range(4)]
    api = FakeStoreApi(studies={f"v{i}": _nested_study(f"v{i}", n_points=i + 1) for i in range(4)})

    payload = ns.base_studies_to_nimads_dict(base_studies, api=api, verbose=False, n_jobs=2)

    assert [study["id"] for study in payload["studies"]] == ["v0", "v1", "v2", "v3"]
    assert [len(study["analyses"][0]["points"]) for study in payload["studies"]] == [1, 2, 3, 4]


def test_fetch_version_wraps_api_errors():
    """A failed version download names the version."""
    import neurostore_sdk

    class FailingApi:
        def studies_id_get(self, id, **kwargs):
            raise neurostore_sdk.ApiException(status=404, reason="Not Found")

    with pytest.raises(ValueError, match="Failed to download NeuroStore study version 'v1'"):
        ns.base_studies_to_nimads_dict(
            [_base_study([_version("v1")], id="base-1")], api=FailingApi(), verbose=False
        )


def test_studyset_from_base_studies_builds_a_studyset():
    """The assembled payload loads as a Studyset in the requested space."""
    api = FakeStoreApi(studies={"v1": _nested_study("v1", n_points=3)})

    studyset = ns.studyset_from_base_studies(
        [_base_study([_version("v1")], id="base-1")],
        target="ale_2mm",
        api=api,
        verbose=False,
    )

    assert isinstance(studyset, Studyset)
    assert studyset.space == "ale_2mm"
    assert len(studyset.coordinates) == 3
    assert studyset.metadata["neurostore_base_study_id"].tolist() == ["base-1"]


def test_search_neurostore_studyset_searches_then_selects(monkeypatch):
    """The one-call entry point threads the query and the heuristic through."""
    seen = {}

    def fake_search(query=None, **kwargs):
        seen["query"] = query
        seen["search_kwargs"] = kwargs
        return [_base_study([_version("v1", source="llm"), _version("v2")], id="base-1")]

    monkeypatch.setattr(ns, "search_base_studies", fake_search)
    api = FakeStoreApi(studies={"v1": _nested_study("v1")})

    studyset = ns.search_neurostore_studyset(
        "working memory", heuristic="llm", year_min=2010, api=api, verbose=False
    )

    assert seen["query"] == "working memory"
    assert seen["search_kwargs"]["year_min"] == 2010
    assert studyset.name == "NeuroStore search: working memory"
    assert api.fetched == ["v1"]


# ---------------------------------------------------------------------------
# against the live API, replayed from cassettes
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def vcr_cassette_dir():
    """Store this module's cassettes in a stable directory."""
    return os.path.join(os.path.dirname(__file__), "cassettes", "test_extract_neurostore")


@pytest.fixture(scope="module")
def vcr_config():
    """Keep NeuroStore cassettes stable and free of credentials."""
    return {
        "filter_headers": ["authorization"],
        "decode_compressed_response": True,
    }


@pytest.mark.vcr
def test_search_base_studies_live():
    """A real search returns base studies with their versions embedded."""
    base_studies = ns.search_base_studies("working memory", max_results=5, page_size=5)

    assert len(base_studies) == 5
    for base_study in base_studies:
        versions = ns._version_dicts(base_study)
        assert versions
        assert all("source" in version for version in versions)


@pytest.mark.vcr
def test_search_neurostore_studyset_live():
    """A real search builds a studyset with one version per paper."""
    studyset = ns.search_neurostore_studyset(
        "working memory",
        heuristic=("user", "llm", "last_updated"),
        max_results=3,
        page_size=3,
        require_coordinates=True,
        target="ale_2mm",
        verbose=False,
    )

    assert isinstance(studyset, Studyset)
    assert studyset.space == "ale_2mm"
    assert len(studyset.coordinates) > 0
    # One study per base study, and each says which version it is.
    ids = studyset.metadata["neurostore_base_study_id"].dropna().unique()
    assert len(ids) == len(studyset.metadata["neurostore_version_id"].dropna().unique())
