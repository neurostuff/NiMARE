"""Test nimare.extract."""

import hashlib
import os
import sys
import tarfile
import types
from glob import glob
from io import BytesIO
from unittest.mock import patch

import pytest

import nimare
import nimare.extract
from nimare.dataset import Dataset
from nimare.extract.extract import (
    _fetch_database,
    _get_available_entities,
    _select_neurostore_release,
)
from nimare.generate import create_coordinate_studyset
from nimare.nimads import Studyset
from nimare.tests.utils import get_test_data_path
from nimare.utils import get_resource_path


def mock_urlopen(url):
    """Mock URL opener that returns appropriate mock data based on file type."""
    if "coordinates.tsv.gz" in url:
        mock_data = b"x\ty\tz\n1\t2\t3\n4\t5\t6"
    elif "metadata.tsv.gz" in url:
        mock_data = b"id\ttitle\n1\tStudy 1\n2\tStudy 2"
    elif "features.npz" in url:
        mock_data = b"mock npz content"
    elif "vocabulary.txt" in url:
        mock_data = b"term1\nterm2\nterm3"
    else:
        mock_data = b"Mock file content"
    return BytesIO(mock_data)


def _local_neurosynth_manifest():
    base = get_test_data_path()
    return [
        {
            "coordinates": os.path.join(base, "data-neurosynth_version-7_coordinates.tsv.gz"),
            "metadata": os.path.join(base, "data-neurosynth_version-7_metadata.tsv.gz"),
            "features": [
                {
                    "features": os.path.join(
                        base,
                        "data-neurosynth_version-7_vocab-terms_source-abstract_type-tfidf_"
                        "features.npz",
                    ),
                    "vocabulary": os.path.join(
                        base,
                        "data-neurosynth_version-7_vocab-terms_vocabulary.txt",
                    ),
                }
            ],
        }
    ]


@patch("nimare.extract.extract.urlopen", side_effect=mock_urlopen)
def test_fetch_neurosynth(mock_url, tmp_path_factory):
    """Smoke test for extract.fetch_neurosynth.

    Taken from the Neurosynth Python package.
    """
    tmpdir = tmp_path_factory.mktemp("test_fetch_neurosynth")
    data_files = nimare.extract.fetch_neurosynth(
        data_dir=tmpdir,
        version="7",
        overwrite=False,
        source="abstract",
        vocab="terms",
        return_type="files",
    )
    # Check data_files structure
    assert isinstance(data_files, list)
    assert len(data_files) == 1

    # Verify expected files in data_files
    files_dict = data_files[0]
    assert "coordinates" in files_dict
    assert "metadata" in files_dict
    assert "features" in files_dict
    assert len(files_dict["features"]) == 1


@patch("nimare.extract.extract.urlopen", side_effect=mock_urlopen)
def test_fetch_neuroquery(mock_url, tmp_path_factory):
    """Smoke test for extract.fetch_neuroquery."""
    tmpdir = tmp_path_factory.mktemp("test_fetch_neuroquery")
    data_files = nimare.extract.fetch_neuroquery(
        data_dir=tmpdir,
        version="1",
        overwrite=False,
        source="abstract",
        vocab="neuroquery7547",
        type="count",
        return_type="files",
    )
    files = glob(os.path.join(tmpdir, "neuroquery", "*"))
    assert len(files) == 4

    # One set of files found
    assert isinstance(data_files, list)
    assert len(data_files) == 1

    # Verify mock was called with expected URLs
    assert mock_url.call_count > 0  # Should be called for each file download
    for call in mock_url.call_args_list:
        url = call[0][0]
        assert "neuroquery/neuroquery_data/blob" in url
        assert "?raw=true" in url


def test_fetch_neurosynth_returns_studyset_by_default(monkeypatch, tmp_path_factory):
    """fetch_neurosynth should return Studysets by default."""
    tmpdir = tmp_path_factory.mktemp("test_fetch_neurosynth_returns_studyset_by_default")
    monkeypatch.setattr(
        nimare.extract.extract,
        "_fetch_database",
        lambda *args, **kwargs: _local_neurosynth_manifest(),
    )

    outputs = nimare.extract.fetch_neurosynth(
        data_dir=tmpdir,
        version="7",
        source="abstract",
        vocab="terms",
    )

    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert isinstance(outputs[0], Studyset)


def test_fetch_neurosynth_can_return_dataset(monkeypatch, tmp_path_factory):
    """fetch_neurosynth should support the legacy Dataset return type."""
    tmpdir = tmp_path_factory.mktemp("test_fetch_neurosynth_can_return_dataset")
    monkeypatch.setattr(
        nimare.extract.extract,
        "_fetch_database",
        lambda *args, **kwargs: _local_neurosynth_manifest(),
    )

    outputs = nimare.extract.fetch_neurosynth(
        data_dir=tmpdir,
        version="7",
        source="abstract",
        vocab="terms",
        return_type="dataset",
    )

    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert isinstance(outputs[0], Dataset)


def test_fetch_neuroquery_returns_studyset_by_default(monkeypatch, tmp_path_factory):
    """fetch_neuroquery should return Studysets by default."""
    tmpdir = tmp_path_factory.mktemp("test_fetch_neuroquery_returns_studyset_by_default")
    monkeypatch.setattr(
        nimare.extract.extract,
        "_fetch_database",
        lambda *args, **kwargs: _local_neurosynth_manifest(),
    )

    outputs = nimare.extract.fetch_neuroquery(
        data_dir=tmpdir,
        version="1",
        source="combined",
        vocab="neuroquery6308",
        type="tfidf",
    )

    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert isinstance(outputs[0], Studyset)


def test_get_available_entities_lists_distinct_values():
    """_get_available_entities returns sorted, source-scoped distinct entity values."""
    manifest = [
        {
            "coordinates": "data-neurosynth_version-7_coordinates.tsv.gz",
            "metadata": "data-neurosynth_version-7_metadata.tsv.gz",
            "features": [
                {
                    "features": (
                        "data-neurosynth_version-7_vocab-terms_"
                        "source-abstract_type-tfidf_features.npz"
                    )
                },
                {
                    "features": (
                        "data-neurosynth_version-7_vocab-LDA200_"
                        "source-abstract_type-weight_features.npz"
                    )
                },
            ],
        },
        {
            "coordinates": "data-neuroquery_version-1_coordinates.tsv.gz",
            "metadata": "data-neuroquery_version-1_metadata.tsv.gz",
            "features": [
                {
                    "features": (
                        "data-neuroquery_version-1_vocab-neuroquery6308_"
                        "source-combined_type-tfidf_features.npz"
                    )
                },
            ],
        },
    ]
    ns = _get_available_entities(manifest, data="neurosynth")
    assert ns["version"] == ["7"]
    assert ns["vocab"] == ["LDA200", "terms"]
    assert ns["source"] == ["abstract"]
    assert ns["type"] == ["tfidf", "weight"]
    assert "neuroquery6308" not in ns["vocab"]  # other source must not leak in

    allsrc = _get_available_entities(manifest)
    assert allsrc["version"] == ["1", "7"]
    assert "neuroquery6308" in allsrc["vocab"]


def test_fetch_neurosynth_raises_on_unmatched_query(tmp_path):
    """An unmatched query raises an informative ValueError before any download."""
    with pytest.raises(ValueError) as excinfo:
        nimare.extract.fetch_neurosynth(data_dir=str(tmp_path), vocab="not_a_real_vocab")
    msg = str(excinfo.value)
    assert "No files matched the query" in msg
    assert "terms" in msg  # an available neurosynth vocab is surfaced


def test_fetch_neuroquery_raises_on_unmatched_query(tmp_path):
    """fetch_neuroquery raises the same informative error on an unmatched query."""
    with pytest.raises(ValueError) as excinfo:
        nimare.extract.fetch_neuroquery(data_dir=str(tmp_path), vocab="not_a_real_vocab")
    assert "No files matched the query" in str(excinfo.value)


def test_get_available_entities_ignores_unknown_segments():
    """Unknown key-value segments are ignored, and known entities still parse correctly."""
    manifest = [
        {
            "features": [
                {
                    "features": (
                        "data-neurosynth_version-7_vocab-terms_source-abstract_"
                        "type-tfidf_garbage-xyz_features.npz"
                    )
                },
            ],
        },
    ]
    out = _get_available_entities(manifest, data="neurosynth")
    assert out["vocab"] == ["terms"]
    assert out["version"] == ["7"]
    assert "garbage" not in out  # unrecognized key is never surfaced


def test_fetch_database_message_omits_data_none(tmp_path):
    """When the query has no 'data' key, the message uses a generic scope (no 'data-None')."""
    with pytest.raises(ValueError) as excinfo:
        _fetch_database({"version": "999"}, "http://example.com/", str(tmp_path))
    msg = str(excinfo.value)
    assert "No files matched the query" in msg
    assert "data-None" not in msg
    assert "the requested database" in msg


def test_fetch_database_message_falls_back_when_no_entities(tmp_path):
    """An unknown data source (no available entities) yields a generic fallback message."""
    with pytest.raises(ValueError) as excinfo:
        _fetch_database({"data": "not_a_database"}, "http://example.com/", str(tmp_path))
    msg = str(excinfo.value)
    assert "No matching entries were found" in msg
    assert "see the database file manifest" in msg


def test_download_abstracts_accepts_studyset(monkeypatch):
    """download_abstracts should update Studyset texts in place."""

    class _DummyEntrez:
        email = None

        @staticmethod
        def efetch(db, id, rettype, retmode):
            return id

    class _DummyMedline:
        @staticmethod
        def parse(handle):
            for pmid in handle:
                yield {"PMID": pmid, "AB": f"Abstract for {pmid}"}

    bio_module = types.ModuleType("Bio")
    bio_module.Entrez = _DummyEntrez
    bio_module.Medline = _DummyMedline
    monkeypatch.setitem(sys.modules, "Bio", bio_module)

    _, studyset = create_coordinate_studyset(foci=2, sample_size=20, n_studies=3, seed=2)
    studyset = nimare.extract.download_abstracts(studyset, "example@example.edu")

    assert "abstract" in studyset.texts.columns
    assert studyset.texts["abstract"].notnull().all()


def test_fetch_neurosynth_warns_and_points_to_neurostore(monkeypatch, tmp_path_factory):
    """fetch_neurosynth is deprecated and should say where to get current data."""
    tmpdir = tmp_path_factory.mktemp("test_fetch_neurosynth_warns")
    monkeypatch.setattr(
        nimare.extract.extract,
        "_fetch_database",
        lambda *args, **kwargs: _local_neurosynth_manifest(),
    )

    with pytest.warns(FutureWarning, match="fetch_neurosynth downloads a frozen") as record:
        nimare.extract.fetch_neurosynth(
            data_dir=tmpdir,
            version="7",
            source="abstract",
            vocab="terms",
            return_type="files",
        )

    assert any("fetch_neurostore" in str(warning.message) for warning in record)


def _release(version, release_type, built_at, content=b""):
    """Build a release entry shaped like the NeuroStore release index."""
    return {
        "version": version,
        "release_type": release_type,
        "built_at": built_at,
        "study_count": 10,
        "archive_name": f"neurostore-studyset-{version}.tar.gz",
        "archive_checksum": hashlib.sha256(content).hexdigest(),
        "download_path": f"/api/neurostore-studyset-releases/{version}/download",
    }


NEUROSTORE_RELEASES = [
    _release("nightly", "nightly", "2026-09-22T08:17:11+00:00"),
    _release("2026-09", "monthly", "2026-09-01T23:23:16+00:00"),
    _release("2026-08", "monthly", "2026-08-01T22:43:46+00:00"),
]


def test_select_neurostore_release_latest_skips_nightly():
    """The "latest" version is the newest dated release, not the rolling nightly build."""
    release = _select_neurostore_release(NEUROSTORE_RELEASES, "latest")
    assert release["version"] == "2026-09"


def test_select_neurostore_release_by_version():
    """An explicit version, nightly included, selects that release."""
    assert _select_neurostore_release(NEUROSTORE_RELEASES, "2026-08")["version"] == "2026-08"
    assert _select_neurostore_release(NEUROSTORE_RELEASES, "nightly")["version"] == "nightly"


def test_select_neurostore_release_unknown_version_lists_options():
    """An unknown version raises an error naming the available releases."""
    with pytest.raises(ValueError) as excinfo:
        _select_neurostore_release(NEUROSTORE_RELEASES, "1999-01")
    msg = str(excinfo.value)
    assert "2026-09" in msg
    assert "latest" in msg


def _make_release_archive(version="2026-09"):
    """Tar the packaged parquet studyset the way NeuroStore ships a release."""
    parquet_dir = os.path.join(get_resource_path(), "neurostore_parquet_studyset")
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tf:
        tf.add(parquet_dir, arcname=f"neurostore-studyset-{version}")
    return buffer.getvalue()


class _FakeResponse:
    """Minimal stand-in for a ``requests`` response, streaming or not."""

    def __init__(self, json_data=None, content=b""):
        self._json = json_data
        self._content = content

    def json(self):
        return self._json

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=1):
        for start in range(0, len(self._content), chunk_size):
            yield self._content[start : start + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeRequests:
    """Serve a release index and archive bytes without touching the network."""

    def __init__(self, releases, archives):
        self.releases = releases
        self.archives = archives
        self.download_count = 0

    def get(self, url, **kwargs):
        if url.endswith("/download"):
            self.download_count += 1
            version = url.rstrip("/").split("/")[-2]
            return _FakeResponse(content=self.archives[version])
        return _FakeResponse(json_data={"results": self.releases})


@pytest.fixture
def fake_neurostore(monkeypatch):
    """Patch the release index and archive download with local fixtures."""
    archive = _make_release_archive("2026-09")
    releases = [
        _release("nightly", "nightly", "2026-09-22T08:17:11+00:00", archive),
        _release("2026-09", "monthly", "2026-09-01T23:23:16+00:00", archive),
        _release("2026-08", "monthly", "2026-08-01T22:43:46+00:00", archive),
    ]
    fake = _FakeRequests(releases, {rel["version"]: archive for rel in releases})
    monkeypatch.setattr(nimare.extract.extract, "requests", fake)
    return fake


def test_fetch_neurostore_releases_lists_available_versions(fake_neurostore):
    """fetch_neurostore_releases surfaces the release index."""
    releases = nimare.extract.fetch_neurostore_releases()
    assert [rel["version"] for rel in releases] == ["nightly", "2026-09", "2026-08"]


def test_fetch_neurostore_returns_studyset(fake_neurostore, tmp_path):
    """The default return type is a Studyset built from the extracted parquet tables."""
    pytest.importorskip("pyarrow")

    studyset = nimare.extract.fetch_neurostore(data_dir=tmp_path)

    assert isinstance(studyset, Studyset)
    assert len(studyset.ids)
    assert os.path.isfile(
        os.path.join(
            tmp_path, "neurostore", "2026-09", "neurostore-studyset-2026-09", "studyset.json"
        )
    )
    # The archive is transport only; the extracted tables are what remain.
    assert not glob(os.path.join(tmp_path, "neurostore", "*.tar.gz"))


def test_fetch_neurostore_returns_files_and_caches(fake_neurostore, tmp_path):
    """The "files" return type skips loading, and a cached release is not re-downloaded."""
    release_dir = nimare.extract.fetch_neurostore(data_dir=tmp_path, return_type="files")
    assert os.path.isfile(os.path.join(release_dir, "studyset.json"))
    assert fake_neurostore.download_count == 1

    again = nimare.extract.fetch_neurostore(data_dir=tmp_path, return_type="files")
    assert again == release_dir
    assert fake_neurostore.download_count == 1  # served from the cache

    nimare.extract.fetch_neurostore(data_dir=tmp_path, return_type="files", overwrite=True)
    assert fake_neurostore.download_count == 2


def test_fetch_neurostore_specific_version(fake_neurostore, tmp_path):
    """An explicit version downloads that release into its own directory."""
    release_dir = nimare.extract.fetch_neurostore(
        version="nightly",
        data_dir=tmp_path,
        return_type="files",
    )
    assert os.path.join("neurostore", "nightly") in release_dir


def test_fetch_neurostore_rejects_bad_checksum(fake_neurostore, tmp_path):
    """A corrupted download is rejected rather than extracted."""
    for release in fake_neurostore.releases:
        release["archive_checksum"] = "0" * 64

    with pytest.raises(ValueError, match="Checksum mismatch"):
        nimare.extract.fetch_neurostore(data_dir=tmp_path, return_type="files")

    assert not glob(os.path.join(tmp_path, "neurostore", "*.tar.gz"))


def test_fetch_neurostore_skips_checksum_when_disabled(fake_neurostore, tmp_path):
    """verify_checksum=False accepts an archive the index disagrees with."""
    for release in fake_neurostore.releases:
        release["archive_checksum"] = "0" * 64

    release_dir = nimare.extract.fetch_neurostore(
        data_dir=tmp_path,
        return_type="files",
        verify_checksum=False,
    )
    assert os.path.isfile(os.path.join(release_dir, "studyset.json"))


def test_fetch_neurostore_rejects_invalid_return_type(tmp_path):
    """An unsupported return_type fails before any request is made."""
    with pytest.raises(ValueError, match="Invalid return_type"):
        nimare.extract.fetch_neurostore(data_dir=tmp_path, return_type="dataset")
