"""Tests for null analyses: analyses that were run and reported no significant foci.

A null analysis is declared, not inferred: it has no points and its metadata says
``outcome: no_significant_effect``. MKDAChi2 counts it; every other coordinate-based
estimator leaves it out, as before, logs that it did, and records the ids it left out.
"""

import logging

import numpy as np
import pytest
from scipy import sparse as sp_sparse
from scipy.stats import chi2_contingency

from nimare.meta.cbma import ALE, MKDAChi2, MKDADensity
from nimare.meta.kernel import MKDAKernel
from nimare.studyset import Studyset
from nimare.studyset.requirements import NULL_OUTCOME, OUTCOME_KEY, null_analyses
from nimare.utils import mm2vox

try:
    import torch  # noqa: F401
except ImportError:
    TORCH_INSTALLED = False
else:
    TORCH_INSTALLED = True

ORIGIN = [0.0, 0.0, 0.0]
ELSEWHERE = [40.0, 40.0, 20.0]


def _studyset(name, foci, n_null=0, n_undeclared=0, sample_size=20):
    """One analysis per study: one per entry of ``foci``, then nulls, then undeclared empties."""
    studies = []
    for i, xyz in enumerate(foci):
        points = [{"space": "MNI", "coordinates": list(c)} for c in np.atleast_2d(xyz)]
        studies.append(
            {
                "id": f"{name}{i}",
                "analyses": [
                    {"id": "a", "metadata": {"sample_sizes": [sample_size]}, "points": points}
                ],
            }
        )
    for i in range(n_null):
        studies.append(
            {
                "id": f"{name}null{i}",
                "analyses": [
                    {
                        "id": "a",
                        "metadata": {OUTCOME_KEY: NULL_OUTCOME, "sample_sizes": [sample_size]},
                        "points": [],
                    }
                ],
            }
        )
    for i in range(n_undeclared):
        studies.append({"id": f"{name}empty{i}", "analyses": [{"id": "a", "points": []}]})
    return Studyset({"id": name, "studies": studies})


def _masked_index(masker, xyz):
    """Return the masked-array position of the voxel containing ``xyz``."""
    mask = masker.mask_img
    flat = np.zeros(mask.shape, dtype=bool)
    flat[tuple(mm2vox(np.atleast_2d(xyz), mask.affine)[0])] = True
    in_mask = np.asanyarray(mask.dataobj).astype(bool)
    assert flat[in_mask].any(), "test voxel is outside the mask"
    return int(np.flatnonzero(flat[in_mask])[0])


def test_null_analyses_need_the_declaration_and_no_points():
    """Zero points alone is not a null: an image-only or uncurated analysis has none either."""
    studyset = _studyset("g", [ORIGIN], n_null=2, n_undeclared=1)
    flags = dict(zip(studyset.ids, null_analyses(studyset.store)))
    assert flags == {"g0-a": False, "gnull0-a": True, "gnull1-a": True, "gempty0-a": False}

    # A declared null that nonetheless has points is an ordinary analysis.
    doc = {
        "id": "x",
        "studies": [
            {
                "id": "s",
                "analyses": [
                    {
                        "id": "a",
                        "metadata": {OUTCOME_KEY: NULL_OUTCOME},
                        "points": [{"space": "MNI", "coordinates": ORIGIN}],
                    }
                ],
            }
        ],
    }
    assert not null_analyses(Studyset(doc).store).any()


def test_MKDAChi2_counts_null_analyses_by_hand():
    """The counts at a voxel match a 2x2 table built by hand, with the nulls in it.

    Group 1 has four analyses with a focus at the origin and four nulls; group 2 has one at the
    origin and three elsewhere. At the origin, a = 4 of n1 = 8 and b = 1 of n2 = 4 are active.
    The undeclared empty analysis in each group is not counted.
    """
    group1 = _studyset("g1", [ORIGIN] * 4, n_null=4, n_undeclared=1)
    group2 = _studyset("g2", [ORIGIN] + [ELSEWHERE] * 3, n_undeclared=1)
    result = MKDAChi2().fit(group1, group2)
    meta = result.estimator

    assert len(meta.inputs_["id1"]) == 8
    assert len(meta.inputs_["id2"]) == 4

    a, n1, b, n2 = 4, 8, 1, 4
    idx = _masked_index(meta.masker, ORIGIN)
    assert result.maps["prob_desc-AgF"][idx] == pytest.approx(a / n1)
    assert result.maps["prob_desc-AgU"][idx] == pytest.approx(b / n2)
    assert result.maps["prob_desc-A"][idx] == pytest.approx((a + b) / (n1 + n2))
    expected_chi2 = chi2_contingency([[a, n1 - a], [b, n2 - b]], correction=False)[0]
    assert result.maps["chi2_desc-association"][idx] == pytest.approx(expected_chi2, rel=1e-5)
    assert "4 of which reported no significant foci" in result.description_
    assert result.dropped_null_analyses == []


def test_MKDAChi2_without_nulls_is_unchanged():
    """A studyset with no declared nulls gives the same maps as before, bit for bit."""
    group1 = _studyset("g1", [ORIGIN] * 4)
    group2 = _studyset("g2", [ORIGIN] + [ELSEWHERE] * 3)
    with_empties = MKDAChi2().fit(
        _studyset("g1", [ORIGIN] * 4, n_undeclared=2), _studyset("g2", [ORIGIN] + [ELSEWHERE] * 3)
    )
    plain = MKDAChi2().fit(group1, group2)
    for key in ("chi2_desc-association", "chi2_desc-uniformity", "prob_desc-AgF"):
        np.testing.assert_array_equal(plain.maps[key], with_empties.maps[key])


def test_MKDAChi2_label_permutation_pools_null_analyses(monkeypatch):
    """Null analyses are shuffled between the groups as empty maps."""
    group1 = _studyset("g1", [ORIGIN] * 3, n_null=3)
    group2 = _studyset("g2", [ELSEWHERE] * 2, n_null=1)
    meta = MKDAChi2(random_state=0)
    result = meta.fit(group1, group2)

    seen = {}
    original = MKDAChi2._run_label_perm_fwe_permutation

    def spy(self, i_iter, pooled_maps, n_selected, n_unselected, *args, **kwargs):
        seen.update(rows=pooled_maps.shape[0], n_selected=n_selected, n_unselected=n_unselected)
        seen["empty_rows"] = int((pooled_maps.getnnz(axis=1) == 0).sum())
        return original(self, i_iter, pooled_maps, n_selected, n_unselected, *args, **kwargs)

    monkeypatch.setattr(MKDAChi2, "_run_label_perm_fwe_permutation", spy)
    meta.correct_fwe_montecarlo(result, n_iters=2)
    assert seen == {"rows": 9, "n_selected": 6, "n_unselected": 3, "empty_rows": 4}


def test_MKDAChi2_accepts_a_group_of_only_null_analyses():
    """A group in which nothing was found activates no voxel, rather than failing the fit."""
    result = MKDAChi2().fit(_studyset("g1", [], n_null=3), _studyset("g2", [ORIGIN] * 2))
    assert len(result.estimator.inputs_["id1"]) == 3
    assert result.maps["prob_desc-AgF"].max() == 0
    assert result.maps["prob_desc-AgU"].max() == 1


def test_combine_analyses_keeps_a_null_only_when_every_analysis_was_null():
    """A study is null after merging only if each of its analyses was a declared null."""
    null = {"metadata": {OUTCOME_KEY: NULL_OUTCOME}, "points": []}
    doc = {
        "id": "x",
        "studies": [
            {"id": "all", "analyses": [dict(null, id="a"), dict(null, id="b")]},
            {"id": "mixed", "analyses": [{"id": "a", "points": []}, dict(null, id="b")]},
        ],
    }
    combined = Studyset(doc).combine_analyses()
    flags = dict(zip(combined.ids, null_analyses(combined.store)))
    assert flags == {"all-a_b": True, "mixed-a_b": False}


@pytest.mark.parametrize("estimator", [ALE, MKDADensity])
def test_other_estimators_drop_nulls_and_say_so(estimator, caplog):
    """ALE and MKDADensity are unaffected by empty maps, so nulls are dropped, with a warning."""
    studyset = _studyset("g", [ORIGIN, ELSEWHERE, ORIGIN], n_null=2, n_undeclared=1)
    with caplog.at_level(logging.INFO, logger="nimare.studyset.requirements"):
        meta = estimator(null_method="approximate")
        result = meta.fit(studyset)

    assert len(meta.inputs_["id"]) == 3
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "2 of 6 analyses are null analyses" in warnings[0].getMessage()
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert any("1 of 6 analyses have no coordinates" in r.getMessage() for r in infos)
    assert result.dropped_null_analyses == ["gnull0-a", "gnull1-a"]
    # Corrected results are copies, and keep the record.
    assert result.copy().dropped_null_analyses == ["gnull0-a", "gnull1-a"]


def test_kernel_transform_reports_null_analyses(caplog):
    """A kernel gives a null analysis no MA map; it says so rather than dropping it silently."""
    studyset = _studyset("g", [ORIGIN, ELSEWHERE], n_null=1)
    with caplog.at_level(logging.WARNING, logger="nimare.meta.kernel"):
        maps = MKDAKernel().transform(studyset, return_type="sparse")
    assert sp_sparse.issparse(maps) and maps.shape[0] == 2
    assert "1 of 3 analyses are null analyses" in caplog.text


@pytest.mark.skipif(not TORCH_INSTALLED, reason="Torch not installed.")
def test_CBMR_drops_null_analyses_and_records_them(caplog):
    """CBMR does not count nulls yet; it leaves them out, warns, and records their ids."""
    from nimare.meta.cbmr import CBMR

    rng = np.random.RandomState(0)
    foci = [rng.randint(-40, 40, size=(5, 3)).astype(float) for _ in range(10)]
    meta = CBMR("~ 1", spline_spacing=100, incidence_threshold=None, n_iter=10, random_state=0)
    with caplog.at_level(logging.WARNING, logger="nimare.studyset.requirements"):
        result = meta.fit(_studyset("g", foci, n_null=2))

    assert meta.inputs_["foci"].shape[0] == 10
    assert "2 of 12 analyses are null analyses" in caplog.text
    assert result.dropped_null_analyses == ["gnull0-a", "gnull1-a"]


def test_null_analyses_survive_slice_and_parquet(tmp_path):
    """compose-runner slices studysets by analysis id and loads them from parquet."""
    studyset = _studyset("g", [ORIGIN, ELSEWHERE], n_null=2)
    sliced = studyset.slice(analyses=["g0-a", "gnull1-a"])
    assert dict(zip(sliced.ids, null_analyses(sliced.store)[sliced._view.index])) == {
        "g0-a": False,
        "gnull1-a": True,
    }

    pytest.importorskip("pyarrow")
    studyset.to_parquet(tmp_path / "ss")
    loaded = Studyset.from_parquet(tmp_path / "ss")
    flags = dict(zip(loaded.ids, null_analyses(loaded.store)[loaded._view.index]))
    assert flags == {"g0-a": False, "g1-a": False, "gnull0-a": True, "gnull1-a": True}
