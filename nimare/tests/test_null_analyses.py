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
THIRD = [-40.0, -20.0, 10.0]


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


def test_MKDAChi2_rejects_two_groups_of_only_null_analyses():
    """With no focus in either group there is nothing to compare, so the fit fails clearly."""
    with pytest.raises(ValueError, match="Neither group has any coordinates"):
        MKDAChi2().fit(_studyset("g1", [], n_null=3), _studyset("g2", [], n_null=2))


def test_dropped_nulls_are_plain_str_and_reach_the_description():
    """The ids are str, and the description (which is written to file) lists them."""
    studyset = _studyset("g", [ORIGIN, ELSEWHERE], n_null=2)
    result = ALE(null_method="approximate").fit(studyset)
    assert all(type(i) is str for i in result.dropped_null_analyses)
    assert "gnull0-a, gnull1-a" in result.description_


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


def _interleaved():
    """Analyses whose sorted ids put a null before, between and after each one with foci.

    The ones with foci are each at their own location with their own sample size, so a map or a
    weight that lands on the wrong analysis shows at the wrong location.
    """
    foci = {"b": ([ORIGIN], 16), "d": ([ELSEWHERE], 64), "f": ([THIRD], 256)}
    studies = []
    for name in "abcdefg":
        points, n = foci.get(name, ([], 100))
        metadata = {"sample_sizes": [n]}
        if name not in foci:
            metadata[OUTCOME_KEY] = NULL_OUTCOME
        points = [{"space": "MNI", "coordinates": list(c)} for c in points]
        studies.append(
            {"id": name, "analyses": [{"id": "a", "metadata": metadata, "points": points}]}
        )
    studyset = Studyset({"id": "interleaved", "studies": studies})
    return studyset, {f"{name}-a": xyz for name, (xyz, _) in foci.items()}


def test_dropped_nulls_leave_each_ma_map_on_its_own_analysis():
    """Dropping nulls cannot shift a map or a weight onto a neighbouring analysis.

    Kernels key their rows by the sorted analysis ids, and a studyset's ids are sorted, so the
    rows are the ids with foci in studyset order, with the nulls simply absent.
    """
    studyset, foci = _interleaved()
    kernel = MKDAKernel(r=4)

    maps = kernel.transform(studyset, return_type="array")
    assert maps.shape[0] == len(foci)
    np.testing.assert_array_equal(
        maps, kernel.transform(studyset.slice(list(foci)), return_type="array")
    )
    for row in range(len(foci)):
        for other, xyz in enumerate(foci.values()):
            hit = maps[row, _masked_index(studyset.masker, xyz[0])]
            assert hit == (1 if other == row else 0)

    # Weights are collected per id and the MA maps per row, so a shift would weight a focus
    # with another analysis's sample size. sqrt(16, 64, 256) = 4, 8, 16, rescaled to sum to 3.
    meta = MKDADensity(
        kernel_transformer=kernel, weighting="sample_size", null_method="approximate"
    )
    result = meta.fit(studyset)
    assert list(meta.inputs_["id"]) == list(foci)
    assert result.dropped_null_analyses == ["a-a", "c-a", "e-a", "g-a"]
    stat = result.get_map("stat", return_type="array")
    for xyz, weight in zip(foci.values(), (4, 8, 16)):
        assert stat[_masked_index(meta.masker, xyz[0])] == pytest.approx(3 * weight / 28)


def test_ml_ma_features_keep_null_analyses_on_their_own_rows():
    """In the ML interface a null kept as a row gets an empty map, and no other row moves."""
    from nimare.ml import MAKernel

    studyset, foci = _interleaved()
    bunch = studyset.to_bunch(missing_coordinates="include")
    maps = MAKernel(MKDAKernel(r=4), source_masker=bunch.masker).fit_transform(
        bunch.data[:, bunch.voxel_columns]
    )

    ids = [str(i) for i in bunch.ids]
    assert sorted(ids) == sorted(studyset.ids)
    for row, analysis in enumerate(ids):
        for other, xyz in foci.items():
            hit = maps[row, _masked_index(bunch.masker, xyz[0])]
            assert hit == (1 if other == analysis else 0), (analysis, other)

    # Dropping them instead (the default) records which, as for any analysis without foci.
    assert studyset.to_bunch().provenance["dropped_ids"] == ["a-a", "c-a", "e-a", "g-a"]


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


def test_a_studyset_of_only_null_analyses_is_reported_before_the_fit_fails(caplog):
    """With every analysis a declared null the fit still fails, but the warning says why."""
    with caplog.at_level(logging.WARNING, logger="nimare.studyset.requirements"):
        with pytest.raises(ValueError, match="no data for 'coordinates'"):
            ALE(null_method="approximate").fit(_studyset("g", [], n_null=2))
    assert "2 of 2 analyses are null analyses" in caplog.text
    assert "gnull0-a, gnull1-a" in caplog.text


def test_warning_names_the_dropped_analyses(caplog):
    """The warning carries the ids, not just the count."""
    with caplog.at_level(logging.WARNING, logger="nimare.studyset.requirements"):
        ALE(null_method="approximate").fit(_studyset("g", [ORIGIN, ELSEWHERE], n_null=2))
    assert "gnull0-a, gnull1-a" in caplog.text


def test_decoder_reports_dropped_null_analyses(testdata_laird_studyset, roi_img, caplog):
    """A decoder's fit returns nothing, so its dropped nulls are on the decoder itself."""
    from nimare.decode.discrete import ROIAssociationDecoder

    doc = testdata_laird_studyset.to_dict()
    nulled = []
    for study in doc["studies"][:2]:
        analysis = study["analyses"][0]
        analysis["points"] = []
        analysis["metadata"] = dict(analysis.get("metadata") or {}, **{OUTCOME_KEY: NULL_OUTCOME})
        nulled.append(f"{study['id']}-{analysis['id']}")
    studyset = Studyset(doc)
    assert sorted(studyset.ids[null_analyses(studyset.store)]) == sorted(nulled)

    decoder = ROIAssociationDecoder(masker=roi_img)
    assert decoder.dropped_null_analyses is None
    with caplog.at_level(logging.WARNING, logger="nimare.studyset.requirements"):
        decoder.fit(studyset)
    assert sorted(decoder.dropped_null_analyses) == sorted(nulled)
    assert "2 of" in caplog.text and nulled[0] in caplog.text
    assert "Decoder.dropped_null_analyses" in caplog.text
