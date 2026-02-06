"""Tests for Studyset-native execution paths."""

import numpy as np

from nimare.io import convert_nimads_to_dataset
from nimare.correct import FDRCorrector
from nimare.meta.cbma import ALE
from nimare.meta.ibma import Stouffers
from nimare.nimads import Studyset
from nimare.reports.base import run_reports
from nimare.transforms import ImageTransformer
from nimare.workflows import CBMAWorkflow, IBMAWorkflow, PairwiseCBMAWorkflow


def test_ale_studyset_parity(testdata_cbma):
    """ALE should accept Studysets and match Dataset outputs."""
    studyset = Studyset.from_dataset(testdata_cbma)

    res_dset = ALE(null_method="approximate").fit(testdata_cbma)
    res_studyset = ALE(null_method="approximate").fit(studyset)

    np.testing.assert_allclose(
        res_dset.get_map("stat", return_type="array"), res_studyset.maps["stat"]
    )
    np.testing.assert_allclose(res_dset.get_map("p", return_type="array"), res_studyset.maps["p"])
    np.testing.assert_allclose(res_dset.get_map("z", return_type="array"), res_studyset.maps["z"])


def test_stouffers_studyset_parity(testdata_ibma):
    """IBMA estimators should accept Studysets and match Dataset outputs."""
    studyset = Studyset.from_dataset(testdata_ibma)

    res_dset = Stouffers().fit(testdata_ibma)
    res_studyset = Stouffers().fit(studyset)

    np.testing.assert_allclose(res_dset.get_map("z", return_type="array"), res_studyset.maps["z"])
    np.testing.assert_allclose(res_dset.get_map("p", return_type="array"), res_studyset.maps["p"])


def test_cbma_workflow_accepts_studyset(tmp_path_factory, testdata_cbma_full):
    """CBMA workflow should run when passed a Studyset."""
    studyset = Studyset.from_dataset(testdata_cbma_full)
    tmpdir = tmp_path_factory.mktemp("test_cbma_workflow_accepts_studyset")

    workflow = CBMAWorkflow(
        estimator="ale",
        corrector="bonferroni",
        diagnostics="jackknife",
        output_dir=tmpdir,
    )
    result = workflow.fit(studyset)
    assert "z" in result.maps


def test_ibma_workflow_accepts_studyset(tmp_path_factory, testdata_ibma):
    """IBMA workflow should run when passed a Studyset."""
    studyset = Studyset.from_dataset(testdata_ibma)
    tmpdir = tmp_path_factory.mktemp("test_ibma_workflow_accepts_studyset")

    workflow = IBMAWorkflow(
        estimator="stouffers",
        corrector="bonferroni",
        diagnostics="jackknife",
        output_dir=tmpdir,
    )
    result = workflow.fit(studyset)
    assert "z" in result.maps


def test_pairwise_cbma_workflow_accepts_studyset(tmp_path_factory, testdata_cbma_full):
    """Pairwise CBMA workflow should run when passed two Studysets."""
    dset1 = testdata_cbma_full.slice(testdata_cbma_full.ids[:10])
    dset2 = testdata_cbma_full.slice(testdata_cbma_full.ids[10:])
    studyset1 = Studyset.from_dataset(dset1)
    studyset2 = Studyset.from_dataset(dset2)
    tmpdir = tmp_path_factory.mktemp("test_pairwise_cbma_workflow_accepts_studyset")

    workflow = PairwiseCBMAWorkflow(
        estimator="mkdachi2",
        corrector="bonferroni",
        diagnostics="jackknife",
        output_dir=tmpdir,
    )
    result = workflow.fit(studyset1, studyset2)
    assert "z_desc-uniformity" in result.maps


def test_dataset_studyset_roundtrip_preserves_images(testdata_ibma):
    """Dataset->Studyset->Dataset conversion should preserve image columns."""
    dset = testdata_ibma.slice(testdata_ibma.ids[:5])
    studyset = Studyset.from_dataset(dset)
    roundtrip = convert_nimads_to_dataset(studyset)

    id_cols = {"id", "study_id", "contrast_id", "space"}
    orig_cols = {
        col for col in dset.images.columns if col not in id_cols and not col.endswith("__relative")
    }
    new_cols = {
        col
        for col in roundtrip.images.columns
        if col not in id_cols and not col.endswith("__relative")
    }

    assert orig_cols.issubset(new_cols)
    for col in orig_cols:
        if dset.images[col].notnull().any():
            assert roundtrip.images[col].notnull().any()


def test_image_transformer_accepts_studyset(testdata_ibma):
    """ImageTransformer should accept Studyset inputs."""
    studyset = Studyset.from_dataset(testdata_ibma)
    transformed = ImageTransformer(target="z").transform(studyset)
    assert "z" in transformed.images.columns


def test_reports_accept_studyset_results(tmp_path_factory, testdata_cbma):
    """Reports should run for Studyset-backed estimator results."""
    studyset = Studyset.from_dataset(testdata_cbma)
    result = ALE(null_method="approximate").fit(studyset)
    result = FDRCorrector(method="indep", alpha=0.05).transform(result)

    out_dir = tmp_path_factory.mktemp("test_reports_accept_studyset_results")
    run_reports(result, out_dir)
    assert (out_dir / "report.html").is_file()
