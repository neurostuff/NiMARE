"""Test nimare.workflows."""
import os.path as op

import pytest

import nimare
from nimare import cli, workflows
from nimare.correct import FWECorrector
from nimare.diagnostics import FocusCounter, Jackknife
from nimare.meta.cbma import ALE, ALESubtraction, MKDAChi2
from nimare.meta.ibma import Fishers, PermutedOLS, Stouffers
from nimare.tests.utils import get_test_data_path
from nimare.workflows import CBMAWorkflow, IBMAWorkflow, PairwiseCBMAWorkflow


def test_ale_workflow_function_smoke(tmp_path_factory):
    """Run smoke test for Sleuth ALE workflow."""
    tmpdir = tmp_path_factory.mktemp("test_ale_workflow_function_smoke")
    sleuth_file = op.join(get_test_data_path(), "test_sleuth_file.txt")
    prefix = "test"

    # The same test is run with both workflow function and CLI
    workflows.ale_sleuth_workflow(
        sleuth_file, output_dir=tmpdir, prefix=prefix, n_iters=10, n_cores=1
    )
    assert op.isfile(op.join(tmpdir, f"{prefix}_input_coordinates.txt"))


def test_ale_workflow_cli_smoke(tmp_path_factory):
    """Run smoke test for Sleuth ALE workflow."""
    tmpdir = tmp_path_factory.mktemp("test_ale_workflow_cli_smoke")
    sleuth_file = op.join(get_test_data_path(), "test_sleuth_file.txt")
    prefix = "test"

    cli._main(
        [
            "ale",
            "--output_dir",
            str(tmpdir),
            "--prefix",
            prefix,
            "--n_iters",
            "10",
            "--n_cores",
            "1",
            sleuth_file,
        ]
    )
    assert op.isfile(op.join(tmpdir, f"{prefix}_input_coordinates.txt"))


def test_ale_workflow_function_smoke_2(tmp_path_factory):
    """Run smoke test for Sleuth ALE workflow with subtraction analysis."""
    tmpdir = tmp_path_factory.mktemp("test_ale_workflow_function_smoke_2")
    sleuth_file = op.join(get_test_data_path(), "test_sleuth_file.txt")
    prefix = "test"

    # The same test is run with both workflow function and CLI
    workflows.ale_sleuth_workflow(
        sleuth_file,
        sleuth_file2=sleuth_file,
        output_dir=tmpdir,
        prefix=prefix,
        n_iters=10,
        n_cores=1,
    )
    assert op.isfile(op.join(tmpdir, f"{prefix}_group2_input_coordinates.txt"))


def test_ale_workflow_cli_smoke_2(tmp_path_factory):
    """Run smoke test for Sleuth ALE workflow with subtraction analysis."""
    tmpdir = tmp_path_factory.mktemp("test_ale_workflow_cli_smoke_2")
    sleuth_file = op.join(get_test_data_path(), "test_sleuth_file.txt")
    prefix = "test"
    cli._main(
        [
            "ale",
            "--output_dir",
            str(tmpdir),
            "--prefix",
            prefix,
            "--n_iters",
            "10",
            "--n_cores",
            "1",
            "--file2",
            sleuth_file,
            sleuth_file,
        ]
    )
    assert op.isfile(op.join(tmpdir, f"{prefix}_group2_input_coordinates.txt"))


@pytest.mark.parametrize(
    "estimator,corrector,diagnostics",
    [
        (ALE, FWECorrector(method="montecarlo", n_iters=10), [Jackknife]),
        ("ales", "bonferroni", Jackknife),
        ("ale", "bonferroni", [Jackknife, FocusCounter]),
        ("kda", "fdr", Jackknife),
        ("mkdadensity", "fdr", "focuscounter"),
        (MKDAChi2, "montecarlo", None),
        (Fishers, "montecarlo", "jackknife"),
    ],
)
def test_cbma_workflow_smoke(
    tmp_path_factory,
    testdata_cbma_full,
    estimator,
    corrector,
    diagnostics,
):
    """Run smoke test for CBMA workflow."""
    tmpdir = tmp_path_factory.mktemp("test_cbma_workflow_smoke")

    if estimator == MKDAChi2:
        with pytest.raises(AttributeError):
            CBMAWorkflow(estimator=estimator, corrector=corrector, diagnostics=diagnostics)
    elif estimator == Fishers:
        with pytest.raises((AttributeError, ValueError)):
            CBMAWorkflow(estimator=estimator, corrector=corrector, diagnostics=diagnostics)
    elif estimator == "ales":
        with pytest.raises(ValueError):
            CBMAWorkflow(estimator=estimator, corrector=corrector, diagnostics=diagnostics)
    else:
        workflow = CBMAWorkflow(
            estimator=estimator,
            corrector=corrector,
            diagnostics=diagnostics,
            output_dir=tmpdir,
        )
        cres = workflow.fit(testdata_cbma_full)

        assert isinstance(cres, nimare.results.MetaResult)
        assert op.isfile(op.join(tmpdir, "boilerplate.txt"))
        assert op.isfile(op.join(tmpdir, "references.bib"))

        for imgtype in cres.maps.keys():
            filename = f"{imgtype}.nii.gz"
            outpath = op.join(tmpdir, filename)
            # For ALE maps are None
            if not cres.maps[imgtype] is None:
                assert op.isfile(outpath)

        for tabletype in cres.tables.keys():
            filename = f"{tabletype}.tsv"
            outpath = op.join(tmpdir, filename)
            # For ALE tables are None
            if not cres.tables[tabletype] is None:
                assert op.isfile(outpath)


@pytest.mark.parametrize(
    "estimator,corrector,diagnostics",
    [
        (MKDAChi2, FWECorrector(method="montecarlo", n_iters=10), [FocusCounter]),
        ("mkdachi", "bonferroni", FocusCounter),
        ("mkdachi2", "bonferroni", "jackknife"),
        (ALESubtraction(n_iters=10), "fdr", Jackknife(voxel_thresh=0.01)),
        (ALE, "montecarlo", None),
        (Fishers, "montecarlo", "jackknife"),
    ],
)
def test_pairwise_cbma_workflow_smoke(
    tmp_path_factory,
    testdata_cbma_full,
    estimator,
    corrector,
    diagnostics,
):
    """Run smoke test for CBMA workflow."""
    tmpdir = tmp_path_factory.mktemp("test_pairwise_cbma_workflow_smoke")

    dset1 = testdata_cbma_full.slice(testdata_cbma_full.ids[:10])
    dset2 = testdata_cbma_full.slice(testdata_cbma_full.ids[10:])
    if estimator in [ALE, "mkdachi"]:
        with pytest.raises(ValueError):
            PairwiseCBMAWorkflow(estimator=estimator, corrector=corrector, diagnostics=diagnostics)
    elif estimator == Fishers:
        with pytest.raises((AttributeError, ValueError)):
            PairwiseCBMAWorkflow(estimator=estimator, corrector=corrector, diagnostics=diagnostics)
    else:
        workflow = PairwiseCBMAWorkflow(
            estimator=estimator,
            corrector=corrector,
            diagnostics=diagnostics,
            output_dir=tmpdir,
        )
        cres = workflow.fit(dset1, dset2)

        assert isinstance(cres, nimare.results.MetaResult)
        assert op.isfile(op.join(tmpdir, "boilerplate.txt"))
        assert op.isfile(op.join(tmpdir, "references.bib"))

        for imgtype in cres.maps.keys():
            filename = f"{imgtype}.nii.gz"
            outpath = op.join(tmpdir, filename)
            # For MKDAChi2 maps are None
            if cres.maps[imgtype] is not None:
                assert op.isfile(outpath)

        for tabletype in cres.tables.keys():
            filename = f"{tabletype}.tsv"
            outpath = op.join(tmpdir, filename)
            # For MKDAChi2 tables are None
            if cres.tables[tabletype] is not None:
                assert op.isfile(outpath)


@pytest.mark.parametrize(
    "estimator,corrector,diagnostics",
    [
        (PermutedOLS, FWECorrector(method="montecarlo", n_iters=10), "jackknife"),
        (Stouffers, "bonferroni", "jackknife"),
        ("fishers", "fdr", "jackknife"),
    ],
)
def test_ibma_workflow_smoke(
    tmp_path_factory,
    testdata_ibma,
    estimator,
    corrector,
    diagnostics,
):
    """Run smoke test for CBMA workflow."""
    tmpdir = tmp_path_factory.mktemp("test_ibma_workflow_smoke")

    workflow = IBMAWorkflow(
        estimator=estimator,
        corrector=corrector,
        diagnostics=diagnostics,
        output_dir=tmpdir,
    )
    cres = workflow.fit(testdata_ibma)

    assert isinstance(cres, nimare.results.MetaResult)
    assert op.isfile(op.join(tmpdir, "boilerplate.txt"))
    assert op.isfile(op.join(tmpdir, "references.bib"))

    for imgtype in cres.maps.keys():
        filename = f"{imgtype}.nii.gz"
        outpath = op.join(tmpdir, filename)
        assert op.isfile(outpath)

    for tabletype in cres.tables.keys():
        filename = f"{tabletype}.tsv"
        outpath = op.join(tmpdir, filename)
        assert op.isfile(outpath)
