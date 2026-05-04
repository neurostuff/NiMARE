"""Test nimare.workflows."""

import os.path as op

import nibabel as nib
import numpy as np
import pytest

import nimare
from nimare.correct import FDRCorrector, FWECorrector
from nimare.diagnostics import FocusCounter, Jackknife
from nimare.meta.cbma import ALE, ALESubtraction, MKDAChi2, MKDADensity
from nimare.meta.ibma import Fishers, PermutedOLS, Stouffers
from nimare.workflows import (
    CBMAWorkflow,
    ContrastWorkflow,
    IBMAWorkflow,
    PairwiseCBMAWorkflow,
    conjunction_analysis,
)


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


def test_conjunction_analysis_smoke(tmp_path_factory):
    """Run smoke test for conjunction analysis workflow."""
    # Create two 3D arrays with random values
    arr1 = np.random.rand(10, 10, 10)
    arr2 = np.random.rand(10, 10, 10)

    # Create two Nifti1Image objects from the arrays
    img1 = nib.Nifti1Image(arr1, np.eye(4))
    img2 = nib.Nifti1Image(arr2, np.eye(4))

    # Perform conjunction analysis on the two images
    conj_img = conjunction_analysis([img1, img2])

    # Check that the output is a Nifti1Image object
    assert isinstance(conj_img, nib.Nifti1Image)

    # Check that the output has the same shape as the input images
    assert conj_img.shape == img1.shape

    # Check that the output has the correct values
    expected_output = np.minimum.reduce([arr1, arr2])
    np.testing.assert_array_equal(conj_img.get_fdata(), expected_output)

    # Test passing in a list of strings
    tmpdir = tmp_path_factory.mktemp("test_conjunction_analysis_smoke")
    img1_fn = op.join(tmpdir, "image1.nii.gz")
    img2_fn = op.join(tmpdir, "image2.nii.gz")
    img1.to_filename(img1_fn)
    img2.to_filename(img2_fn)

    # Perform conjunction analysis on the two images from nifti files
    conj_img_fromstr = conjunction_analysis([img1_fn, img2_fn])

    # Check that the output has the correct values
    np.testing.assert_array_equal(conj_img.get_fdata(), conj_img_fromstr.get_fdata())

    # Raise error if only one image is provided
    with pytest.raises(ValueError):
        conjunction_analysis([img1])

    # Raise error if invalid image type is provided
    with pytest.raises(ValueError):
        conjunction_analysis([1, 2])


@pytest.mark.parametrize(
    "main_estimator,corrector,pairwise_estimator",
    [
        (
            ALE(),
            FWECorrector(method="montecarlo", n_iters=8, voxel_thresh=0.05, n_cores=1),
            ALESubtraction(n_iters=8, n_cores=1, generate_description=False),
        ),
        (
            MKDADensity(),
            FWECorrector(method="montecarlo", n_iters=8, voxel_thresh=0.05, n_cores=1),
            MKDAChi2(generate_description=False),
        ),
        (
            ALE(),
            FDRCorrector(method="indep", alpha=0.05),
            ALESubtraction(n_iters=8, n_cores=1, generate_description=False),
        ),
        (
            MKDADensity(),
            FDRCorrector(method="indep", alpha=0.05),
            MKDAChi2(generate_description=False),
        ),
    ],
)
def test_contrast_workflow_smoke(
    testdata_cbma_full, main_estimator, corrector, pairwise_estimator
):
    """Contrast Workflow should emit generic contrast/main-effect/conjunction maps."""
    dset1 = testdata_cbma_full.slice(testdata_cbma_full.ids[:10])
    dset2 = testdata_cbma_full.slice(testdata_cbma_full.ids[10:20])

    workflow = ContrastWorkflow(
        main_estimator=main_estimator,
        pairwise_estimator=pairwise_estimator,
        corrector=corrector,
        alpha=0.05,
        n_cores=1,
    )
    result = workflow.fit(dset1, dset2)

    assert isinstance(result, nimare.results.MetaResult)
    assert "z_desc-group1MainEffect" in result.maps
    assert "z_desc-group2MainEffect" in result.maps
    assert "z_desc-conjunction" in result.maps
    assert "z_desc-contrast" in result.maps
    assert "p_desc-contrast" in result.maps
    assert "logp_desc-contrast" in result.maps
    assert "Group 1 main-effect analysis" in result.description_
    assert "Group 2 main-effect analysis" in result.description_
    assert "Masked contrast workflow" in result.description_
    assert "directional inference masks" in result.description_
    assert "laird2005ale" in result.bibtex_
    assert "Frahm_Monimu_Hoffstaedter" in result.bibtex_


def test_contrast_workflow_ale_thresholding_path_smoke(testdata_cbma_full):
    """ALE ContrastWorkflow should run without the removed estimator helper."""
    assert not hasattr(ALE, "jale_corrected_map")

    dset1 = testdata_cbma_full.slice(testdata_cbma_full.ids[:10])
    dset2 = testdata_cbma_full.slice(testdata_cbma_full.ids[10:20])
    workflow = ContrastWorkflow(
        corrector=FWECorrector(method="montecarlo", n_iters=8, voxel_thresh=0.05, n_cores=1),
        pairwise_estimator=ALESubtraction(n_iters=8, n_cores=1, generate_description=False),
        alpha=0.05,
        n_cores=1,
    )

    result = workflow.fit(dset1, dset2)

    assert isinstance(result, nimare.results.MetaResult)
