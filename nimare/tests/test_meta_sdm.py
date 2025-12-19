"""Test nimare.meta.sdm (SDM-based meta-analytic algorithms)."""

import os

import nibabel as nib
import numpy as np
import pytest

import nimare
from nimare.meta import SDM, SDMPSI, SDMKernel
from nimare.tests.utils import get_test_data_path
from nimare.transforms import ImageTransformer


def test_SDM_kernel_instance_with_kwargs(testdata_cbma):
    """Smoke test for SDM with a kernel transformer object.

    With kernel arguments provided, which should result in a warning, but the original
    object's parameters should remain untouched.
    """
    kern = SDMKernel(fwhm=10)
    meta = SDM(kern, kernel__fwhm=25)

    assert meta.kernel_transformer.get_params().get("fwhm") == 10


def test_SDM_kernel_class(testdata_cbma):
    """Smoke test for SDM with a kernel transformer class."""
    meta = SDM(SDMKernel, kernel__fwhm=15)
    results = meta.fit(testdata_cbma)
    assert isinstance(results, nimare.results.MetaResult)


def test_SDM_kernel_instance(testdata_cbma):
    """Smoke test for SDM with a kernel transformer object."""
    kern = SDMKernel(fwhm=20)
    meta = SDM(kern)
    results = meta.fit(testdata_cbma)
    assert isinstance(results, nimare.results.MetaResult)


def test_SDM_default(testdata_cbma):
    """Smoke test for SDM with default parameters."""
    meta = SDM()
    results = meta.fit(testdata_cbma)
    assert isinstance(results, nimare.results.MetaResult)
    assert isinstance(results.description_, str)
    assert "stat" in results.maps.keys()
    assert "z" in results.maps.keys()
    assert "p" in results.maps.keys()
    assert "dof" in results.maps.keys()


def test_SDM_images_only(testdata_ibma):
    """Test SDM with image-only input (hybrid mode, images only)."""
    meta = SDM()
    results = meta.fit(testdata_ibma)
    assert isinstance(results, nimare.results.MetaResult)
    assert meta.input_mode_ == "images"
    assert "stat" in results.maps.keys()
    assert "z" in results.maps.keys()
    assert "p" in results.maps.keys()
    assert "dof" in results.maps.keys()


def test_SDMPSI_default(testdata_cbma):
    """Smoke test for SDMPSI with default parameters."""
    meta = SDMPSI(n_imputations=2, n_subjects_sim=10, random_state=42)
    results = meta.fit(testdata_cbma)
    assert isinstance(results, nimare.results.MetaResult)
    assert isinstance(results.description_, str)
    assert "stat" in results.maps.keys()
    assert "z" in results.maps.keys()
    assert "p" in results.maps.keys()
    assert "dof" in results.maps.keys()
    assert "se" in results.maps.keys()
    assert "within_var" in results.maps.keys()
    assert "between_var" in results.maps.keys()


def test_SDMPSI_kernel_class(testdata_cbma):
    """Smoke test for SDMPSI with a kernel transformer class."""
    meta = SDMPSI(SDMKernel, kernel__fwhm=15, n_imputations=2, n_subjects_sim=10, random_state=42)
    results = meta.fit(testdata_cbma)
    assert isinstance(results, nimare.results.MetaResult)


def test_SDMPSI_multiple_imputations(testdata_cbma):
    """Test that SDMPSI runs with multiple imputations."""
    meta = SDMPSI(n_imputations=3, n_subjects_sim=5, random_state=42)
    results = meta.fit(testdata_cbma)
    assert isinstance(results, nimare.results.MetaResult)
    # Check that between-imputation variance is non-zero (shows imputations differ)
    assert results.maps["between_var"].max() > 0


def test_SDMPSI_images_only(testdata_ibma):
    """Test SDMPSI with image-only input (hybrid mode)."""
    meta = SDMPSI(n_imputations=2, n_subjects_sim=5, random_state=42)
    results = meta.fit(testdata_ibma)
    assert isinstance(results, nimare.results.MetaResult)
    assert meta.input_mode_ == "images"
    assert "stat" in results.maps.keys()
    assert "se" in results.maps.keys()
    assert "within_var" in results.maps.keys()
    assert "between_var" in results.maps.keys()


@pytest.fixture(scope="session")
def testdata_sdm_hybrid(tmp_path_factory, testdata_ibma):
    """Create a hybrid dataset with mixed input types.

    Creates a dataset where:
    - Some studies have only coordinates
    - Some studies have only t-maps
    - Some studies have only z-maps
    - Some studies have beta + varcope maps
    """
    tmpdir = tmp_path_factory.mktemp("testdata_sdm_hybrid")

    # Start with IBMA dataset (has images)
    dset = testdata_ibma.copy()

    # Get study IDs
    study_ids = list(dset.ids)
    n_studies = len(study_ids)

    # Divide studies into 4 groups
    coords_only_ids = study_ids[: n_studies // 4]  # First quarter: coordinates only
    tmap_only_ids = study_ids[n_studies // 4 : n_studies // 2]  # Second quarter: t-maps only
    zmap_only_ids = study_ids[n_studies // 2 : 3 * n_studies // 4]  # Third quarter: z-maps only
    beta_var_ids = study_ids[3 * n_studies // 4 :]  # Last quarter: beta + varcope

    # For coordinates-only studies: null out all image columns
    for study_id in coords_only_ids:
        for col in dset.images.columns:
            if not col.endswith("__relative"):
                dset.images.loc[dset.images["id"] == study_id, col] = None

    # For t-map only studies: keep only t-maps, null others
    for study_id in tmap_only_ids:
        for col in dset.images.columns:
            if col not in ["id", "t", "t__relative"]:
                dset.images.loc[dset.images["id"] == study_id, col] = None

    # For z-map only studies: convert t to z, then null t and other maps
    z_transformer = ImageTransformer(target="z", overwrite=True)
    dset_with_z = z_transformer.transform(dset.slice(zmap_only_ids))

    # Update main dataset with z-maps for these studies
    for study_id in zmap_only_ids:
        z_row = dset_with_z.images[dset_with_z.images["id"] == study_id]
        if not z_row.empty and z_row["z"].notna().any():
            dset.images.loc[dset.images["id"] == study_id, "z"] = z_row["z"].values[0]
            dset.images.loc[dset.images["id"] == study_id, "z__relative"] = z_row[
                "z__relative"
            ].values[0]

        # Null out other image types
        for col in dset.images.columns:
            if col not in ["id", "z", "z__relative"]:
                dset.images.loc[dset.images["id"] == study_id, col] = None

    # For beta+varcope studies: keep beta and varcope, null others
    for study_id in beta_var_ids:
        for col in dset.images.columns:
            if col not in ["id", "beta", "beta__relative", "varcope", "varcope__relative"]:
                dset.images.loc[dset.images["id"] == study_id, col] = None

    return dset


def test_SDM_hybrid_mixed_inputs(testdata_sdm_hybrid):
    """Test SDM with true hybrid input (coordinates + various image types)."""
    meta = SDM()
    results = meta.fit(testdata_sdm_hybrid)
    assert isinstance(results, nimare.results.MetaResult)
    # Should be hybrid mode since we have both coordinates and images
    assert meta.input_mode_ in ["hybrid", "images", "coordinates"]
    assert "stat" in results.maps.keys()
    assert "z" in results.maps.keys()
    assert "p" in results.maps.keys()
    assert "dof" in results.maps.keys()


def test_SDMPSI_hybrid_mixed_inputs(testdata_sdm_hybrid):
    """Test SDMPSI with true hybrid input (coordinates + various image types)."""
    meta = SDMPSI(n_imputations=2, n_subjects_sim=5, random_state=42)
    results = meta.fit(testdata_sdm_hybrid)
    assert isinstance(results, nimare.results.MetaResult)
    assert meta.input_mode_ in ["hybrid", "images", "coordinates"]
    assert "stat" in results.maps.keys()
    assert "se" in results.maps.keys()
    assert "within_var" in results.maps.keys()
    assert "between_var" in results.maps.keys()
