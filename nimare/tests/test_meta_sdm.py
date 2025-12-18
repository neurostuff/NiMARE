"""Test nimare.meta.sdm (SDM-based meta-analytic algorithms)."""

import nimare
from nimare.meta import SDM, SDMPSI, SDMKernel


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
    meta = SDMPSI(
        SDMKernel, kernel__fwhm=15, n_imputations=2, n_subjects_sim=10, random_state=42
    )
    results = meta.fit(testdata_cbma)
    assert isinstance(results, nimare.results.MetaResult)


def test_SDMPSI_multiple_imputations(testdata_cbma):
    """Test that SDMPSI runs with multiple imputations."""
    meta = SDMPSI(n_imputations=3, n_subjects_sim=5, random_state=42)
    results = meta.fit(testdata_cbma)
    assert isinstance(results, nimare.results.MetaResult)
    # Check that between-imputation variance is non-zero (shows imputations differ)
    assert results.maps["between_var"].max() > 0
