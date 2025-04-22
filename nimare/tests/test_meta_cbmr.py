"""Tests for CBMR meta-analytic methods."""

import logging
import warnings

import pytest

try:
    import torch
except ImportError:
    warnings.warn("Torch not installed. CBMR tests will be skipped.")
    TORCH_INSTALLED = False
else:
    TORCH_INSTALLED = True
    from nimare.meta import models
    from nimare.meta.cbmr import CBMREstimator, CBMRInference

import nimare
from nimare.correct import FDRCorrector, FWECorrector
from nimare.transforms import StandardizeField

# numba has a lot of debug messages that are not useful for testing
logging.getLogger("numba").setLevel(logging.WARNING)
# indexed_gzip has a few debug messages that are not useful for testing
logging.getLogger("indexed_gzip").setLevel(logging.WARNING)


if TORCH_INSTALLED:

    @pytest.fixture(
        scope="session",
        params=[
            pytest.param(models.PoissonEstimator, id="Poisson"),
            pytest.param(models.NegativeBinomialEstimator, id="NegativeBinomial"),
            pytest.param(
                models.ClusteredNegativeBinomialEstimator, id="ClusteredNegativeBinomial"
            ),
        ],
    )
    def model(request):
        """CBMR models."""
        return request.param

else:
    model = None


@pytest.fixture(scope="session")
def cbmr_result(testdata_cbmr_simulated, model):
    """Test CBMR estimator."""
    dset = StandardizeField(fields=["sample_sizes", "avg_age", "schizophrenia_subtype"]).transform(
        testdata_cbmr_simulated
    )

    cbmr = CBMREstimator(
        group_categories=["diagnosis", "drug_status"],
        moderators=["standardized_sample_sizes", "standardized_avg_age", "schizophrenia_subtype"],
        spline_spacing=100,
        model=model,
        penalty=False,
        n_iter=1000,
        lr=1,
        tol=1e4,
        device="cpu",
    )
    res = cbmr.fit(dataset=dset)
    assert isinstance(res, nimare.results.MetaResult)
    assert isinstance(res.description_, str)

    return res


@pytest.fixture(scope="session")
def inference_results(testdata_cbmr_simulated, cbmr_result):
    """Test inference results for CBMR estimator."""
    inference = CBMRInference(device="cpu")
    inference.fit(cbmr_result)
    t_con_groups = inference.create_contrast(
        [
            "DepressionYes-DepressionNo",
        ],
        source="groups",
    )
    t_con_moderators = inference.create_contrast(
        ["standardized_sample_sizes-standardized_avg_age"],
        source="moderators",
    )
    contrast_result = inference.transform(
        t_con_groups=t_con_groups, t_con_moderators=t_con_moderators
    )

    return contrast_result


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(FWECorrector(method="bonferroni"), id="bonferroni"),
        pytest.param(FDRCorrector(method="indep"), id="indep"),
        pytest.param(FDRCorrector(method="negcorr"), id="negcorr"),
    ],
)
def corrector(request):
    """Corrector classes."""
    return request.param


def test_cbmr_estimator(cbmr_result):
    """Unit test for CBMR estimator."""
    assert isinstance(cbmr_result, nimare.results.MetaResult)


def test_cbmr_inference(inference_results):
    """Unit test for CBMR inference."""
    assert isinstance(inference_results, nimare.results.MetaResult)


def test_cbmr_correctors(inference_results, corrector):
    """Unit test for Correctors that work with CBMR."""
    corrected_results = corrector.transform(inference_results)
    assert isinstance(corrected_results, nimare.results.MetaResult)


def test_firth_penalty(testdata_cbmr_simulated):
    """Unit test for Firth penalty."""
    dset = StandardizeField(fields=["sample_sizes", "avg_age", "schizophrenia_subtype"]).transform(
        testdata_cbmr_simulated
    )
    cbmr = CBMREstimator(
        group_categories=["diagnosis", "drug_status"],
        moderators=["standardized_sample_sizes", "standardized_avg_age", "schizophrenia_subtype"],
        spline_spacing=100,
        model=models.PoissonEstimator,
        penalty=True,
        lr=1,
        tol=1e4,
        device="cpu",
    )
    res = cbmr.fit(dataset=dset)
    assert isinstance(res, nimare.results.MetaResult)


def test_moderators_none(testdata_cbmr_simulated):
    """Unit test for Firth penalty."""
    dset = StandardizeField(fields=["sample_sizes", "avg_age", "schizophrenia_subtype"]).transform(
        testdata_cbmr_simulated
    )
    cbmr = CBMREstimator(
        group_categories=["diagnosis", "drug_status"],
        moderators=None,
        spline_spacing=100,
        model=models.PoissonEstimator,
        penalty=False,
        lr=1,
        tol=1e4,
        device="cpu",
    )
    res = cbmr.fit(dataset=dset)
    assert isinstance(res, nimare.results.MetaResult)
    inference = CBMRInference(device="cpu")
    inference.fit(res)

    t_con_groups = inference.create_contrast(
        [
            "DepressionYes",
        ],
        source="groups",
    )
    inference_results = inference.transform(t_con_groups=t_con_groups)

    assert isinstance(inference_results, nimare.results.MetaResult)


def test_CBMREstimator_update(testdata_cbmr_simulated):
    """Unit test for CBMR estimator update function."""
    testdata_cbmr_simulated = StandardizeField(
        fields=["sample_sizes", "avg_age", "schizophrenia_subtype"]
    ).transform(testdata_cbmr_simulated)
    cbmr = CBMREstimator(
        moderators=["standardized_sample_sizes", "standardized_avg_age", "schizophrenia_subtype"],
        model=models.PoissonEstimator,
        lr=1,
    )

    cbmr._collect_inputs(testdata_cbmr_simulated, drop_invalid=True)
    cbmr._preprocess_input(testdata_cbmr_simulated)

    # fit the model
    init_weight_kwargs = {
        "groups": cbmr.groups,
        "moderators": cbmr.moderators,
        "spatial_coef_dim": cbmr.inputs_["coef_spline_bases"].shape[1],
        "moderators_coef_dim": len(cbmr.moderators) if cbmr.moderators else None,
    }

    cbmr.model.init_weights(**init_weight_kwargs)
    optimizer = torch.optim.LBFGS(cbmr.model.parameters(), cbmr.lr)
    # load dataset info to torch.tensor
    if cbmr.moderators:
        moderators_by_group_tensor = dict()
        for group in cbmr.model.groups:
            moderators_tensor = torch.tensor(
                cbmr.inputs_["moderators_by_group"][group],
                dtype=torch.float64,
                device=cbmr.device,
            )
            moderators_by_group_tensor[group] = moderators_tensor
    else:
        moderators_by_group_tensor = None
    foci_per_voxel_tensor, foci_per_study_tensor = dict(), dict()
    for group in cbmr.model.groups:
        group_foci_per_voxel_tensor = torch.tensor(
            cbmr.inputs_["foci_per_voxel"][group], dtype=torch.float64, device=cbmr.device
        )
        group_foci_per_study_tensor = torch.tensor(
            cbmr.inputs_["foci_per_study"][group], dtype=torch.float64, device=cbmr.device
        )
        foci_per_voxel_tensor[group] = group_foci_per_voxel_tensor
        foci_per_study_tensor[group] = group_foci_per_study_tensor

    if cbmr.iter == 0:
        prev_loss = torch.tensor(float("inf"))  # initialization loss difference

    cbmr.model._update(
        optimizer,
        torch.tensor(cbmr.inputs_["coef_spline_bases"], dtype=torch.float64, device=cbmr.device),
        moderators_by_group_tensor,
        foci_per_voxel_tensor,
        foci_per_study_tensor,
        prev_loss,
    )
    # deliberately set the first spatial coefficient to nan
    for group in cbmr.model.groups:
        nan_coef = torch.tensor(cbmr.model.spatial_coef_linears[group].weight)
        nan_coef[:, 0] = float("nan")
        cbmr.model.spatial_coef_linears[group].weight = torch.nn.Parameter(nan_coef)

    # Expect exceptions when one of the spatial coefficients is nan.
    with pytest.raises(ValueError):
        cbmr.model._update(
            optimizer,
            torch.tensor(
                cbmr.inputs_["coef_spline_bases"], dtype=torch.float64, device=cbmr.device
            ),
            moderators_by_group_tensor,
            foci_per_voxel_tensor,
            foci_per_study_tensor,
            prev_loss,
        )


def test_StandardizeField(testdata_cbmr_simulated):
    """Unit test for StandardizeField."""
    dset = StandardizeField(fields=["sample_sizes", "avg_age"]).transform(testdata_cbmr_simulated)
    assert isinstance(dset, nimare.dataset.Dataset)
    assert "standardized_sample_sizes" in dset.annotations
    assert "standardized_avg_age" in dset.annotations
    assert dset.annotations["standardized_sample_sizes"].mean() == pytest.approx(0.0, abs=1e-3)
    assert dset.annotations["standardized_sample_sizes"].std() == pytest.approx(1.0, abs=1e-3)
    assert dset.annotations["standardized_avg_age"].mean() == pytest.approx(0.0, abs=1e-3)
    assert dset.annotations["standardized_avg_age"].std() == pytest.approx(1.0, abs=1e-3)


@pytest.mark.cbmr_importerror
def test_cbmr_importerror():
    """Test that ImportErrors are raised when torch is not installed."""
    with pytest.raises(ImportError):
        from nimare.meta.cbmr import CBMREstimator

        CBMREstimator()

    with pytest.raises(ImportError):
        from nimare.meta.cbmr import CBMRInference

        CBMRInference()

    with pytest.raises(ImportError):
        from nimare.meta.models import GeneralLinearModelEstimator

        GeneralLinearModelEstimator()

    with pytest.raises(ImportError):
        from nimare.meta.models import PoissonEstimator

        PoissonEstimator()

    with pytest.raises(ImportError):
        from nimare.meta.models import NegativeBinomialEstimator

        NegativeBinomialEstimator()

    with pytest.raises(ImportError):
        from nimare.meta.models import ClusteredNegativeBinomialEstimator

        ClusteredNegativeBinomialEstimator()
