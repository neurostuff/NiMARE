from nimare.meta.cbmr import CBMREstimator, CBMRInference
from nimare.tests.utils import standardize_field
from nimare.meta import models
import logging
import torch
import numpy as np

def test_CBMREstimator(testdata_cbmr_simulated):
    logging.getLogger().setLevel(logging.DEBUG)
    """Unit test for CBMR estimator."""
    dset = standardize_field(dataset=testdata_cbmr_simulated, metadata=["sample_sizes", "avg_age", "schizophrenia_subtype"])
    cbmr = CBMREstimator(
        group_categories=["diagnosis", "drug_status"],
        moderators=["standardized_sample_sizes", "standardized_avg_age", "schizophrenia_subtype"],
        spline_spacing=10,
        model=models.PoissonEstimator,
        penalty=False,
        lr=1e-1,
        tol=1e4,
        device="cpu"
    )
    cbmr.fit(dataset=dset)

def test_CBMRInference(testdata_cbmr_simulated):
    logging.getLogger().setLevel(logging.DEBUG)
    """Unit test for CBMR estimator."""
    dset = standardize_field(dataset=testdata_cbmr_simulated, metadata=["sample_sizes", "avg_age", "schizophrenia_subtype"])
    cbmr = CBMREstimator(
        group_categories=["diagnosis", "drug_status"],
        moderators=["standardized_sample_sizes", "standardized_avg_age"
                    , "schizophrenia_subtype"],
        spline_spacing=10,
        model=models.PoissonEstimator,
        penalty=False,
        lr=1e-1,
        tol=1e4,
        device="cpu",
    )
    # ["standardized_sample_sizes", "standardized_avg_age", "schizophrenia_subtype"],
    cbmr_res = cbmr.fit(dataset=dset)
    inference = CBMRInference(
        CBMRResults=cbmr_res, device="cuda"
    )
    t_con_groups = inference.create_contrast(["schizophrenia_Yes", "schizophrenia_Yes-schizophrenia_No"], type="groups")
    t_con_moderators = inference.create_contrast(["standardized_sample_sizes", "standardized_sample_sizes-standardized_avg_age"], type="moderators")
    contrast_result = inference.compute_contrast(t_con_groups=[[1,-1,0,0],[0,0,1,0]], t_con_moderators=[[1,-1,0,0],[0,0,1,0]])
    # self.maps.schizophrenia_Yes_p_values = ...
    # self.maps.schizophrenia_Yes_chi_square_vals = ...
    # self.tables.standardized_sample_sizes = ...

def test_CBMREstimator_update(testdata_cbmr_simulated):
    cbmr = CBMREstimator(model=models.ClusteredNegativeBinomial, lr=1e-4)

    cbmr._collect_inputs(testdata_cbmr_simulated, drop_invalid=True)
    cbmr._preprocess_input(testdata_cbmr_simulated)
    cbmr_model = cbmr.model(
        spatial_coef_dim=cbmr.inputs_["coef_spline_bases"].shape[1],
        moderators_coef_dim=len(cbmr.moderators) if cbmr.moderators else None,
        groups=cbmr.groups,
        penalty=cbmr.penalty,
        device=cbmr.device,
        )
    
    optimizer = torch.optim.LBFGS(cbmr_model.parameters(), cbmr.lr)
    # load dataset info to torch.tensor
    coef_spline_bases = torch.tensor(cbmr.inputs_["coef_spline_bases"], dtype=torch.float64, device=cbmr.device)
    if cbmr.moderators:
        moderators_by_group_tensor = dict()
        for group in cbmr_model.groups:
            moderators_tensor = torch.tensor(
                cbmr_model.inputs_["moderators_by_group"][group], dtype=torch.float64, device=cbmr.device
            )
            moderators_by_group_tensor[group] = moderators_tensor
    else:
        moderators_by_group_tensor = None
    foci_per_voxel_tensor, foci_per_study_tensor = dict(), dict()
    for group in cbmr_model.groups:
        group_foci_per_voxel_tensor = torch.tensor(
            cbmr.inputs_["foci_per_voxel"][group], dtype=torch.float64, device=cbmr.device
        )
        group_foci_per_study_tensor = torch.tensor(
            cbmr.inputs_["foci_per_study"][group], dtype=torch.float64, device=cbmr.device
        )
        foci_per_voxel_tensor[group] = group_foci_per_voxel_tensor
        foci_per_study_tensor[group] = group_foci_per_study_tensor
    optimizer = torch.optim.LBFGS(cbmr_model.parameters(), cbmr.lr)
    if cbmr.iter == 0:
        prev_loss = torch.tensor(float("inf"))  # initialization loss difference
    
    loss = cbmr._update(
                    cbmr_model,
                    optimizer,
                    torch.tensor(cbmr.inputs_["coef_spline_bases"], dtype=torch.float64, device=cbmr.device),
                    moderators_by_group_tensor,
                    foci_per_voxel_tensor,
                    foci_per_study_tensor,
                    prev_loss,
            )
    
    # deliberately set the first spatial coefficient to nan
    nan_coef = torch.tensor(cbmr_model.spatial_coef_linears['default'].weight)
    nan_coef[:, 0] = float('nan')
    cbmr_model.spatial_coef_linears['default'].weight = torch.nn.Parameter(nan_coef)
    
    loss = cbmr._update(
                    cbmr_model,
                    optimizer,
                    torch.tensor(cbmr.inputs_["coef_spline_bases"], dtype=torch.float64, device=cbmr.device),
                    moderators_by_group_tensor,
                    foci_per_voxel_tensor,
                    foci_per_study_tensor,
                    prev_loss,
            )




