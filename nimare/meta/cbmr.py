from nimare.base import Estimator
from nimare.utils import get_masker, B_spline_bases
import nibabel as nib
import numpy as np
import pandas as pd
import scipy
from nimare.utils import mm2vox
from nimare.diagnostics import FocusFilter
from nimare.meta import models
import torch
import functorch
import logging
import copy

LGR = logging.getLogger(__name__)


class CBMREstimator(Estimator):
    """Coordinate-based meta-regression with a spatial model.

    Parameters
    ----------
    group_categories : :obj:`~str` or obj:`~list` or obj:`~None`, optional
        CBMR allows dataset to be categorized into mutiple groups, according to group names.
        Default is one-group CBMR.
    moderators : :obj:`~str` or obj:`~list` or obj:`~None`, optional
        CBMR can accommodate study-level moderators (e.g. sample size, year of publication).
        Default is CBMR without study-level moderators.
    model : {"Poisson", "NB", "clustered NB"}, optional
        Stochastic models in CBMR. The available options are

        ======================= =================================================================
        "Poisson" (default)     This is the most efficient and widely used method, but slightly
                                less accurate, because Poisson model is an approximation for
                                low-rate Binomial data, but cannot account over-dispersion in
                                foci counts and may underestimate the standard error.

        "NB"                    This method is much slower and less stable, but slightly more
                                accurate. Negative Binomial (NB) model asserts foci counts follow
                                a NB distribution, and allows for anticipated excess variance
                                relative to Poisson (there's an overdispersion parameter shared 
                                by all studies and all voxels to index excess variance).

        "clustered NB"          This method is also an efficient but less accurate approach.
                                Clustered NB model is "random effect" Poisson model, which asserts
                                that the random effects are latent characteristics of each study,
                                and represent a shared effect over the entire brain for a given
                                study.
        ======================= =================================================================
    penalty: :obj:`~bool`, optional
    Currently, the only available option is Firth-type penalty, which penalizes likelihood function
    by Jeffrey's invariant prior and guarantees convergent estimates.
    spline_spacing: :obj:`~int`, optional
    Spatial structure of foci counts is parameterized by coefficient of cubic B-spline bases
    in CBMR. Spatial smoothness in CBMR is determined by spline spacing, which is shared across
    x,y,z dimension.
    Default is 10 (20mm with 2mm brain atlas template).
    n_iters: :obj:`int`, optional
        Number of iterations limit in optimisation of log-likelihood function.
        Default is 10000.
    lr: :obj:`float`, optional
        Learning rate in optimization of log-likelihood function.
        Default is 1e-2 for Poisson and clustered NB model, and 1e-3 for NB model.
    tol: :obj:`float`, optional
        Stopping criteria w.r.t difference of log-likelihood function in two consecutive
        iterations.
        Default is 1e-2
    device: :obj:`string`, optional
        Device type ('cpu' or 'cuda') represents the device on which operations will be allocated
        Default is 'cpu'
    **kwargs
        Keyword arguments. Arguments for the Estimator can be assigned here,
        Another optional argument is ``mask``.

    Attributes
    ----------
    masker : :class:`~nilearn.input_data.NiftiMasker` or similar
        Masker object.
    inputs_ : :obj:`dict`
        Inputs to the Estimator. For CBMR estimators, there is only multiple keys:
        coordinates,
        mask_img (Niftiimage of brain mask),
        id (study id),
        studies_by_groups (study id categorized by groups),
        all_group_moderators (study-level moderators categorized by groups if exist),
        coef_spline_bases (spatial matrix of coefficient of cubic B-spline
        bases in x,y,z dimension),
        foci_per_voxel (voxelwise sum of foci count across studies, categorized by groups),
        foci_per_study (study-wise sum of foci count across space, categorized by groups).


    Notes
    -----
    Available correction methods: :meth:`~nimare.meta.cbmr.CBMRInference`.
    """

    _required_inputs = {"coordinates": ("coordinates", None)}

    def __init__(
        self,
        group_categories=None,
        moderators=None,
        mask=None,
        spline_spacing=10,
        model=models.Poisson,
        penalty=False,
        n_iter=1000,
        lr=1e-2,
        tol=1e-2,
        device="cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if mask is not None:
            mask = get_masker(mask)
        self.masker = mask

        self.group_categories = group_categories
        self.moderators = moderators

        self.spline_spacing = spline_spacing
        self.model = model
        self.penalty = penalty
        self.n_iter = n_iter
        self.lr = lr
        self.tol = tol
        self.device = device
        if self.device == "cuda" and not torch.cuda.is_available():
            LGR.debug("cuda not found, use device cpu")
            self.device = "cpu"

        # Initialize optimisation parameters
        self.iter = 0

    def _preprocess_input(self, dataset):
        """Mask required input images using either the Dataset's mask or the Estimator's.

        Also, categorize study id, voxelwise sum of foci counts across studies, study-wise sum of
        foci counts across space into multiple groups. And summarize study-level moderators into
        multiple groups (if exist).

        Parameters
        ----------
        dataset : :obj:`~nimare.dataset.Dataset`
            In this method, the Dataset is used to (1) select the appropriate mask image,
            (2) categorize it into multiple groups according to group type in annotations,
            (3) summarize group-wise study id, foci per voxel, foci per study, moderators
            (if exist),
            (4) extract sample size metadata and use it as one of study-level moderators.

        Attributes
        ----------
        inputs_ : :obj:`dict`
            Specifically, (1) a “mask_img” key will be added (Niftiimage of brain mask),
            (2) an 'id' key will be added (id of all studies in the dataset),
            (3) an 'studies_by_group' key will be added (study id categorized by groups),
            (4) a 'coef_spline_bases' key will be added (spatial matrix of coefficient of cubic
            B-spline bases in x,y,z dimension),
            (5) an 'foci_per_voxel' key will be added (voxelwise sum of foci count across
            studies, categorized by groups),
            (6) an 'foci_per_study' key will be added (study-wise sum of foci count across
            space, categorized by groups),
            (7) an 'moderators_by_group' key may be added if study-level moderators exists
        """
        masker = self.masker or dataset.masker

        mask_img = masker.mask_img or masker.labels_img
        if isinstance(mask_img, str):
            mask_img = nib.load(mask_img)
        self.inputs_["mask_img"] = mask_img
        
        # generate spatial matrix of coefficient of cubic B-spline bases in x,y,z dimension
        coef_spline_bases = B_spline_bases(
            masker_voxels=mask_img._dataobj, spacing=self.spline_spacing
        )
        self.inputs_["coef_spline_bases"] = coef_spline_bases
        
        for name, (type_, _) in self._required_inputs.items():
            if type_ == "coordinates":
                # remove dataset coordinates outside of mask
                focus_filter = FocusFilter(mask=masker)
                dataset = focus_filter.transform(dataset)
                valid_dset_annotations = dataset.annotations[
                    dataset.annotations["id"].isin(self.inputs_["id"])
                ]
                studies_by_group = dict()
                if self.group_categories is None:
                    studies_by_group["default"] = (
                        valid_dset_annotations["study_id"].unique().tolist()
                    )
                    unique_groups = ["default"]
                elif isinstance(self.group_categories, str):
                    if self.group_categories not in valid_dset_annotations.columns:
                        raise ValueError(
                            f"Category_names: {self.group_categories} does not exist in the dataset"
                        )
                    else:
                        unique_groups = list(valid_dset_annotations[self.group_categories].unique())
                        for group in unique_groups:
                            group_study_id_bool = valid_dset_annotations[self.group_categories] == group
                            group_study_id = valid_dset_annotations.loc[group_study_id_bool][
                                "study_id"
                            ]
                            studies_by_group[group] = group_study_id.unique().tolist()
                elif isinstance(self.group_categories, list):
                    missing_categories = set(self.group_categories) - set(dataset.annotations.columns) 
                    if missing_categories:
                        raise ValueError(
                            f"Category_names: {missing_categories} do/does not exist in the dataset."
                        )
                    unique_groups = (
                        valid_dset_annotations[self.group_categories].drop_duplicates().values.tolist()
                    )
                    for group in unique_groups:
                        group_study_id_bool = (
                            valid_dset_annotations[self.group_categories] == group
                        ).all(axis=1)
                        group_study_id = valid_dset_annotations.loc[group_study_id_bool][
                            "study_id"
                        ]
                        studies_by_group["_".join(group)] = group_study_id.unique().tolist()
                self.inputs_["studies_by_group"] = studies_by_group
                self.groups = self.inputs_["studies_by_group"].keys()
                # collect studywise moderators if specficed
                if self.moderators:
                    if isinstance(self.moderators, str):
                        self.moderators = [
                            self.moderators
                        ]  # convert moderators to a single-element list if it's a string
                    moderators_by_group = dict()
                    for group in self.groups:
                        df_group = valid_dset_annotations.loc[
                            valid_dset_annotations["study_id"].isin(studies_by_group[group])
                        ]
                        group_moderators = np.stack(
                            [df_group[moderator_name] for moderator_name in self.moderators],
                            axis=1,
                        )
                        moderators_by_group[group] = group_moderators
                    self.inputs_["moderators_by_group"] = moderators_by_group
            
                foci_per_voxel, foci_per_study = dict(), dict()
                for group in self.groups:
                    group_study_id = studies_by_group[group]
                    group_coordinates = dataset.coordinates.loc[
                        dataset.coordinates["study_id"].isin(group_study_id)
                    ]
                    # Group-wise foci coordinates
                    # Calculate IJK matrix indices for target mask
                    # Mask space is assumed to be the same as the Dataset's space
                    group_xyz = group_coordinates[["x", "y", "z"]].values
                    group_ijk = mm2vox(group_xyz, mask_img.affine)
                    group_foci_per_voxel = np.zeros(mask_img.shape, dtype=np.int32)
                    for ijk in group_ijk:
                        group_foci_per_voxel[ijk[0], ijk[1], ijk[2]] += 1
                    # will not work with maskers that aren't NiftiMaskers
                    group_foci_per_voxel = nib.Nifti1Image(
                        group_foci_per_voxel, mask_img.affine, mask_img.header
                    )
                    group_foci_per_voxel = masker.transform(group_foci_per_voxel).transpose()
                    # number of foci per voxel/study
                    n_group_study = len(group_study_id)
                    group_foci_per_study = np.array(
                        [(group_coordinates["study_id"] == i).sum() for i in group_study_id]
                    )
                    group_foci_per_study = group_foci_per_study.reshape((n_group_study, 1))

                    foci_per_voxel[group] = group_foci_per_voxel
                    foci_per_study[group] = group_foci_per_study

                self.inputs_["foci_per_voxel"] = foci_per_voxel
                self.inputs_["foci_per_study"] = foci_per_study

    def _update(
        self,
        model,
        optimizer,
        coef_spline_bases,
        moderators,
        foci_per_voxel,
        foci_per_study,
        prev_loss,
        gamma=0.999,
    ):
        """One iteration in optimization with L-BFGS.

        Adjust learning rate based on the number of iteration (with learning rate decay parameter
        `gamma`, default value is 0.999).Reset L-BFGS optimizer if NaN occurs.
        """
        self.iter += 1
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=gamma
        )  # learning rate decay

        def closure():
            optimizer.zero_grad()
            loss = model(coef_spline_bases, moderators, foci_per_voxel, foci_per_study)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        scheduler.step()
        # reset the L-BFGS params if NaN appears in coefficient of regression
        if any(
            [
                torch.any(torch.isnan(model.spatial_coef_linears[group].weight))
                for group in self.groups
            ]
        ):
            if self.iter == 1:  # NaN occurs in the first iteration
                raise ValueError(
                    """The current learing rate {str(self.lr)} gives rise to NaN values, adjust
                    to a smaller value."""
                )
            spatial_coef_linears, overdispersion_sqrt, overdispersion = dict(), dict(), dict()
            for group in self.groups:
    
                group_spatial_linear = torch.nn.Linear(model.spatial_coef_dim, 1, bias=False).double()
                group_spatial_linear.weight = torch.nn.Parameter(
                    self.last_state["spatial_coef_linears." + group + ".weight"]
                )
                spatial_coef_linears[group] = group_spatial_linear

                if isinstance(model, models.NegativeBinomial):
                    group_overdispersion_sqrt = torch.nn.Parameter(
                        self.last_state["overdispersion_sqrt." + group]
                    )
                    overdispersion_sqrt[group] = group_overdispersion_sqrt
                elif isinstance(model, models.ClusteredNegativeBinomial):
                    group_overdispersion = torch.nn.Parameter(self.last_state["overdispersion." + group])
                    overdispersion[group] = group_overdispersion

            model.spatial_coef_linears = torch.nn.ModuleDict(spatial_coef_linears)
            if isinstance(model, models.NegativeBinomial):
                model.overdispersion_sqrt = torch.nn.ParameterDict(overdispersion_sqrt)
            elif isinstance(model, models.ClusteredNegativeBinomial):
                model.overdispersion = torch.nn.ParameterDict(overdispersion)

            LGR.debug("Reset L-BFGS optimizer......")
        else:
            self.last_state = copy.deepcopy(
                model.state_dict()
            ) 

        return loss

    def _optimizer(self, model, lr, tol, n_iter, device):
        """Optimize regression coefficient of CBMR via L-BFGS algorithm.

        Optimization terminates if the absolute value of difference of log-likelihood in
        two consecutive iterations is below `tol`

        Parameters
        ----------
        model : :obj:`~nimare.dataset.Dataset`
            Stochastic model used in CBMR.
        lr  : :obj:`~float`
            Learning rate of L-BFGS.
        tol : :obj:`~float`
            Stopping criteria of L-BFGS.
        n_iter : :obj:`~int`
            Maximum iterations limit of L-BFGS.
        device : :obj:`~str`
            Device type ('cpu' or 'cuda') represents the device on
            which operations will be allocated.
        """
        optimizer = torch.optim.LBFGS(model.parameters(), lr)
        # load dataset info to torch.tensor
        coef_spline_bases = torch.tensor(
            self.inputs_["coef_spline_bases"], dtype=torch.float64, device=device
        )
        if self.moderators:
            moderators_by_group_tensor = dict()
            for group in self.groups:
                moderators_tensor = torch.tensor(
                    self.inputs_["moderators_by_group"][group], dtype=torch.float64, device=device
                )
                moderators_by_group_tensor[group] = moderators_tensor
        else:
            moderators_by_group_tensor = None
        foci_per_voxel_tensor, foci_per_study_tensor = dict(), dict()
        for group in self.groups:
            group_foci_per_voxel_tensor = torch.tensor(
                self.inputs_["foci_per_voxel"][group], dtype=torch.float64, device=device
            )
            group_foci_per_study_tensor = torch.tensor(
                self.inputs_["foci_per_study"][group], dtype=torch.float64, device=device
            )
            foci_per_voxel_tensor[group] = group_foci_per_voxel_tensor
            foci_per_study_tensor[group] = group_foci_per_study_tensor

        if self.iter == 0:
            prev_loss = torch.tensor(float("inf"))  # initialization loss difference

        for i in range(n_iter):
            loss = self._update(
                model,
                optimizer,
                coef_spline_bases,
                moderators_by_group_tensor,
                foci_per_voxel_tensor,
                foci_per_study_tensor,
                prev_loss,
            )
            loss_diff = loss - prev_loss
            LGR.debug(f"Iter {self.iter:04d}: log-likelihood {loss:.4f}")
            if torch.abs(loss_diff) < tol:
                break
            prev_loss = loss

        return

    def _fit(self, dataset):
        """Perform coordinate-based meta-regression (CBMR) on dataset.

        (1)Estimate group-wise spatial regression coefficients and its standard error via inverse
        Fisher Information matrix;
        (2)estimate standard error of group-wise log intensity, group-wise intensity via delta
        method. For NB or clustered model, estimate regression coefficient of overdispersion.
        Similarly, estimate regression coefficient of study-level moderators (if exist), as well
        as its standard error via Fisher Information matrix. Save these outcomes in `tables`.
        Also, estimate group-wise spatial intensity (per study) and save the results in `maps`.

        Parameters
        ----------
        dataset : :obj:`~nimare.dataset.Dataset`
            Dataset to analyze.
        """
        cbmr_model = self.model(
            spatial_coef_dim=self.inputs_["coef_spline_bases"].shape[1],
            moderators_coef_dim=len(self.moderators) if self.moderators else None,
            groups=self.groups,
            penalty=self.penalty,
            device=self.device,
        )
        
        self._optimizer(cbmr_model, self.lr, self.tol, self.n_iter, self.device)

        maps, tables = dict(), dict()
        Spatial_Regression_Coef, overdispersion_param = dict(), dict()
        # regression coef of spatial effect
        for group in self.groups:
            group_spatial_coef_linear_weight = cbmr_model.spatial_coef_linears[group].weight
            group_spatial_coef_linear_weight = (
                group_spatial_coef_linear_weight.cpu().detach().numpy().flatten()
            )
            Spatial_Regression_Coef[group] = group_spatial_coef_linear_weight
            group_studywise_spatial_intensity = np.exp(
                np.matmul(self.inputs_["coef_spline_bases"], group_spatial_coef_linear_weight)
            )
            maps[
                "Group_" + group + "_Studywise_Spatial_Intensity"
            ] = group_studywise_spatial_intensity  # .reshape((1,-1))
            # overdispersion parameter
            if isinstance(cbmr_model, models.NegativeBinomial):
                group_overdispersion = cbmr_model.overdispersion_sqrt[group] ** 2
                group_overdispersion = group_overdispersion.cpu().detach().numpy()
                overdispersion_param[group] = group_overdispersion
            elif isinstance(cbmr_model, models.ClusteredNegativeBinomial):
                group_overdispersion = cbmr_model.overdispersion[group]
                group_overdispersion = group_overdispersion.cpu().detach().numpy()
                overdispersion_param[group] = group_overdispersion

        tables["Spatial_Regression_Coef"] = pd.DataFrame.from_dict(
            Spatial_Regression_Coef, orient="index"
        )
        if isinstance(cbmr_model, (models.NegativeBinomial, models.ClusteredNegativeBinomial)):
            tables["Overdispersion_Coef"] = pd.DataFrame.from_dict(
                overdispersion_param, orient="index", columns=["overdispersion"]
            )
        # study-level moderators
        if self.moderators:
            self.moderators_effect = dict()
            self._moderators_coef = cbmr_model.moderators_linear.weight
            self._moderators_coef = self._moderators_coef.cpu().detach().numpy()
            for group in self.groups:
                group_moderators = self.inputs_["moderators_by_group"][group]
                group_moderators_effect = np.exp(np.matmul(group_moderators, self._moderators_coef.T))
                self.moderators_effect[group] = group_moderators_effect
            tables["Moderators_Regression_Coef"] = pd.DataFrame(
                self._moderators_coef, columns=self.moderators
            )
        else:
            self._moderators_coef = None
        # standard error
        spatial_regression_coef_se, log_spatial_intensity_se, spatial_intensity_se = (
            dict(),
            dict(),
            dict(),
        )
        coef_spline_bases = torch.tensor(
            self.inputs_["coef_spline_bases"], dtype=torch.float64, device=self.device
        )
        for group in self.groups:
            group_foci_per_voxel = torch.tensor(
                self.inputs_["foci_per_voxel"][group], dtype=torch.float64, device=self.device
            )
            group_foci_per_study = torch.tensor(
                self.inputs_["foci_per_study"][group], dtype=torch.float64, device=self.device
            )
            group_spatial_coef = torch.tensor(cbmr_model.spatial_coef_linears[group].weight,
                                              dtype=torch.float64, device=self.device)
            if self.moderators:
                group_moderators = torch.tensor(
                    self.inputs_["moderators_by_group"][group], dtype=torch.float64, device=self.device
                )
                moderators_coef = torch.tensor(self._moderators_coef, dtype=torch.float64, device=self.device)
            else:
                group_moderators, moderators_coef = None, None
            
            ll_single_group_kwargs = {
                "moderators_coef": moderators_coef,
                "coef_spline_bases": coef_spline_bases,
                "moderators": group_moderators,
                "foci_per_voxel": group_foci_per_voxel,
                "foci_per_study": group_foci_per_study,
                "device": self.device,
            }

            if "Overdispersion_Coef" in tables.keys():
                ll_single_group_kwargs['overdispersion'] = torch.tensor(
                    tables["Overdispersion_Coef"].to_dict()["overdispersion"][group],
                    dtype=torch.float64,
                    device=self.device,
                )

            # create a negative log-likelihood function
            def nll_spatial_coef(group_spatial_coef):
                return -self.model._log_likelihood_single_group(
                    group_spatial_coef=group_spatial_coef, **ll_single_group_kwargs,
                )

            F_spatial_coef = functorch.hessian(nll_spatial_coef)(group_spatial_coef)
            # Inference on regression coefficient of spatial effect
    
            F_spatial_coef = F_spatial_coef.reshape((cbmr_model.spatial_coef_dim, cbmr_model.spatial_coef_dim))
            Cov_spatial_coef = np.linalg.inv(F_spatial_coef.detach().numpy())
            Var_spatial_coef = np.diag(Cov_spatial_coef)
            SE_spatial_coef = np.sqrt(Var_spatial_coef)
            spatial_regression_coef_se[group] = SE_spatial_coef

            Var_log_spatial_intensity = np.einsum(
                "ij,ji->i",
                self.inputs_["coef_spline_bases"],
                Cov_spatial_coef @ self.inputs_["coef_spline_bases"].T,
            )
            SE_log_spatial_intensity = np.sqrt(Var_log_spatial_intensity)
            log_spatial_intensity_se[group] = SE_log_spatial_intensity

            group_studywise_spatial_intensity = maps[
                "Group_" + group + "_Studywise_Spatial_Intensity"
            ]
            SE_spatial_intensity = group_studywise_spatial_intensity * SE_log_spatial_intensity
            spatial_intensity_se[group] = SE_spatial_intensity

        tables["Spatial_Regression_Coef_SE"] = pd.DataFrame.from_dict(
            spatial_regression_coef_se, orient="index"
        )
        tables["Log_Spatial_Intensity_SE"] = pd.DataFrame.from_dict(
            log_spatial_intensity_se, orient="index"
        )
        tables["Spatial_Intensity_SE"] = pd.DataFrame.from_dict(
            spatial_intensity_se, orient="index"
        )

        # Inference on regression coefficient of moderators
        if self.moderators:
            # modify ll_single_group_kwargs so that beta is fixed and gamma can vary
            del ll_single_group_kwargs["moderators_coef"]
            ll_single_group_kwargs["group_spatial_coef"] = group_spatial_coef

            def nll_moderators_coef(moderators_coef):
                return -self.model._log_likelihood_single_group(
                    moderators_coef=moderators_coef, **ll_single_group_kwargs,
                )

            F_moderators_coef = torch.autograd.functional.hessian(
                nll_moderators_coef,
                moderators_coef,
                create_graph=False,
                vectorize=True,
                outer_jacobian_strategy="forward-mode",
            )
            F_moderators_coef = F_moderators_coef.reshape((cbmr_model.moderators_coef_dim, cbmr_model.moderators_coef_dim))
            Cov_moderators_coef = np.linalg.inv(F_moderators_coef.detach().numpy())
            Var_moderators = np.diag(Cov_moderators_coef).reshape((1, cbmr_model.moderators_coef_dim))
            SE_moderators = np.sqrt(Var_moderators)
            tables["Moderators_Regression_SE"] = pd.DataFrame(
                SE_moderators, columns=self.moderators
            )

        return maps, tables


class CBMRInference(object):
    """Statistical inference on outcomes (intensity estimation and study-level
    moderator regressors) of CBMR.

    Parameters
    ----------
    CBMRResults : :obj:`~nimare.results.MetaResult`
        Results of optimized regression coefficients of CBMR, as well as their
        standard error in `tables`. Results of estimated spatial intensity function
        (per study) in `maps`.
    t_con_group : :obj:`~bool` or obj:`~list` or obj:`~None`, optional
        Contrast matrix for homogeneity test or group comparison on estimated spatial
        intensity function.
        For boolean inputs, no statistical inference will be conducted for spatial intensity
        if `t_con_group` is False, and spatial homogeneity test for groupwise intensity
        function will be conducted if `t_con_group` is True.
        For list inputs, generialized linear hypothesis (GLH) testing will be conducted for
        each element independently. We also allow any element of `t_con_group` in list type,
        which represents GLH is conducted for all contrasts in this element simultaneously.
        Default is homogeneity test on group-wise estimated intensity function.
    t_con_moderators : :obj:`~bool` or obj:`~list` or obj:`~None`, optional
        Contrast matrix for testing the existence of one or more study-level moderator effects.
        For boolean inputs, no statistical inference will be conducted for study-level moderators
        if `t_con_moderators` is False, and statistical inference on the effect of each study-level
        moderators will be conducted if `t_con_group` is True.
        For list inputs, generialized linear hypothesis (GLH) testing will be conducted for
        each element independently. We also allow any element of `t_con_moderators` in list type,
        which represents GLH is conducted for all contrasts in this element simultaneously.
        Default is statistical inference on the effect of each study-level moderators
    device: :obj:`string`, optional
        Device type ('cpu' or 'cuda') represents the device on which operations will be allocated.
        Default is 'cpu'.
    """

    def __init__(self, CBMRResults, t_con_group=None, t_con_moderator=None, device="cpu"):
        self.device = device
        self.CBMRResults = CBMRResults
        self.t_con_group = t_con_group
        self.t_con_moderator = t_con_moderator
        self.group_names = self.CBMRResults.tables["Spatial_Regression_Coef"].index.values.tolist()
        self.n_groups = len(self.group_names)
        if self.t_con_group is not False:
            # Conduct group-wise spatial homogeneity test by default
            self.t_con_group = (
                [np.eye(self.n_groups)]
                if not self.t_con_group
                else [np.array(con_group) for con_group in self.t_con_group]
            )
            self.t_con_group = [
                con_group.reshape((1, -1)) if len(con_group.shape) == 1 else con_group
                for con_group in self.t_con_group
            ]  # 2D contrast matrix/vector
            if np.any([con_group.shape[1] != self.n_groups for con_group in self.t_con_group]):
                wrong_con_group_idx = np.where(
                    [con_group.shape[1] != self.n_groups for con_group in self.t_con_group]
                )[0].tolist()
                raise ValueError(
                    f"""The shape of {str(wrong_con_group_idx)}th contrast vector(s) in group-wise
                    intensity contrast matrix doesn't match with groups"""
                )
            con_group_zero_row = [
                np.where(np.sum(np.abs(con_group), axis=1) == 0)[0]
                for con_group in self.t_con_group
            ]
            if np.any(
                [len(zero_row) > 0 for zero_row in con_group_zero_row]
            ):  # remove zero rows in contrast matrix
                self.t_con_group = [
                    np.delete(self.t_con_group[i], con_group_zero_row[i], axis=0)
                    for i in range(len(self.t_con_group))
                ]
                if np.any([con_group.shape[0] == 0 for con_group in self.t_con_group]):
                    raise ValueError(
                        """One or more of contrast vectors(s) in group-wise intensity
                        contrast matrix are all zeros"""
                    )
            self._Name_of_con_group()
            # standardization
            self.t_con_group = [
                con_group / np.sum(np.abs(con_group), axis=1).reshape((-1, 1))
                for con_group in self.t_con_group
            ]

        if self.t_con_moderator is not False:
            if self.CBMRResults.estimator.moderators:
                self.moderator_names = self.CBMRResults.estimator.moderators
                self.n_moderators = len(self.moderator_names)
                self.t_con_moderator = (
                    [np.eye(self.n_moderators)]
                    if not self.t_con_moderator
                    else [np.array(con_moderator) for con_moderator in self.t_con_moderator]
                )
                self.t_con_moderator = [
                    con_moderator.reshape((1, -1))
                    if len(con_moderator.shape) == 1
                    else con_moderator
                    for con_moderator in self.t_con_moderator
                ]
                # test the existence of effect of moderators
                if np.any(
                    [
                        con_moderator.shape[1] != self.n_moderators
                        for con_moderator in self.t_con_moderator
                    ]
                ):
                    wrong_con_moderator_idx = np.where(
                        [
                            con_moderator.shape[1] != self.n_moderators
                            for con_moderator in self.t_con_moderator
                        ]
                    )[0].tolist()
                    raise ValueError(
                        f"""The shape of {str(wrong_con_moderator_idx)}th contrast vector(s) in
                        moderators contrast matrix doesn't match with moderators"""
                    )
                con_moderator_zero_row = [
                    np.where(np.sum(np.abs(con_modereator), axis=1) == 0)[0]
                    for con_modereator in self.t_con_moderator
                ]
                if np.any(
                    [len(zero_row) > 0 for zero_row in con_moderator_zero_row]
                ):  # remove zero rows in contrast matrix
                    self.t_con_moderator = [
                        np.delete(self.t_con_moderator[i], con_moderator_zero_row[i], axis=0)
                        for i in range(len(self.t_con_moderator))
                    ]
                    if np.any(
                        [con_moderator.shape[0] == 0 for con_moderator in self.t_con_moderator]
                    ):
                        raise ValueError(
                            """One or more of contrast vectors(s) in modereators contrast matrix
                            are all zeros"""
                        )
                self._Name_of_con_moderator()
                self.t_con_moderator = [
                    con_moderator / np.sum(np.abs(con_moderator), axis=1).reshape((-1, 1))
                    for con_moderator in self.t_con_moderator
                ]
            else:
                self.t_con_moderator = False
        if self.device == "cuda" and not torch.cuda.is_available():
            LGR.debug("cuda not found, use device 'cpu'")
            self.device = "cpu"

    def _Name_of_con_group(self):
        """Define the name of GLH contrasts on spatial intensity estimation.

        And the names will be displayed as keys of `CBMRResults.maps` (if `t_con_group`
        exists).
        """
        self.t_con_group_name = list()
        for con_group in self.t_con_group:
            con_group_name = list()
            for num, idx in enumerate(con_group):
                if np.sum(idx) != 0:  # homogeneity test
                    nonzero_con_group_info = str()
                    nonzero_group_index = np.where(idx != 0)[0].tolist()
                    nonzero_group_name = [self.group_names[i] for i in nonzero_group_index]
                    nonzero_con = [int(idx[i]) for i in nonzero_group_index]
                    for i in range(len(nonzero_group_index)):
                        nonzero_con_group_info += (
                            str(abs(nonzero_con[i])) + "x" + str(nonzero_group_name[i])
                        )
                    con_group_name.append("homo_test_" + nonzero_con_group_info)
                else:  # group-comparison test
                    pos_group_idx, neg_group_idx = (
                        np.where(idx > 0)[0].tolist(),
                        np.where(idx < 0)[0].tolist(),
                    )
                    pos_group_name, neg_group_name = [
                        self.group_names[i] for i in pos_group_idx
                    ], [self.group_names[i] for i in neg_group_idx]
                    pos_group_con, neg_group_con = [int(idx[i]) for i in pos_group_idx], [
                        int(idx[i]) for i in neg_group_idx
                    ]
                    pos_con_group_info, neg_con_group_info = str(), str()
                    for i in range(len(pos_group_idx)):
                        pos_con_group_info += str(pos_group_con[i]) + "x" + str(pos_group_name[i])
                    for i in range(len(neg_group_idx)):
                        neg_con_group_info += (
                            str(abs(neg_group_con[i])) + "x" + str(neg_group_name[i])
                        )
                    con_group_name.append(pos_con_group_info + "VS" + neg_con_group_info)
            self.t_con_group_name.append(con_group_name)
        return

    def _Name_of_con_moderator(self):
        """Define the name of GLH contrasts on regressors of study-level moderators.

        And the names will be displayed as keys of `CBMRResults.maps` (if `t_con_moderators`
        exists).
        """
        self.t_con_moderator_name = list()
        for con_moderator in self.t_con_moderator:
            con_moderator_name = list()
            for num, idx in enumerate(con_moderator):
                if np.sum(idx) != 0:  # homogeneity test
                    nonzero_con_moderator_info = str()
                    nonzero_moderator_index = np.where(idx != 0)[0].tolist()
                    nonzero_moderator_name = [
                        self.moderator_names[i] for i in nonzero_moderator_index
                    ]
                    nonzero_con = [int(idx[i]) for i in nonzero_moderator_index]
                    for i in range(len(nonzero_moderator_index)):
                        nonzero_con_moderator_info += (
                            str(abs(nonzero_con[i])) + "x" + str(nonzero_moderator_name[i])
                        )
                    con_moderator_name.append("Effect_of_" + nonzero_con_moderator_info)
                else:  # group-comparison test
                    pos_moderator_idx, neg_moderator_idx = (
                        np.where(idx > 0)[0].tolist(),
                        np.where(idx < 0)[0].tolist(),
                    )
                    pos_moderator_name, neg_moderator_name = [
                        self.moderator_names[i] for i in pos_moderator_idx
                    ], [self.moderator_names[i] for i in neg_moderator_idx]
                    pos_moderator_con, neg_moderator_con = [
                        int(idx[i]) for i in pos_moderator_idx
                    ], [int(idx[i]) for i in neg_moderator_idx]
                    pos_con_moderator_info, neg_con_moderator_info = str(), str()
                    for i in range(len(pos_moderator_idx)):
                        pos_con_moderator_info += (
                            str(pos_moderator_con[i]) + "x" + str(pos_moderator_name[i])
                        )
                    for i in range(len(neg_moderator_idx)):
                        neg_con_moderator_info += (
                            str(abs(neg_moderator_con[i])) + "x" + str(neg_moderator_name[i])
                        )
                    con_moderator_name.append(
                        pos_con_moderator_info + "VS" + neg_con_moderator_info
                    )
            self.t_con_moderator_name.append(con_moderator_name)
        return

    def _Fisher_info_spatial_coef(self, GLH_involved_index):
        coef_spline_bases = torch.tensor(
            self.CBMRResults.estimator.inputs_["coef_spline_bases"],
            dtype=torch.float64,
            device=self.device,
        )
        GLH_involved = [self.group_names[i] for i in GLH_involved_index]
        involved_group_foci_per_voxel = [
            torch.tensor(
                self.CBMRResults.estimator.inputs_["foci_per_voxel"][group],
                dtype=torch.float64,
                device=self.device,
            )
            for group in GLH_involved
        ]
        involved_group_foci_per_study = [
            torch.tensor(
                self.CBMRResults.estimator.inputs_["foci_per_study"][group],
                dtype=torch.float64,
                device=self.device,
            )
            for group in GLH_involved
        ]
        if "Overdispersion_Coef" in self.CBMRResults.tables.keys():
            involved_overdispersion_coef = torch.tensor(
                [
                    self.CBMRResults.tables["Overdispersion_Coef"].to_numpy()[i, :]
                    for i in GLH_involved_index
                ],
                dtype=torch.float64,
                device=self.device,
            )
        involved_spatial_coef = np.stack(
            [
                self.CBMRResults.tables["Spatial_Regression_Coef"]
                .to_numpy()[i, :]
                .reshape((-1, 1))
                for i in GLH_involved_index
            ]
        )
        involved_spatial_coef = torch.tensor(
            involved_spatial_coef, dtype=torch.float64, device=self.device
        )
        n_involved_groups, spatial_coef_dim, _ = involved_spatial_coef.shape
        if self.CBMRResults.estimator.moderators:
            involved_group_moderators = [
                torch.tensor(
                    self.CBMRResults.estimator.inputs_["all_group_moderators"][group],
                    dtype=torch.float64,
                    device=self.device,
                )
                for group in GLH_involved
            ]
            involved_moderator_coef = torch.tensor(
                self.CBMRResults.tables["Moderators_Regression_Coef"].to_numpy().T,
                dtype=torch.float64,
                device=self.device,
            )
        else:
            involved_group_moderators, involved_moderator_coef = None, None
        if self.CBMRResults.estimator.model == "Poisson":
            nll = lambda spatial_coef: -GLMPoisson._log_likelihood_mult_group(
                spatial_coef,
                coef_spline_bases,
                involved_group_foci_per_voxel,
                involved_group_foci_per_study,
                involved_moderator_coef,
                involved_group_moderators,
            )
        elif self.CBMRResults.estimator.model == "NB":
            nll = lambda spatial_coef: -GLMNB._log_likelihood_mult_group(
                involved_overdispersion_coef,
                spatial_coef,
                coef_spline_bases,
                involved_group_foci_per_voxel,
                involved_group_foci_per_study,
                involved_moderator_coef,
                involved_group_moderators,
            )
        elif self.CBMRResults.estimator.model == "clustered_NB":
            nll = lambda spatial_coef: -GLMCNB._log_likelihood_mult_group(
                involved_overdispersion_coef,
                spatial_coef,
                coef_spline_bases,
                involved_group_foci_per_voxel,
                involved_group_foci_per_study,
                involved_moderator_coef,
                involved_group_moderators,
            )
        h = functorch.hessian(nll)(involved_spatial_coef)
        h = h.view(n_involved_groups * spatial_coef_dim, -1)

        return h.detach().cpu().numpy()

    def _Fisher_info_moderator_coef(self):
        coef_spline_bases = torch.tensor(
            self.CBMRResults.estimator.inputs_["coef_spline_bases"],
            dtype=torch.float64,
            device=self.device,
        )
        all_group_foci_per_voxel = [
            torch.tensor(
                self.CBMRResults.estimator.inputs_["foci_per_voxel"][group],
                dtype=torch.float64,
                device=self.device,
            )
            for group in self.group_names
        ]
        all_group_foci_per_study = [
            torch.tensor(
                self.CBMRResults.estimator.inputs_["foci_per_study"][group],
                dtype=torch.float64,
                device=self.device,
            )
            for group in self.group_names
        ]
        spatial_coef = np.stack(
            [
                self.CBMRResults.tables["Spatial_Regression_Coef"]
                .to_numpy()[i, :]
                .reshape((-1, 1))
                for i in range(self.n_groups)
            ]
        )
        spatial_coef = torch.tensor(spatial_coef, dtype=torch.float64, device=self.device)

        all_moderator_coef = torch.tensor(
            self.CBMRResults.tables["Moderators_Regression_Coef"].to_numpy().T,
            dtype=torch.float64,
            device=self.device,
        )
        moderator_coef_dim, _ = all_moderator_coef.shape
        all_group_moderators = [
            torch.tensor(
                self.CBMRResults.estimator.inputs_["all_group_moderators"][group],
                dtype=torch.float64,
                device=self.device,
            )
            for group in self.group_names
        ]

        if "Overdispersion_Coef" in self.CBMRResults.tables.keys():
            overdispersion_coef = torch.tensor(
                self.CBMRResults.tables["Overdispersion_Coef"].to_numpy(),
                dtype=torch.float64,
                device=self.device,
            )

        if self.CBMRResults.estimator.model == "Poisson":
            nll = lambda all_moderator_coef: -GLMPoisson._log_likelihood_mult_group(
                spatial_coef,
                coef_spline_bases,
                all_group_foci_per_voxel,
                all_group_foci_per_study,
                all_moderator_coef,
                all_group_moderators,
            )
        elif self.CBMRResults.estimator.model == "NB":
            nll = lambda all_moderator_coef: -GLMNB._log_likelihood_mult_group(
                overdispersion_coef,
                spatial_coef,
                coef_spline_bases,
                all_group_foci_per_voxel,
                all_group_foci_per_study,
                all_moderator_coef,
                all_group_moderators,
            )
        elif self.CBMRResults.estimator.model == "clustered_NB":
            nll = lambda all_moderator_coef: -GLMCNB._log_likelihood_mult_group(
                overdispersion_coef,
                spatial_coef,
                coef_spline_bases,
                all_group_foci_per_voxel,
                all_group_foci_per_study,
                all_moderator_coef,
                all_group_moderators,
            )
        h = functorch.hessian(nll)(all_moderator_coef)
        h = h.view(moderator_coef_dim, moderator_coef_dim)

        return h.detach().cpu().numpy()

    def _contrast(self):
        """Conduct generalized linear hypothesis (GLH) testing on CBMR estimates.

        Estimate group-wise spatial regression coefficients and its standard error via inverse
        Fisher Information matrix, estimate standard error of group-wise log intensity,
        group-wise intensity via delta method.  For NB or clustered model, estimate regression
        coefficient of overdispersion. Similarly, estimate regression coefficient of study-level
        moderators (if exist), as well as its standard error via Fisher Information matrix.
        Save these outcomes in `tables`. Also, estimate group-wise spatial intensity (per study)
        and save the results in `maps`.

        Parameters
        ----------
        dataset : :obj:`~nimare.dataset.Dataset`
            Dataset to analyze.
        """
        # Log_Spatial_Intensity_SE = self.CBMRResults.tables["Log_Spatial_Intensity_SE"]
        if self.t_con_group is not False:
            con_group_count = 0
            for con_group in self.t_con_group:
                con_group_involved_index = np.where(np.any(con_group != 0, axis=0))[0].tolist()
                con_group_involved = [self.group_names[i] for i in con_group_involved_index]
                n_con_group_involved = len(con_group_involved)
                simp_con_group = con_group[
                    :, ~np.all(con_group == 0, axis=0)
                ]  # contrast matrix of involved groups only
                if np.all(np.count_nonzero(con_group, axis=1) == 1):  # GLH: homogeneity test
                    involved_log_intensity_per_voxel = list()
                    for group in con_group_involved:
                        group_foci_per_voxel = self.CBMRResults.estimator.inputs_[
                            "foci_per_voxel"
                        ][group]
                        group_foci_per_study = self.CBMRResults.estimator.inputs_[
                            "foci_per_study"
                        ][group]
                        n_voxels, n_study = (
                            group_foci_per_voxel.shape[0],
                            group_foci_per_study.shape[0],
                        )
                        group_null_log_spatial_intensity = np.log(
                            np.sum(group_foci_per_voxel) / (n_voxels * n_study)
                        )
                        group_log_intensity_per_voxel = np.log(
                            self.CBMRResults.maps[
                                "Group_" + group + "_Studywise_Spatial_Intensity"
                            ]
                        )
                        group_log_intensity_per_voxel = (
                            group_log_intensity_per_voxel - group_null_log_spatial_intensity
                        )
                        involved_log_intensity_per_voxel.append(group_log_intensity_per_voxel)
                    involved_log_intensity_per_voxel = np.stack(
                        involved_log_intensity_per_voxel, axis=0
                    )
                else:  # GLH: group comparison
                    involved_log_intensity_per_voxel = list()
                    for group in con_group_involved:
                        group_log_intensity_per_voxel = np.log(
                            self.CBMRResults.maps[
                                "Group_" + group + "_Studywise_Spatial_Intensity"
                            ]
                        )
                        involved_log_intensity_per_voxel.append(group_log_intensity_per_voxel)
                    involved_log_intensity_per_voxel = np.stack(
                        involved_log_intensity_per_voxel, axis=0
                    )
                Contrast_log_intensity = np.matmul(
                    simp_con_group, involved_log_intensity_per_voxel
                )
                m, n_brain_voxel = Contrast_log_intensity.shape
                # Correlation of involved group-wise spatial coef
                F_spatial_coef = self._Fisher_info_spatial_coef(con_group_involved_index)
                Cov_spatial_coef = np.linalg.inv(F_spatial_coef)
                spatial_coef_dim = (
                    self.CBMRResults.tables["Spatial_Regression_Coef"].to_numpy().shape[1]
                )
                Cov_log_intensity = np.empty(shape=(0, n_brain_voxel))
                for k in range(n_con_group_involved):
                    for s in range(n_con_group_involved):
                        Cov_beta_ks = Cov_spatial_coef[
                            k * spatial_coef_dim : (k + 1) * spatial_coef_dim,
                            s * spatial_coef_dim : (s + 1) * spatial_coef_dim,
                        ]
                        X = self.CBMRResults.estimator.inputs_["coef_spline_bases"]
                        Cov_group_log_intensity = (X.dot(Cov_beta_ks) * X).sum(axis=1).reshape((1, -1))
                        Cov_log_intensity = np.concatenate(
                            (Cov_log_intensity, Cov_group_log_intensity), axis=0
                        )  # (m^2, n_voxels)
                # GLH on log_intensity (eta)
                chi_sq_spatial = np.empty(shape=(0,))
                for j in range(n_brain_voxel):
                    Contrast_log_intensity_j = Contrast_log_intensity[:, j].reshape(m, 1)
                    V_j = Cov_log_intensity[:, j].reshape(
                        (n_con_group_involved, n_con_group_involved)
                    )
                    CV_jC = simp_con_group @ V_j @ simp_con_group.T
                    CV_jC_inv = np.linalg.inv(CV_jC)
                    chi_sq_spatial_j = (
                        Contrast_log_intensity_j.T @ CV_jC_inv @ Contrast_log_intensity_j
                    )
                    chi_sq_spatial = np.concatenate(
                        (
                            chi_sq_spatial,
                            chi_sq_spatial_j.reshape(
                                1,
                            ),
                        ),
                        axis=0,
                    )
                p_vals_spatial = 1 - scipy.stats.chi2.cdf(chi_sq_spatial, df=m)

                con_group_name = self.t_con_group_name[con_group_count]
                if len(con_group_name) == 1:
                    self.CBMRResults.maps[con_group_name[0] + "_chi_sq"] = chi_sq_spatial
                    self.CBMRResults.maps[con_group_name[0] + "_p"] = p_vals_spatial
                else:
                    self.CBMRResults.maps[
                        "spatial_coef_GLH_" + str(con_group_count) + "_chi_sq"
                    ] = chi_sq_spatial
                    self.CBMRResults.maps[
                        "spatial_coef_GLH_" + str(con_group_count) + "_p"
                    ] = p_vals_spatial
                    self.CBMRResults.metadata[
                        "spatial_coef_GLH_" + str(con_group_count)
                    ] = con_group_name
                con_group_count += 1

        if self.t_con_moderator is not False:
            con_moderator_count = 0
            for con_moderator in self.t_con_moderator:
                m_con_moderator, _ = con_moderator.shape
                moderator_coef = self.CBMRResults.tables["Moderators_Regression_Coef"].to_numpy().T
                Contrast_moderator_coef = np.matmul(con_moderator, moderator_coef)
                F_moderator_coef = self._Fisher_info_moderator_coef()
                Cov_moderator_coef = np.linalg.inv(F_moderator_coef)
                chi_sq_moderator = (
                    Contrast_moderator_coef.T
                    @ np.linalg.inv(con_moderator @ Cov_moderator_coef @ con_moderator.T)
                    @ Contrast_moderator_coef
                )
                chi_sq_moderator = chi_sq_moderator.item()
                p_vals_moderator = 1 - scipy.stats.chi2.cdf(chi_sq_moderator, df=m_con_moderator)

                con_moderator_name = self.t_con_moderator_name[con_moderator_count]
                if len(con_moderator_name) == 1:
                    self.CBMRResults.tables[con_moderator_name[0] + "_chi_sq"] = chi_sq_moderator
                    self.CBMRResults.tables[con_moderator_name[0] + "_p"] = p_vals_moderator
                else:
                    self.CBMRResults.tables[
                        "moderator_coef_GLH_" + str(con_moderator_count) + "_chi_sq"
                    ] = chi_sq_moderator
                    self.CBMRResults.tables[
                        "moderator_coef_GLH_" + str(con_moderator_count) + "_p"
                    ] = p_vals_moderator
                    self.CBMRResults.metadata[
                        "moderator_coef_GLH_" + str(con_moderator_count)
                    ] = con_moderator_name
                con_moderator_count += 1

        return
