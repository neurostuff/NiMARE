from nimare.base import Estimator
from nimare.utils import get_masker, B_spline_bases, dummy_encoding_moderators
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
import re


LGR = logging.getLogger(__name__)


class CBMREstimator(Estimator):
    """Coordinate-based meta-regression with a spatial model.

    Parameters
    ----------
    group_categories : :obj:`~str` or obj:`~list` or obj:`~None`, optional
        CBMR allows dataset to be categorized into mutiple groups, according to group categories.
        Default is one-group CBMR.
    moderators : :obj:`~str` or obj:`~list` or obj:`~None`, optional
        CBMR can accommodate study-level moderators (e.g. sample size, year of publication).
        Default is CBMR without study-level moderators.
    model : : :obj:`~nimare.meta.models.GeneralLinearModel`, optional
        Stochastic models in CBMR. The available options are

        ======================= ==================================================================
        Poisson (default)       This is the most efficient and widely used method, but slightly
                                less accurate, because Poisson model is an approximation for
                                low-rate Binomial data, but cannot account over-dispersion in
                                foci counts and may underestimate the standard error.

        NegativeBinomial        This method might be slower and less stable, but slightly more
                                accurate. Negative Binomial (NB) model asserts foci counts follow
                                a NB distribution, and allows for anticipated excess variance
                                relative to Poisson (there's an group-wise overdispersion parameter
                                shared by all studies and all voxels to index excess variance).

        ClusteredNegativeBinomial This method is also an efficient but less accurate approach.
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
    lr_decay: :obj:`float`, optional
        Multiplicative factor of learning rate decay.
        Default is 0.999.
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
        model=models.PoissonEstimator,
        penalty=False,
        n_iter=1000,
        lr=1e-2,
        lr_decay=0.999,
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
        self.model = model(penalty=penalty, lr=lr, lr_decay=lr_decay, n_iter=n_iter, tol=tol, device=device)
        self.penalty = penalty
        self.n_iter = n_iter
        self.lr = lr
        self.lr_decay = lr_decay
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
            (2) categorize studies into multiple groups according to group categories in annotations,
            (3) summarize group-wise study id, moderators (if exist), foci per voxel, foci per study,
            (4) extract sample size metadata and use it as one of study-level moderators.

        Attributes
        ----------
        inputs_ : :obj:`dict`
            Specifically, (1) a “mask_img” key will be added (Niftiimage of brain mask),
            (2) an 'id' key will be added (id of all studies in the dataset),
            (3) a 'coef_spline_bases' key will be added (spatial matrix of coefficient of cubic
            B-spline bases in x,y,z dimension),
            (4) an 'studies_by_group' key will be added (study id categorized by groups),
            (5) an 'moderators_by_group' key will be added (study-level moderators categorized 
            by groups) if study-level moderators are considered,
            (6) an 'foci_per_voxel' key will be added (voxelwise sum of foci count across
            studies, categorized by groups),
            (7) an 'foci_per_study' key will be added (study-wise sum of foci count across
            space, categorized by groups).
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
                self.groups = list(self.inputs_["studies_by_group"].keys())
                # collect studywise moderators if specficed
                if self.moderators:
                    valid_dset_annotations, self.moderators = dummy_encoding_moderators(valid_dset_annotations, self.moderators)
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

    def _fit(self, dataset):
        """Perform coordinate-based meta-regression (CBMR) on dataset.

        (1) Estimate group-wise spatial regression coefficients and its standard error via 
        inverse of Fisher Information matrix; Similarly, estimate regression coefficient of 
        study-level moderators (if exist), as well as its standard error via inverse of
        Fisher Information matrix;
        (2) Estimate standard error of group-wise log intensity, group-wise intensity via delta
        method;
        (3) For NegativeBinomial or ClusteredNegativeBinomial model, estimate regression 
        coefficient of overdispersion.s

        Parameters
        ----------
        dataset : :obj:`~nimare.dataset.Dataset`
            Dataset to analyze.
        """
        init_weight_kwargs = {
            'groups': self.groups,
            'spatial_coef_dim': self.inputs_["coef_spline_bases"].shape[1],
            'moderators_coef_dim': len(self.moderators) if self.moderators else None,
        }
        self.model.init_weights(**init_weight_kwargs)

        moderators_by_group = self.inputs_["moderators_by_group"] if self.moderators else None
        self.model.fit(self.inputs_["coef_spline_bases"], moderators_by_group, self.inputs_["foci_per_voxel"], self.inputs_["foci_per_study"])

        maps, tables = self.model.summary()

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
    
    def __init__(self, CBMRResults, device="cpu"): 
        self.device = device
        self.CBMRResults = CBMRResults
        self.groups = self.CBMRResults.estimator.groups
        self.n_groups = len(self.groups)
        
        # visialize group/moderator names and their indices in contrast array
        self.group_reference_dict, self.moderator_reference_dict = dict(), dict()
        LGR.info("Group Reference in contrast array")
        for i in range(self.n_groups):
            self.group_reference_dict[self.groups[i]] = i
            LGR.info(f"{self.groups[i]} = index_{i}")
        if self.CBMRResults.estimator.moderators:
            n_moderators = len(self.CBMRResults.estimator.moderators)
            LGR.info("Moderator Reference in contrast array")
            for j in range(n_moderators):
                self.moderator_reference_dict[self.CBMRResults.estimator.moderators[j]] = j
                LGR.info(f"{self.CBMRResults.estimator.moderators[j]} = index_{j}")
            
        # device check
        if self.device == "cuda" and not torch.cuda.is_available():
            LGR.debug("cuda not found, use device 'cpu'")
            self.device = "cpu"

    def create_contrast(self, contrast_name, type="group"):
        """Create contrast matrix for generalized hypothesis testing (GLH).

        (1) if `type` is "group", create contrast matrix for GLH on spatial intensity;
        if `contrast_name` begins with 'homo_test_', followed by a valid group name, 
        create a contrast matrix for one-group homogeneity test on spatial intensity;
        if `contrast_name` comes in the form of "group1VSgroup2", with valid group names 
        "group1" and "group2", create a contrast matrix for group comparison on estimated 
        group spatial intensity;
        (2) if `type` is "moderator", create contrast matrix for GLH on study-level moderators;
        if `contrast_name` begins with 'moderator_', followed by a valid moderator name,
        we create a contrast matrix for testing if the effect of this moderator exists;
        if `contrast_name` comes in the form of "moderator1VSmoderator2", with valid moderator names
        "modeator1" and "moderator2", we create a contrast matrix for testing if the effect of
        these two moderators are different.

        Parameters
        ----------
        contrast_name : :obj:`~string`
            Name of contrast in GLH.
        """
        if isinstance(contrast_name, str):
            contrast_name = [contrast_name]
        contrast_matrix = list()
        if type == "group": # contrast matrix for spatial intensity
            for contrast in contrast_name:
                contrast_vector = np.zeros(self.n_groups)
                if contrast.startswith("homo_test_"): # homogeneity test
                    contrast_groups = contrast.split("homo_test_",1)[1]
                    if contrast_groups not in self.groups:
                        raise ValueError(f"{contrast_groups} is not a valid group name.")
                    contrast_vector[self.group_reference_dict[contrast_groups]] = 1 
                elif "VS" in contrast: # group comparison
                    contrast_groups = contrast.split("VS")
                    if not set(contrast_groups).issubset(set(self.groups)):
                        not_valid_groups = set(contrast_groups).difference(set(self.groups))
                        raise ValueError(f"{not_valid_groups} is not a valid group name.")
                    contrast_vector[self.group_reference_dict[contrast_groups[0]]] = 1
                    contrast_vector[self.group_reference_dict[contrast_groups[1]]] = -1
                contrast_matrix.append(contrast_vector)
        
        elif type == "moderator": # contrast matrix for moderator effect
            n_moderators = len(self.CBMRResults.estimator.moderators)
            for contrast in contrast_name:
                contrast_vector = np.zeros(n_moderators)
                if contrast.startswith("moderator_"): # moderator effect
                    contrast_moderators = contrast.split("moderator_",1)[1]
                    if contrast_moderators not in self.CBMRResults.estimator.moderators:
                        raise ValueError(f"{contrast_moderators} is not a valid moderator name.")
                    contrast_vector[self.moderator_reference_dict[contrast_moderators]] = 1
                elif "VS" in contrast:
                    contrast_moderators = contrast.split("VS")
                    if not set(contrast_moderators).issubset(set(self.CBMRResults.estimator.moderators)):
                        not_valid_moderators = set(contrast_moderators).difference(set(self.CBMRResults.estimator.moderators))
                        raise ValueError(f"{not_valid_moderators} is not a valid moderator name.")
                    contrast_vector[self.moderator_reference_dict[contrast_moderators[0]]] = 1
                    contrast_vector[self.moderator_reference_dict[contrast_moderators[1]]] = -1
                else:
                    raise ValueError(f"{contrast} is not a valid contrast type.")
                contrast_matrix.append(contrast_vector)
                
        return contrast_matrix
    
    def compute_contrast(self, t_con_group=None, t_con_moderator=None): 
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
        t_con_group : :obj:`~list`, optional
            Contrast matrix for GLH on group-wise spatial intensity estimation.
            Default is None (group-wise homogeneity test for all groups).
        t_con_moderator : :obj:`~list`, optional
            Contrast matrix for GLH on moderator effects.
            Default is None (tests if moderator effects exist for all moderators).
        """
        
        self.t_con_group = t_con_group
        self.t_con_moderator = t_con_moderator
        
        if self.t_con_group is not False:
            # preprocess and standardize group contrast
            self._preprocess_t_con_group()
            # GLH test for group contrast
            self._GLH_con_group()
        if self.t_con_moderator is not False:
            # preprocess and standardize moderator contrast
            self._preprocess_t_con_moderator()
            # GLH test for moderator contrast
            self._GLH_con_moderator()

    def _preprocess_t_con_group(self):
        # Conduct group-wise spatial homogeneity test by default
        self.t_con_group = (
            [np.eye(self.n_groups)]
            if not self.t_con_group
            else [np.array(con_group) for con_group in self.t_con_group]
        )
        # make sure contrast matrix/vector is 2D
        self.t_con_group = [
            con_group.reshape((1, -1)) if len(con_group.shape) == 1 else con_group
            for con_group in self.t_con_group
        ]  
        # raise error if dimension of contrast matrix/vector doesn't match with number of groups
        if np.any([con_group.shape[1] != self.n_groups for con_group in self.t_con_group]):
            wrong_con_group_idx = np.where(
                [con_group.shape[1] != self.n_groups for con_group in self.t_con_group]
            )[0].tolist()
            raise ValueError(
                f"""The shape of {str(wrong_con_group_idx)}th contrast vector(s) in group-wise
                intensity contrast matrix doesn't match with groups"""
            )
        # remove zero rows in contrast matrix (if exist)
        con_group_zero_row = [
            np.where(np.sum(np.abs(con_group), axis=1) == 0)[0]
            for con_group in self.t_con_group
        ]
        if np.any(
            [len(zero_row) > 0 for zero_row in con_group_zero_row]
        ):  
            self.t_con_group = [
                np.delete(self.t_con_group[i], con_group_zero_row[i], axis=0)
                for i in range(len(self.t_con_group))
            ]
            if np.any([con_group.shape[0] == 0 for con_group in self.t_con_group]):
                raise ValueError(
                    """One or more of contrast vectors(s) in group-wise intensity
                    contrast matrix are all zeros"""
                )
        # name of GLH contrasts and save to `tables` later
        self._Name_of_con_group()
        # standardization (row sum 1)
        self.t_con_group = [
            con_group / np.sum(np.abs(con_group), axis=1).reshape((-1, 1))
            for con_group in self.t_con_group
        ]
        # remove duplicate rows in contrast matrix (after standardization)
        uniq_con_group_idx = np.unique(self.t_con_group, axis=0, return_index=True)[1].tolist()
        self.t_con_group = [self.t_con_group[i] for i in uniq_con_group_idx[::-1]]
        
    def _preprocess_t_con_moderator(self):
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
        # remove duplicate rows in contrast matrix (after standardization)
        uniq_con_moderator_idx = np.unique(self.t_con_moderator, axis=0, return_index=True)[1].tolist()
        self.t_con_moderator = [self.t_con_moderator[i] for i in uniq_con_moderator_idx[::-1]]
        return 
    
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
                    nonzero_group_name = [self.groups[i] for i in nonzero_group_index]
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
                        self.groups[i] for i in pos_group_idx
                    ], [self.groups[i] for i in neg_group_idx]
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
                    con_moderator_name.append("ModeratorEffect_of_" + nonzero_con_moderator_info)
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
    
    def _GLH_con_group(self):
        con_group_count = 0
        for con_group in self.t_con_group:
            con_group_involved_index = np.where(np.any(con_group != 0, axis=0))[0].tolist()
            con_group_involved = [self.groups[i] for i in con_group_involved_index]
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
            moderators_by_group = self.CBMRResults.estimator.inputs_["moderators_by_group"] if self.CBMRResults.estimator.moderators else None
            F_spatial_coef = self.CBMRResults.estimator.model.FisherInfo_MultipleGroup_spatial(con_group_involved, self.CBMRResults.estimator.inputs_["coef_spline_bases"], 
                            moderators_by_group, self.CBMRResults.estimator.inputs_["foci_per_voxel"], self.CBMRResults.estimator.inputs_["foci_per_study"])            
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

    def _GLH_con_moderator(self):
        con_moderator_count = 0
        for con_moderator in self.t_con_moderator:
            m_con_moderator, _ = con_moderator.shape
            moderator_coef = self.CBMRResults.tables["Moderators_Regression_Coef"].to_numpy().T
            Contrast_moderator_coef = np.matmul(con_moderator, moderator_coef)
            
            moderators_by_group = self.CBMRResults.estimator.inputs_["moderators_by_group"] if self.CBMRResults.estimator.moderators else None
            F_moderator_coef = self.CBMRResults.estimator.model.FisherInfo_MultipleGroup_moderator(self.CBMRResults.estimator.inputs_["coef_spline_bases"], 
                            moderators_by_group, self.CBMRResults.estimator.inputs_["foci_per_voxel"], self.CBMRResults.estimator.inputs_["foci_per_study"])            
            
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