import string
from attr import has
from numpy import spacing
from nimare.base import Estimator
from nimare.utils import get_template, get_masker, B_spline_bases
import nibabel as nib
import numpy as np
import pandas as pd
import scipy
from nimare.utils import mm2vox, vox2idx, intensity2voxel
from nimare.diagnostics import FocusFilter
import torch
import logging

LGR = logging.getLogger(__name__)
class CBMREstimator(Estimator):
    _required_inputs = {"coordinates": ("coordinates", None)}

    def __init__(self, group_names=None, moderators=None, moderators_center=True, moderators_scale=True, mask=None, 
                spline_spacing=5, model='Poisson', penalty=False, n_iter=1000, lr=1e-2, tol=1e-2, device='cpu', **kwargs):
        super().__init__(**kwargs)
        if mask is not None:
            mask = get_masker(mask)
        self.masker = mask

        self.group_names = group_names
        self.moderators = moderators 
        self.moderators_center = moderators_center # either boolean or a list of strings
        self.moderators_scale = moderators_scale

        self.spline_spacing = spline_spacing
        self.model = model
        self.penalty = penalty
        self.n_iter = n_iter
        self.lr = lr
        self.tol = tol
        self.device = device

        # Initialize optimisation parameters
        self.iter = 0

    def _preprocess_input(self, dataset):
        masker = self.masker or dataset.masker

        mask_img = masker.mask_img or masker.labels_img
        if isinstance(mask_img, str):
            mask_img = nib.load(mask_img)
        
        ma_values = self._collect_inputs(dataset, drop_invalid=True)
        self.inputs_['mask_img'] = mask_img

        for name, (type_, _) in self._required_inputs.items():
            if type_ == "coordinates":
                # remove dataset coordinates outside of mask
                focus_filter = FocusFilter(mask=masker)
                dataset = focus_filter.transform(dataset)
                # remove study_id without any coordinates
                study_id_annotations = dataset.annotations.set_index('study_id').index
                study_id_coordinates = dataset.coordinates.set_index('study_id').index
                valid_study_bool = study_id_annotations.isin(study_id_coordinates)
                dataset_annotations = dataset.annotations[valid_study_bool]
                all_group_study_id = dict()
                if isinstance(self.group_names, type(None)):
                    all_group_study_id[self.group_names] = dataset_annotations['study_id'].unique().tolist()
                elif isinstance(self.group_names, str):
                    if self.group_names not in dataset_annotations.columns: 
                        raise ValueError("group_names: {} does not exist in the dataset".format(self.group_names))
                    else:
                        uniq_groups = list(dataset_annotations[self.group_names].unique())
                        for group in uniq_groups:
                            group_study_id_bool = dataset_annotations[self.group_names] == group
                            group_study_id = dataset_annotations.loc[group_study_id_bool]['study_id']
                            all_group_study_id[group] = group_study_id.unique().tolist()
                elif isinstance(self.group_names, list):
                    not_exist_group_names = [group for group in self.group_names if group not in dataset_annotations.columns]
                    if len(not_exist_group_names) > 0:
                        raise ValueError("group_names: {} does not exist in the dataset".format(not_exist_group_names))
                    uniq_group_splits = dataset_annotations[self.group_names].drop_duplicates().values.tolist()
                    for group in uniq_group_splits:
                        group_study_id_bool = (dataset_annotations[self.group_names] == group).all(axis=1)
                        group_study_id = dataset_annotations.loc[group_study_id_bool]['study_id']
                        all_group_study_id['_'.join(group)] = group_study_id.unique().tolist()
                self.inputs_['all_group_study_id'] = all_group_study_id
                # collect studywise moderators if specficed
                if hasattr(self, "moderators"):
                    all_group_moderators = dict()
                    for group in all_group_study_id.keys():
                        df_group = dataset_annotations.loc[dataset_annotations['study_id'].isin(all_group_study_id[group])] 
                        group_moderators = np.stack([df_group[moderator_name] for moderator_name in self.moderators], axis=1)
                        group_moderators = group_moderators.astype(np.float64)
                        group_moderators = self._standardize_moderators(group_moderators)
                        all_group_moderators[group] = group_moderators
                    self.inputs_["all_group_moderators"] = all_group_moderators
                # Calculate IJK matrix indices for target mask
                # Mask space is assumed to be the same as the Dataset's space
                # These indices are used directly by any KernelTransformer
                all_foci_per_voxel, all_foci_per_study = dict(), dict()
                for group in all_group_study_id.keys():
                    group_study_id = all_group_study_id[group]
                    group_coordinates = dataset.coordinates.loc[dataset.coordinates['study_id'].isin(group_study_id)] 
                    # group-wise foci coordinates
                    group_xyz = group_coordinates[['x', 'y', 'z']].values
                    group_ijk = mm2vox(group_xyz, mask_img.affine)
                    group_foci_idx = vox2idx(group_ijk, mask_img._dataobj)
                    # number of foci per voxel/study
                    n_group_study = len(group_study_id)
                    masker_voxels = np.sum(mask_img._dataobj).astype(int)
                    group_foci_per_voxel = np.zeros((masker_voxels, 1))
                    group_foci_per_voxel[group_foci_idx, :] += 1
                    group_foci_per_study = np.array([(group_coordinates['study_id']==i).sum() for i in group_study_id])
                    group_foci_per_study = group_foci_per_study.reshape((n_group_study, 1))

                    all_foci_per_voxel[group] = group_foci_per_voxel
                    all_foci_per_study[group] = group_foci_per_study
                
                self.inputs_['all_foci_per_voxel'] = all_foci_per_voxel
                self.inputs_['all_foci_per_study'] = all_foci_per_study

    def _standardize_moderators(self, moderators_array):
        # standardize mean
        if isinstance(self.moderators_center, bool):
            if self.moderators_center: 
                moderators_array -= np.mean(moderators_array, axis=0)
        elif isinstance(self.moderators_center, str):
            index_moderators_center = self.moderators.index(self.moderators_center)
            moderators_array[:,index_moderators_center] -= np.mean(moderators_array[:, index_moderators_center], axis=0)
        elif isinstance(self.moderators_center, list):
            index_moderators_center = [self.moderators.index(moderator_name) for moderator_name in self.moderators_center]
            for i in index_moderators_center:
                moderators_array[:,i] -= np.mean(moderators_array[:, i], axis=0)
        
        # standardize var
        if isinstance(self.moderators_scale, bool):
            if self.moderators_scale: 
                moderators_array /= np.std(moderators_array, axis=0)
        elif isinstance(self.moderators_scale, str):
            index_moderators_scale = self.moderators.index(self.moderators_scale)
            moderators_array[:,index_moderators_scale] /= np.std(moderators_array[:, index_moderators_scale], axis=0)
        elif isinstance(self.moderators_scale, list):
            index_moderators_scale = [self.moderators.index(moderator_name) for moderator_name in self.moderators_scale]
            for i in index_moderators_scale:
                moderators_array[:,i] /= np.std(moderators_array[:, i], axis=0)

        return moderators_array

    def _model_structure(self, model, penalty, device):
        beta_dim = self.inputs_['Coef_spline_bases'].shape[1] # regression coef of spatial effect
        if hasattr(self, "moderators"):
            gamma_dim = list(self.inputs_["all_group_moderators"].values())[0].shape[1]
            study_level_moderators = True
        else:
            gamma_dim = None
            study_level_moderators = False
        self.groups = list(self.inputs_['all_group_study_id'].keys())
        if model == 'Poisson':
            cbmr_model = GLMPoisson(beta_dim=beta_dim, gamma_dim=gamma_dim, groups=self.groups, study_level_moderators=study_level_moderators, penalty=penalty)
        if 'cuda' in device:
            cbmr_model = cbmr_model.cuda()
        
        return cbmr_model

    def _update(self, model, optimizer, Coef_spline_bases, all_moderators, all_foci_per_voxel, all_foci_per_study, prev_loss, gamma=0.999):
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=gamma) # learning rate decay
        scheduler.step()
        
        self.iter += 1
        scheduler.step()
        def closure():
            optimizer.zero_grad()
            loss = model(Coef_spline_bases, all_moderators, all_foci_per_voxel, all_foci_per_study)
            loss.backward()
            return loss
        loss = optimizer.step(closure)
                
        return loss

    def _optimizer(self, model, lr, tol, n_iter, device): 
        optimizer = torch.optim.LBFGS(model.parameters(), lr)
        # load dataset info to torch.tensor
        Coef_spline_bases = torch.tensor(self.inputs_['Coef_spline_bases'], dtype=torch.float64, device=device)
        if hasattr(self, "moderators"):
            all_group_moderators_tensor = dict()
            for group in self.inputs_['all_group_study_id'].keys():
                group_moderators_tensor = torch.tensor(self.inputs_['all_group_moderators'][group], dtype=torch.float64, device=device)
                all_group_moderators_tensor[group] = group_moderators_tensor
        else:
            all_group_moderators_tensor = None
        all_foci_per_voxel_tensor, all_foci_per_study_tensor = dict(), dict()
        for group in self.inputs_['all_group_study_id'].keys():
            group_foci_per_voxel = torch.tensor(self.inputs_['all_foci_per_voxel'][group], dtype=torch.float64, device=device)
            group_foci_per_study = torch.tensor(self.inputs_['all_foci_per_study'][group], dtype=torch.float64, device=device)
            all_foci_per_voxel_tensor[group] = group_foci_per_voxel
            all_foci_per_study_tensor[group] = group_foci_per_study

        if self.iter == 0:
            prev_loss = torch.tensor(float('inf')) # initialization loss difference
        for i in range(n_iter):
            loss = self._update(model, optimizer, Coef_spline_bases, all_group_moderators_tensor, all_foci_per_voxel_tensor, all_foci_per_study_tensor, prev_loss)
            loss_diff = loss - prev_loss
            LGR.debug(f"Iter {self.iter:04d}: log-likelihood {loss:.4f}")
            if torch.abs(loss_diff) < tol:
                break
            prev_loss = loss
        
        return

    def _fit(self, dataset):
        masker_voxels = self.inputs_['mask_img']._dataobj
        Coef_spline_bases = B_spline_bases(masker_voxels=masker_voxels, spacing=self.spline_spacing)
        P = Coef_spline_bases.shape[1]
        self.inputs_['Coef_spline_bases'] = Coef_spline_bases

        cbmr_model = self._model_structure(self.model, self.penalty, self.device)
        optimisation = self._optimizer(cbmr_model, self.lr, self.tol, self.n_iter, self.device)

        spatial_regression_coef, spatial_intensity_values = dict(), dict()
        # beta: regression coef of spatial effect
        for group in self.inputs_['all_group_study_id'].keys():
            group_beta_linear_weight = cbmr_model.all_beta_linears[group].weight
            group_beta_linear_weight = group_beta_linear_weight.cpu().detach().numpy().reshape((P,))
            spatial_regression_coef[group] = group_beta_linear_weight

            studywise_spatial_intensity = np.exp(np.matmul(Coef_spline_bases, group_beta_linear_weight))
            # studywise_spatial_intensity = intensity2voxel(studywise_spatial_intensity, self.inputs_['mask_img']._dataobj)
            spatial_intensity_values[group] = studywise_spatial_intensity
        spatial_regression_coef = pd.DataFrame.from_dict(spatial_regression_coef, orient='columns')
        # study-level moderators
        moderators_effect_values = dict()
        if hasattr(self, "moderators"):
            self._gamma = cbmr_model.gamma_linear.weight
            self._gamma = self._gamma.cpu().detach().numpy().flatten()
            # moderators_regression_coef['all_groups'] = self._gamma
            for group in self.inputs_['all_group_study_id'].keys():
                group_moderators = self.inputs_["all_group_moderators"][group]
                moderators_effect = np.exp(np.matmul(group_moderators, self._gamma))
                moderators_effect_values[group] = moderators_effect
            moderators_regression_coef = pd.DataFrame(self._gamma)
        
        maps = {
            "group-specific_StudywiseIntensity": spatial_intensity_values,
            'group-specific_moderators_effect': moderators_effect_values,
        }

        tables = {
            'spatial_regression_coef': spatial_regression_coef,
            'moderators_regression_coef': moderators_regression_coef,
        }


        return maps, tables

    

class GLMPoisson(torch.nn.Module):
    def __init__(self, beta_dim=None, gamma_dim=None, groups=None, study_level_moderators=False, penalty='No'):
        super().__init__()
        self.groups = groups
        self.study_level_moderators = study_level_moderators
        # initialization for beta
        all_beta_linears = dict()
        for group in groups:
            beta_linear_group = torch.nn.Linear(beta_dim, 1, bias=False).double()
            torch.nn.init.uniform_(beta_linear_group.weight, a=-0.01, b=0.01)
            all_beta_linears[group] = beta_linear_group
        self.all_beta_linears = torch.nn.ModuleDict(all_beta_linears)
        # gamma 
        if self.study_level_moderators:
            self.gamma_linear = torch.nn.Linear(gamma_dim, 1, bias=False).double()
            torch.nn.init.uniform_(self.gamma_linear.weight, a=-0.01, b=0.01)
    
    def forward(self, Coef_spline_bases, all_moderators, all_foci_per_voxel, all_foci_per_study):
        if isinstance(all_moderators, dict):
            all_log_mu_moderators = dict()
            for group in all_moderators.keys():
                group_moderators = all_moderators[group]
                # mu^Z = exp(Z * gamma)
                log_mu_moderators = self.gamma_linear(group_moderators)
                all_log_mu_moderators[group] = log_mu_moderators
        log_l = 0
        # spatial effect: mu^X = exp(X * beta)
        for group in all_foci_per_voxel.keys(): 
            log_mu_spatial = self.all_beta_linears[group](Coef_spline_bases)
            mu_spatial = torch.exp(log_mu_spatial)
            log_mu_moderators = all_log_mu_moderators[group]
            mu_moderators = torch.exp(log_mu_moderators)
            group_foci_per_voxel = all_foci_per_voxel[group]
            group_foci_per_study = all_foci_per_study[group]
            # Under the assumption that Y_ij is either 0 or 1
            # l = [Y_g]^T * log(mu^X) + [Y^t]^T * log(mu^Z) - [1^T mu_g^X]*[1^T mu_g^Z]
            group_log_l = torch.sum(torch.mul(group_foci_per_voxel, log_mu_spatial)) + torch.sum(torch.mul(group_foci_per_study, log_mu_moderators)) - torch.sum(mu_spatial) * torch.sum(mu_moderators)
            log_l += group_log_l
        
        return -log_l
