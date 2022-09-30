from importlib.util import set_loader
import string
from attr import has
from numpy import spacing
from nimare.base import Estimator
from nimare.utils import get_template, get_masker, B_spline_bases
import nibabel as nib
import numpy as np
import pandas as pd
import scipy
from nimare.utils import mm2vox
from nimare.diagnostics import FocusFilter
from nimare.transforms import z_to_p
from nimare import transforms
import torch
import logging
import copy
from functorch import hessian

LGR = logging.getLogger(__name__)
class CBMREstimator(Estimator):
    _required_inputs = {"coordinates": ("coordinates", None)}

    def __init__(self, group_names=None, moderators=None, mask=None, spline_spacing=5, model='Poisson', penalty=False, 
                n_iter=1000, lr=1e-2, tol=1e-2, device='cpu', **kwargs):
        super().__init__(**kwargs)
        if mask is not None:
            mask = get_masker(mask)
        self.masker = mask

        self.group_names = group_names
        self.moderators = moderators 

        self.spline_spacing = spline_spacing
        self.model = model
        self.penalty = penalty
        self.n_iter = n_iter
        self.lr = lr
        self.tol = tol
        self.device = device
        if self.device == 'cuda' and not torch.cuda.is_available(): 
            LGR.debug(f"cuda not found, use device 'cpu'")
            self.device = 'cpu'

        # Initialize optimisation parameters
        self.iter = 0

    def _preprocess_input(self, dataset):
        masker = self.masker or dataset.masker

        mask_img = masker.mask_img or masker.labels_img
        if isinstance(mask_img, str):
            mask_img = nib.load(mask_img)
        self.inputs_['mask_img'] = mask_img

        for name, (type_, _) in self._required_inputs.items():
            if type_ == "coordinates":
                # remove dataset coordinates outside of mask
                focus_filter = FocusFilter(mask=masker)
                dataset = focus_filter.transform(dataset)
                valid_dset_annotations = dataset.annotations[dataset.annotations['id'].isin(self.inputs_['id'])]
                all_group_study_id = dict()
                if isinstance(self.group_names, type(None)):
                    all_group_study_id[str(self.group_names)] = valid_dset_annotations['study_id'].unique().tolist()
                elif isinstance(self.group_names, str):
                    if self.group_names not in valid_dset_annotations.columns: 
                        raise ValueError("group_names: {} does not exist in the dataset".format(self.group_names))
                    else:
                        uniq_groups = list(valid_dset_annotations[self.group_names].unique())
                        for group in uniq_groups:
                            group_study_id_bool = valid_dset_annotations[self.group_names] == group
                            group_study_id = valid_dset_annotations.loc[group_study_id_bool]['study_id']
                            all_group_study_id[group] = group_study_id.unique().tolist()
                elif isinstance(self.group_names, list):
                    not_exist_group_names = [group for group in self.group_names if group not in dataset.annotations.columns]
                    if len(not_exist_group_names) > 0:
                        raise ValueError("group_names: {} does not exist in the dataset".format(not_exist_group_names))
                    uniq_group_splits = valid_dset_annotations[self.group_names].drop_duplicates().values.tolist()
                    for group in uniq_group_splits:
                        group_study_id_bool = (valid_dset_annotations[self.group_names] == group).all(axis=1)
                        group_study_id = valid_dset_annotations.loc[group_study_id_bool]['study_id']
                        all_group_study_id['_'.join(group)] = group_study_id.unique().tolist()
                self.inputs_['all_group_study_id'] = all_group_study_id
                # collect studywise moderators if specficed
                if hasattr(self, "moderators"):
                    all_group_moderators = dict()
                    for group in all_group_study_id.keys():
                        df_group = valid_dset_annotations.loc[valid_dset_annotations['study_id'].isin(all_group_study_id[group])] 
                        group_moderators = np.stack([df_group[moderator_name] for moderator_name in self.moderators], axis=1)
                        group_moderators = group_moderators.astype(np.float64)
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
                    group_foci_per_voxel = np.zeros(mask_img.shape, dtype=int)
                    for ijk in group_ijk:
                        group_foci_per_voxel[ijk[0], ijk[1], ijk[2]] += 1
                    # will not work with maskers that aren't NiftiMaskers
                    group_foci_per_voxel = nib.Nifti1Image(group_foci_per_voxel, mask_img.affine, mask_img.header)
                    group_foci_per_voxel = masker.transform(group_foci_per_voxel).transpose()
                    # number of foci per voxel/study
                    n_group_study = len(group_study_id)
                    group_foci_per_study = np.array([(group_coordinates['study_id']==i).sum() for i in group_study_id])
                    group_foci_per_study = group_foci_per_study.reshape((n_group_study, 1))

                    all_foci_per_voxel[group] = group_foci_per_voxel
                    all_foci_per_study[group] = group_foci_per_study
                
                self.inputs_['all_foci_per_voxel'] = all_foci_per_voxel
                self.inputs_['all_foci_per_study'] = all_foci_per_study

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
            cbmr_model = GLMPoisson(beta_dim=beta_dim, gamma_dim=gamma_dim, groups=self.groups, study_level_moderators=study_level_moderators, penalty=penalty, device=device)
        elif model == 'NB':
            cbmr_model = GLMNB(beta_dim=beta_dim, gamma_dim=gamma_dim, groups=self.groups, study_level_moderators=study_level_moderators, penalty=penalty, device=device)
        elif model == 'clustered_NB':
            cbmr_model = GLMCNB(beta_dim=beta_dim, gamma_dim=gamma_dim, groups=self.groups, study_level_moderators=study_level_moderators, penalty=penalty, device=device)
        if 'cuda' in device:
            cbmr_model = cbmr_model.cuda()
        
        return cbmr_model

    def _update(self, model, optimizer, Coef_spline_bases, all_moderators, all_foci_per_voxel, all_foci_per_study, prev_loss, gamma=0.999): 
        self.iter += 1
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=gamma) # learning rate decay
        scheduler.step()
        def closure():
            optimizer.zero_grad()
            loss = model(Coef_spline_bases, all_moderators, all_foci_per_voxel, all_foci_per_study)
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        # reset the L-BFGS params if NaN appears in coefficient of regression
        if any([torch.any(torch.isnan(model.all_beta_linears[group].weight)) for group in self.inputs_['all_group_study_id'].keys()]): 
            all_beta_linears, all_alpha_sqrt, all_alpha = dict(), dict(), dict()
            for group in self.inputs_['all_group_study_id'].keys():
                beta_dim = model.all_beta_linears[group].weight.shape[1]
                beta_linear_group = torch.nn.Linear(beta_dim, 1, bias=False).double()
                beta_linear_group.weight = torch.nn.Parameter(self.last_state['all_beta_linears.'+group+'.weight'])
                all_beta_linears[group] = beta_linear_group
                
                if self.model == 'NB':
                    group_alpha_sqrt = torch.nn.Parameter(self.last_state['all_alpha_sqrt.'+group])
                    all_alpha_sqrt[group] = group_alpha_sqrt
                elif self.model == 'clustered_NB':
                    group_alpha = torch.nn.Parameter(self.last_state['all_alpha.'+group])
                    all_alpha[group] = group_alpha
                
            model.all_beta_linears = torch.nn.ModuleDict(all_beta_linears)
            if self.model == 'NB':
                model.all_alpha_sqrt = torch.nn.ParameterDict(all_alpha_sqrt)
            elif self.model == 'clustered_NB':
                model.all_alpha = torch.nn.ParameterDict(all_alpha)

            LGR.debug(f"Reset L-BFGS optimizer......")
        else: 
            self.last_state = copy.deepcopy(model.state_dict()) # need to change the variable name?

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
        
        maps, tables = dict(), dict()
        Spatial_Regression_Coef = dict()
        # beta: regression coef of spatial effect
        for group in self.inputs_['all_group_study_id'].keys():
            group_beta_linear_weight = cbmr_model.all_beta_linears[group].weight
            group_beta_linear_weight = group_beta_linear_weight.cpu().detach().numpy().reshape((P,))
            Spatial_Regression_Coef[group] = group_beta_linear_weight
            group_studywise_spatial_intensity = np.exp(np.matmul(Coef_spline_bases, group_beta_linear_weight))
            maps[group+'_Studywise_Spatial_Intensity'] = group_studywise_spatial_intensity
            group_foci_per_voxel = self.inputs_['all_foci_per_voxel'][group].reshape((-1))
            maps[group+'_Foci_Per_Voxel'] = group_foci_per_voxel
            # overdispersion parameter: alpha
            # if self.model == 'NB':
            #     alpha = cbmr_model.all_alpha_sqrt[group]**2
            #     alpha = alpha.cpu().detach().numpy()
            #     overdispersion_param[group] = alpha
        tables['Spatial_Regression_Coef'] = pd.DataFrame.from_dict(Spatial_Regression_Coef, orient='index')
        # study-level moderators
        if hasattr(self, "moderators"):
            self._gamma = cbmr_model.gamma_linear.weight
            self._gamma = self._gamma.cpu().detach().numpy()
            for group in self.inputs_['all_group_study_id'].keys():
                group_moderators = self.inputs_["all_group_moderators"][group]
                moderators_effect = np.exp(np.matmul(group_moderators, self._gamma.T))
                maps[group+'_ModeratorsEffect'] = moderators_effect.flatten()
            tables['Moderators_Regression_Coef'] = pd.DataFrame(self._gamma, columns=self.moderators)
        else:
            self._gamma = None
        # if self.model == 'NB':
        #     tables['over_dispersion_param'] = pd.DataFrame.from_dict(overdispersion_param, orient='index')
        
        # standard error
        spatial_regression_coef_se, log_spatial_intensity_se, spatial_intensity_se = dict(), dict(), dict()
        Coef_spline_bases = torch.tensor(self.inputs_['Coef_spline_bases'], dtype=torch.float64, device=self.device)
        for group in self.inputs_['all_group_study_id'].keys():
            group_foci_per_voxel = torch.tensor(self.inputs_['all_foci_per_voxel'][group], dtype=torch.float64, device=self.device)
            group_foci_per_study = torch.tensor(self.inputs_['all_foci_per_study'][group], dtype=torch.float64, device=self.device)
            group_beta_linear_weight = cbmr_model.all_beta_linears[group].weight
            if hasattr(self, "moderators"):
                gamma = cbmr_model.gamma_linear.weight
                group_moderators = self.inputs_["all_group_moderators"][group]
                group_moderators = torch.tensor(group_moderators, dtype=torch.float64, device=self.device)
            else:
                group_moderators = None
            nll = lambda beta, gamma: -GLMPoisson._log_likelihood(beta, gamma, Coef_spline_bases, group_moderators, group_foci_per_voxel, group_foci_per_study)
            params = (group_beta_linear_weight, gamma)
            F = torch.autograd.functional.hessian(nll, params, create_graph=False, vectorize=True, outer_jacobian_strategy='forward-mode') 
            # Inference on regression coefficient of spatial effect
            spatial_dim = group_beta_linear_weight.shape[1]
            F_spatial_coef = F[0][0].reshape((spatial_dim, spatial_dim))
            Cov_spatial_coef = np.linalg.inv(F_spatial_coef.detach().numpy())
            Var_spatial_coef = np.diag(Cov_spatial_coef)
            SE_spatial_coef = np.sqrt(Var_spatial_coef)
            spatial_regression_coef_se[group] = SE_spatial_coef
    
            Var_log_spatial_intensity = np.einsum('ij,ji->i', self.inputs_['Coef_spline_bases'], Cov_spatial_coef @ self.inputs_['Coef_spline_bases'].T)
            SE_log_spatial_intensity = np.sqrt(Var_log_spatial_intensity)
            log_spatial_intensity_se[group] = SE_log_spatial_intensity
            
            group_studywise_spatial_intensity = maps[group+'_Studywise_Spatial_Intensity']
            SE_spatial_intensity = group_studywise_spatial_intensity * SE_log_spatial_intensity
            spatial_intensity_se[group] = SE_spatial_intensity

        tables['Spatial_Regression_Coef_SE'] = pd.DataFrame.from_dict(spatial_regression_coef_se, orient='index')
        tables['Log_Spatial_Intensity_SE'] = pd.DataFrame.from_dict(log_spatial_intensity_se, orient='index')
        tables['Spatial_Intensity_SE'] = pd.DataFrame.from_dict(spatial_intensity_se, orient='index')

        # Inference on regression coefficient of moderators
        if hasattr(self, "moderators"):
            gamma = gamma.cpu().detach().numpy()
            moderators_dim = gamma.shape[1]
            F_moderators_coef = F[1][1].reshape((moderators_dim, moderators_dim))
            Cov_moderators_coef = np.linalg.inv(F_moderators_coef.detach().numpy())
            Var_moderators = np.diag(Cov_moderators_coef).reshape((1, moderators_dim))
            SE_moderators = np.sqrt(Var_moderators)
            tables['Moderators_Regression_SE'] = pd.DataFrame(SE_moderators, columns=self.moderators)

        return maps, tables

class CBMRInference(object):

    def __init__(self, CBMRResults, spatial_homogeneity=True, t_con_group=None, t_con_moderator=None):
        self.maps, self.tables = CBMRResults.maps, CBMRResults.tables
        self.group_names = self.tables['Spatial_Regression_Coef'].index.values.tolist()
        self.n_groups = len(self.group_names)
        self.spatial_homogeneity = spatial_homogeneity
        self.t_con_group = t_con_group
        self.t_con_moderator = t_con_moderator
    
    def _contrast(self):
        if not isinstance(self.t_con_group, type(None)):
            self.t_con_group = np.array(self.t_con_group)
            if self.t_con_group.shape[1] != self.n_groups:
                raise ValueError("The shape of contrast matrix doesn't match with groups")
            if np.any(np.sum(self.t_con_group, axis=1)==0):
                raise ValueError("Conflict happens between the input contrast matrix and spatial homogeneity test")
            self.t_con_group = self.t_con_group / np.sum(self.t_con_group, axis=1)

            Spatial_Intensity_SE = self.tables['Spatial_Intensity_SE']
            if self.spatial_homogeneity: # GLH 1 group
                for group in self.group_names:
                    # mu_0 under null hypothesis 
                    group_foci_per_voxel, group_moderators_effect = self.maps[group+'_Foci_Per_Voxel'], self.maps[group+'_ModeratorsEffect']
                    n_voxels, n_study = group_foci_per_voxel.shape[0], group_moderators_effect.shape[0]
                    null_spatial_intensity = np.sum(group_foci_per_voxel) / (n_voxels * n_study)
                    SE_spatial_intensity = Spatial_Intensity_SE.loc[Spatial_Intensity_SE.index == group].to_numpy().reshape((-1))
                    group_Z_stat = (self.maps[group+'_Studywise_Spatial_Intensity'] - null_spatial_intensity) / SE_spatial_intensity
                    self.maps[group+'_z'] = group_Z_stat
                    group_p_vals = z_to_p(group_Z_stat, tail='one')
                    self.maps[group+'_p'] = group_p_vals
            else: # GLH multiple groups
                print('123')
                
        # Wald_statistics_moderators = gamma / np.sqrt(Var_moderators)
        # p_moderators = transforms.z_to_p(z=Wald_statistics_moderators, tail='two')
        
        return

class GLMPoisson(torch.nn.Module):
    def __init__(self, beta_dim=None, gamma_dim=None, groups=None, study_level_moderators=False, penalty=False, device='cpu'):
        super().__init__()
        self.beta_dim = beta_dim
        self.gamma_dim = gamma_dim
        self.groups = groups
        self.study_level_moderators = study_level_moderators
        self.penalty = penalty
        self.device = device
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
    
    def _log_likelihood(beta, gamma, Coef_spline_bases, moderators, foci_per_voxel, foci_per_study):
        log_mu_spatial = torch.matmul(Coef_spline_bases, beta.T)
        mu_spatial = torch.exp(log_mu_spatial)
        log_mu_moderators = torch.matmul(moderators, gamma.T)
        mu_moderators = torch.exp(log_mu_moderators)
        log_l = torch.sum(torch.mul(foci_per_voxel, log_mu_spatial)) + torch.sum(torch.mul(foci_per_study, log_mu_moderators)) \
                        - torch.sum(mu_spatial) * torch.sum(mu_moderators)

        return log_l

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
        
        if self.penalty == True:
            # Firth-type penalty 
            for group in all_foci_per_voxel.keys(): 
                beta = self.all_beta_linears[group].weight.T
                beta_dim = beta.shape[0]
                gamma = self.gamma_linear.weight.T
                group_foci_per_voxel = all_foci_per_voxel[group]
                group_foci_per_study = all_foci_per_study[group] 
                group_moderators = all_moderators[group]
                nll = lambda beta: -self._log_likelihood(beta, gamma, Coef_spline_bases, group_moderators, group_foci_per_voxel, group_foci_per_study)
                params = (beta)
                F = torch.autograd.functional.hessian(nll, params, create_graph=False, vectorize=True, outer_jacobian_strategy='forward-mode') 
                F = F.reshape((beta_dim, beta_dim))
                eig_vals = torch.real(torch.linalg.eigvals(F)) #torch.eig(F, eigenvectors=False)[0][:,0] 
                del F
                group_firth_penalty = 0.5 * torch.sum(torch.log(eig_vals))
                del eig_vals
                log_l += group_firth_penalty
        return -log_l

class GLMNB(torch.nn.Module):
    def __init__(self, beta_dim=None, gamma_dim=None, groups=None, study_level_moderators=False, penalty='No', device='cpu'):
        super().__init__()
        self.groups = groups
        self.study_level_moderators = study_level_moderators
        self.penalty = penalty
        self.device = device
        # initialization for beta
        all_beta_linears, all_alpha_sqrt = dict(), dict()
        for group in groups:
            beta_linear_group = torch.nn.Linear(beta_dim, 1, bias=False).double()
            torch.nn.init.uniform_(beta_linear_group.weight, a=-0.01, b=0.01)
            all_beta_linears[group] = beta_linear_group
            # initialization for alpha
            alpha_init_group = torch.tensor(1e-2).double()
            all_alpha_sqrt[group] = torch.nn.Parameter(torch.sqrt(alpha_init_group), requires_grad=True) 
        self.all_beta_linears = torch.nn.ModuleDict(all_beta_linears)
        self.all_alpha_sqrt = torch.nn.ParameterDict(all_alpha_sqrt)
        # gamma 
        if self.study_level_moderators:
            self.gamma_linear = torch.nn.Linear(gamma_dim, 1, bias=False).double()
            torch.nn.init.uniform_(self.gamma_linear.weight, a=-0.01, b=0.01)
    
    def _three_term(y, r, device):
        max_foci = torch.max(y).to(dtype=torch.int64, device=device)
        sum_three_term = 0
        for k in range(max_foci):
            foci_index = (y == k+1).nonzero()[:,0]
            r_j = r[foci_index]
            n_voxel = list(foci_index.shape)[0]
            y_j = torch.tensor([k+1]*n_voxel, device=device).double()
            y_j = y_j.reshape((n_voxel, 1))
            # y=0 => sum_three_term = 0
            sum_three_term += torch.sum(torch.lgamma(y_j+r_j) - torch.lgamma(y_j+1) - torch.lgamma(r_j))
        
        return sum_three_term
    
    def _log_likelihood(self, alpha, beta, gamma, Coef_spline_bases, group_moderators, group_foci_per_voxel, group_foci_per_study):
        v = 1 / alpha
        log_mu_spatial = Coef_spline_bases @ beta
        mu_spatial = torch.exp(log_mu_spatial)
        log_mu_moderators = group_moderators @ gamma
        mu_moderators = torch.exp(log_mu_moderators)
        numerator = mu_spatial**2 * torch.sum(mu_moderators**2)
        denominator = mu_spatial**2 * torch.sum(mu_moderators)**2
        estimated_sum_alpha = alpha * numerator / denominator

        p = numerator / (v * mu_spatial * torch.sum(mu_moderators) + numerator)
        r = v * denominator / numerator

        log_l = GLMNB._three_term(group_foci_per_voxel,r, device=self.device) + torch.sum(r*torch.log(1-p) + group_foci_per_voxel*torch.log(p))

        return log_l
    
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
            alpha = self.all_alpha_sqrt[group]**2
            v = 1 / alpha
            log_mu_spatial = self.all_beta_linears[group](Coef_spline_bases)
            mu_spatial = torch.exp(log_mu_spatial)
            log_mu_moderators = all_log_mu_moderators[group]
            mu_moderators = torch.exp(log_mu_moderators)
            # Now the sum of NB variates are no long NB distributed (since mu_ij != mu_i'j),
            # Therefore, we use moment matching approach,
            # create a new NB approximation to the mixture of NB distributions: 
            # alpha' = sum_i mu_{ij}^2 / (sum_i mu_{ij})^2 * alpha
            numerator = mu_spatial**2 * torch.sum(mu_moderators**2)
            denominator = mu_spatial**2 * torch.sum(mu_moderators)**2
            estimated_sum_alpha = alpha * numerator / denominator
            ## moment matching NB distribution
            p = numerator / (v*mu_spatial*torch.sum(mu_moderators) + numerator)
            r = v * denominator / numerator

            group_foci_per_voxel = all_foci_per_voxel[group]
            # group_foci_per_study = all_foci_per_study[group]
            group_log_l = GLMNB._three_term(group_foci_per_voxel,r, device=self.device) + torch.sum(r*torch.log(1-p) + group_foci_per_voxel*torch.log(p)) 
            log_l += group_log_l
        
        if self.penalty == True:
            # Firth-type penalty 
            for group in all_foci_per_voxel.keys(): 
                alpha = self.all_alpha_sqrt[group]**2
                beta = self.all_beta_linears[group].weight.T
                beta_dim = beta.shape[0]
                gamma = self.gamma_linear.weight.detach().T
                group_foci_per_voxel = all_foci_per_voxel[group]
                group_foci_per_study = all_foci_per_study[group]
                group_moderators = all_moderators[group]
                # a = -self._log_likelihood(alpha, beta, gamma, Coef_spline_bases, group_moderators, group_foci_per_voxel, group_foci_per_study)
                nll = lambda beta: -self._log_likelihood(alpha, beta, gamma, Coef_spline_bases, group_moderators, group_foci_per_voxel, group_foci_per_study)
                params = (beta)
                F = torch.autograd.functional.hessian(nll, params, create_graph=True)
                F = F.reshape((beta_dim, beta_dim))
                eig_vals = eig_vals = torch.real(torch.linalg.eigvals(F))
                del F
                group_firth_penalty = 0.5 * torch.sum(torch.log(eig_vals))
                del eig_vals
                log_l += group_firth_penalty
        
        return -log_l

class GLMCNB(torch.nn.Module):
    def __init__(self, beta_dim=None, gamma_dim=None, groups=None, study_level_moderators=False, penalty=True, device='cpu'):
        super().__init__()
        self.groups = groups
        self.study_level_moderators = study_level_moderators
        self.penalty = penalty
        # initialization for beta
        all_beta_linears, all_alpha = dict(), dict()
        for group in groups:
            beta_linear_group = torch.nn.Linear(beta_dim, 1, bias=False).double()
            torch.nn.init.uniform_(beta_linear_group.weight, a=-0.01, b=0.01)
            all_beta_linears[group] = beta_linear_group
            # initialization for alpha
            alpha_init_group = torch.tensor(1e-2).double()
            all_alpha[group] = torch.nn.Parameter(alpha_init_group, requires_grad=True) 
        self.all_beta_linears = torch.nn.ModuleDict(all_beta_linears)
        self.all_alpha = torch.nn.ParameterDict(all_alpha)
        # gamma 
        if self.study_level_moderators:
            self.gamma_linear = torch.nn.Linear(gamma_dim, 1, bias=False).double()
            torch.nn.init.uniform_(self.gamma_linear.weight, a=-0.01, b=0.01)
    
    def _log_likelihood(self, alpha, beta, gamma, Coef_spline_bases, group_moderators, group_foci_per_voxel, group_foci_per_study):
        v = 1 / alpha
        log_mu_spatial = Coef_spline_bases @ beta
        mu_spatial = torch.exp(log_mu_spatial)
        log_mu_moderators = group_moderators @ gamma
        mu_moderators = torch.exp(log_mu_moderators)
        mu_sum_per_study = torch.sum(mu_spatial) * mu_moderators

        group_n_study, group_n_voxel = mu_moderators.shape[0], mu_spatial.shape[0]

        log_l = group_n_study * v * torch.log(v) - group_n_study * torch.lgamma(v) + torch.sum(torch.lgamma(group_foci_per_study + v)) - torch.sum((group_foci_per_study + v) * torch.log(mu_sum_per_study + v)) \
            + torch.sum(group_foci_per_voxel * log_mu_spatial) + torch.sum(group_foci_per_study * log_mu_moderators)

        return log_l


    def forward(self, Coef_spline_bases, all_moderators, all_foci_per_voxel, all_foci_per_study):
        if isinstance(all_moderators, dict):
            all_log_mu_moderators = dict()
            for group in all_moderators.keys():
                group_moderators = all_moderators[group]
                # mu^Z = exp(Z * gamma)
                log_mu_moderators = self.gamma_linear(group_moderators)
                all_log_mu_moderators[group] = log_mu_moderators
        log_l = 0
        for group in all_foci_per_voxel.keys(): 
            alpha = self.all_alpha[group]
            v = 1 / alpha
            log_mu_spatial = self.all_beta_linears[group](Coef_spline_bases)
            mu_spatial = torch.exp(log_mu_spatial)
            log_mu_moderators = all_log_mu_moderators[group]
            mu_moderators = torch.exp(log_mu_moderators)

            group_foci_per_voxel = all_foci_per_voxel[group]
            group_foci_per_study = all_foci_per_study[group]
            group_n_study, group_n_voxel = mu_moderators.shape[0], mu_spatial.shape[0]
            
            mu_sum_per_study = torch.sum(mu_spatial) * mu_moderators
            group_log_l = group_n_study * v * torch.log(v) - group_n_study * torch.lgamma(v) + torch.sum(torch.lgamma(group_foci_per_study + v)) - torch.sum((group_foci_per_study + v) * torch.log(mu_sum_per_study + v)) \
                + torch.sum(group_foci_per_voxel * log_mu_spatial) + torch.sum(group_foci_per_study * log_mu_moderators)
            log_l += group_log_l
        
        if self.penalty == True:
            # Firth-type penalty 
            for group in all_foci_per_voxel.keys(): 
                alpha = self.all_alpha[group]
                beta = self.all_beta_linears[group].weight.T
                beta_dim = beta.shape[0]
                gamma = self.gamma_linear.weight.T
                group_foci_per_voxel = all_foci_per_voxel[group]
                group_foci_per_study = all_foci_per_study[group]
                group_moderators = all_moderators[group]
                nll = lambda beta: -self._log_likelihood(alpha, beta, gamma, Coef_spline_bases, group_moderators, group_foci_per_voxel, group_foci_per_study)
                params = (beta)
                F = torch.autograd.functional.hessian(nll, params, create_graph=True) # vectorize=True, outer_jacobian_strategy='forward-mode'
                # F = hessian(nll)(beta)
                F = F.reshape((beta_dim, beta_dim))
                eig_vals = torch.real(torch.linalg.eigvals(F))
                del F
                group_firth_penalty = 0.5 * torch.sum(torch.log(eig_vals))
                del eig_vals
                log_l += group_firth_penalty

        return -log_l