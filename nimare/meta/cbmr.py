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
import torch
import logging
import copy

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
                    all_group_study_id[self.group_names] = valid_dset_annotations['study_id'].unique().tolist()
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
            cbmr_model = GLMPoisson(beta_dim=beta_dim, gamma_dim=gamma_dim, groups=self.groups, study_level_moderators=study_level_moderators, penalty=penalty)
        elif model == 'NB':
            cbmr_model = GLMNB(beta_dim=beta_dim, gamma_dim=gamma_dim, groups=self.groups, study_level_moderators=study_level_moderators, penalty=penalty)
        elif model == 'clustered_NB':
            cbmr_model = GLMCNB(beta_dim=beta_dim, gamma_dim=gamma_dim, groups=self.groups, study_level_moderators=study_level_moderators, penalty=penalty)
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
        # reset the L-BFGS params if NaN appears in coefficient of regression
        if any([torch.any(torch.isnan(model.all_beta_linears[group].weight)) for group in self.inputs_['all_group_study_id'].keys()]): 
            all_beta_linears, all_alpha_sqrt = dict(), dict()
            for group in self.inputs_['all_group_study_id'].keys():
                beta_dim = model.all_beta_linears[group].weight.shape[1]
                beta_linear_group = torch.nn.Linear(beta_dim, 1, bias=False).double()
                beta_linear_group.weight = torch.nn.Parameter(self.last_state['all_beta_linears.'+group+'.weight'])
                group_alpha_sqrt = torch.nn.Parameter(self.last_state['all_alpha_sqrt.'+group])
                
                all_beta_linears[group] = beta_linear_group
                all_alpha_sqrt[group] = group_alpha_sqrt
            model.all_beta_linears = torch.nn.ModuleDict(all_beta_linears)
            model.all_alpha_sqrt = torch.nn.ParameterDict(all_alpha_sqrt)
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
        spatial_regression_coef, overdispersion_param = dict(), dict()
        # beta: regression coef of spatial effect
        for group in self.inputs_['all_group_study_id'].keys():
            group_beta_linear_weight = cbmr_model.all_beta_linears[group].weight
            group_beta_linear_weight = group_beta_linear_weight.cpu().detach().numpy().reshape((P,))
            spatial_regression_coef[group] = group_beta_linear_weight
            studywise_spatial_intensity = np.exp(np.matmul(Coef_spline_bases, group_beta_linear_weight))
            maps[group+'_group_StudywiseIntensity'] = studywise_spatial_intensity
            # overdispersion parameter: alpha
            if self.model == 'NB':
                alpha = cbmr_model.all_alpha_sqrt[group]**2
                alpha = alpha.cpu().detach().numpy()
                overdispersion_param[group] = alpha
        tables['spatial_regression_coef'] = pd.DataFrame.from_dict(spatial_regression_coef, orient='index')

        # study-level moderators
        if hasattr(self, "moderators"):
            self._gamma = cbmr_model.gamma_linear.weight
            self._gamma = self._gamma.cpu().detach().numpy()
            for group in self.inputs_['all_group_study_id'].keys():
                group_moderators = self.inputs_["all_group_moderators"][group]
                moderators_effect = np.exp(np.matmul(group_moderators, self._gamma.T))
                maps[group+'_group_ModeratorsEffect'] = moderators_effect.flatten()
            tables['moderators_regression_coef'] = pd.DataFrame(self._gamma, columns=self.moderators)
        if self.model == 'NB':
            tables['over_dispersion_param'] = pd.DataFrame.from_dict(overdispersion_param, orient='index')

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

class GLMNB(torch.nn.Module):
    def __init__(self, beta_dim=None, gamma_dim=None, groups=None, study_level_moderators=False, penalty='No'):
        super().__init__()
        self.groups = groups
        self.study_level_moderators = study_level_moderators
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
    
    def _three_term(y, r):
        max_foci = np.int(torch.max(y).item())
        sum_three_term = 0
        for k in range(max_foci):
            foci_index = (y == k+1).nonzero()[:,0]
            r_j = r[foci_index]
            n_voxel = list(foci_index.shape)[0]
            y_j = torch.tensor([k+1]*n_voxel).double()
            y_j = y_j.reshape((n_voxel, 1))
            # y=0 => sum_three_term = 0
            sum_three_term += torch.sum(torch.lgamma(y_j+r_j) - torch.lgamma(y_j+1) - torch.lgamma(r_j))
        
        return sum_three_term
    
    
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
            group_log_l = GLMNB._three_term(group_foci_per_voxel,r) + torch.sum(r*torch.log(1-p) + group_foci_per_voxel*torch.log(p))
            log_l += group_log_l
        
        return -log_l

class GLMCNB(torch.nn.Module):
    def __init__(self, beta_dim=None, gamma_dim=None, groups=None, study_level_moderators=False, penalty='No'):
        super().__init__()
        self.groups = groups
        self.study_level_moderators = study_level_moderators
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
            
            group_log_l = group_n_study * v * torch.log(v) - group_n_study * torch.lgamma(v) + torch.sum(torch.lgamma(group_foci_per_study + v)) - torch.sum((group_foci_per_study + v) * torch.log(mu_moderators + v)) \
                + torch.sum(group_foci_per_voxel * log_mu_spatial) + torch.sum(group_foci_per_study * log_mu_moderators)
            log_l += group_log_l

        return -log_l