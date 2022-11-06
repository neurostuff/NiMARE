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
import functorch
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
                if self.moderators:
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
        if self.moderators:
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
        if self.moderators:
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
        Spatial_Regression_Coef, overdispersion_param = dict(), dict()
        # beta: regression coef of spatial effect
        for group in self.inputs_['all_group_study_id'].keys():
            group_beta_linear_weight = cbmr_model.all_beta_linears[group].weight
            group_beta_linear_weight = group_beta_linear_weight.cpu().detach().numpy().reshape((P,))
            Spatial_Regression_Coef[group] = group_beta_linear_weight
            group_studywise_spatial_intensity = np.exp(np.matmul(Coef_spline_bases, group_beta_linear_weight))
            maps['Group_'+group+'_Studywise_Spatial_Intensity'] = group_studywise_spatial_intensity#.reshape((1,-1))
            # overdispersion parameter: alpha
            if self.model == 'NB':
                alpha = cbmr_model.all_alpha_sqrt[group]**2
                alpha = alpha.cpu().detach().numpy()
                overdispersion_param[group] = alpha
            elif self.model == 'clustered_NB':
                alpha = cbmr_model.all_alpha[group]
                alpha = alpha.cpu().detach().numpy()
                overdispersion_param[group] = alpha
        tables['Spatial_Regression_Coef'] = pd.DataFrame.from_dict(Spatial_Regression_Coef, orient='index')
        if self.model == 'NB' or self.model == 'clustered_NB':
            tables['Overdispersion_Coef'] = pd.DataFrame.from_dict(overdispersion_param, orient='index', columns=['alpha'])
        # study-level moderators
        if self.moderators:
            self.moderators_effect = dict()
            self._gamma = cbmr_model.gamma_linear.weight
            self._gamma = self._gamma.cpu().detach().numpy()
            for group in self.inputs_['all_group_study_id'].keys():
                group_moderators = self.inputs_["all_group_moderators"][group]
                group_moderators_effect = np.exp(np.matmul(group_moderators, self._gamma.T))
                self.moderators_effect[group] = group_moderators_effect
            tables['Moderators_Regression_Coef'] = pd.DataFrame(self._gamma, columns=self.moderators)
        else:
            self._gamma = None
        # standard error
        spatial_regression_coef_se, log_spatial_intensity_se, spatial_intensity_se = dict(), dict(), dict()
        Coef_spline_bases = torch.tensor(self.inputs_['Coef_spline_bases'], dtype=torch.float64, device=self.device)
        for group in self.inputs_['all_group_study_id'].keys():
            group_foci_per_voxel = torch.tensor(self.inputs_['all_foci_per_voxel'][group], dtype=torch.float64, device=self.device)
            group_foci_per_study = torch.tensor(self.inputs_['all_foci_per_study'][group], dtype=torch.float64, device=self.device)
            group_beta_linear_weight = cbmr_model.all_beta_linears[group].weight
            if self.moderators:
                gamma = cbmr_model.gamma_linear.weight
                group_moderators = self.inputs_["all_group_moderators"][group]
                group_moderators = torch.tensor(group_moderators, dtype=torch.float64, device=self.device)
            else:
                gamma, group_moderators = None, None
            if 'Overdispersion_Coef' in tables.keys():
                alpha = torch.tensor(tables['Overdispersion_Coef'].to_dict()['alpha'][group], dtype=torch.float64, device=self.device)
            # a = -GLMCNB._log_likelihood_single_group(alpha, group_beta_linear_weight, gamma, Coef_spline_bases, group_moderators, group_foci_per_voxel, group_foci_per_study, self.device)
            if self.model == 'Poisson':
                nll = lambda beta: -GLMPoisson._log_likelihood_single_group(beta, gamma, Coef_spline_bases, group_moderators, group_foci_per_voxel, group_foci_per_study, self.device)
            elif self.model == 'NB': 
                nll = lambda beta: -GLMNB._log_likelihood_single_group(alpha, beta, gamma, Coef_spline_bases, group_moderators, group_foci_per_voxel, group_foci_per_study, self.device)
            elif self.model == 'clustered_NB': 
                nll = lambda beta: -GLMCNB._log_likelihood_single_group(alpha, beta, gamma, Coef_spline_bases, group_moderators, group_foci_per_voxel, group_foci_per_study, self.device)
            F = functorch.hessian(nll)(group_beta_linear_weight)
            # Inference on regression coefficient of spatial effect
            spatial_dim = group_beta_linear_weight.shape[1]
            F_spatial_coef = F.reshape((spatial_dim, spatial_dim))
            Cov_spatial_coef = np.linalg.inv(F_spatial_coef.detach().numpy())
            Var_spatial_coef = np.diag(Cov_spatial_coef)
            SE_spatial_coef = np.sqrt(Var_spatial_coef)
            spatial_regression_coef_se[group] = SE_spatial_coef
    
            Var_log_spatial_intensity = np.einsum('ij,ji->i', self.inputs_['Coef_spline_bases'], Cov_spatial_coef @ self.inputs_['Coef_spline_bases'].T)
            SE_log_spatial_intensity = np.sqrt(Var_log_spatial_intensity)
            log_spatial_intensity_se[group] = SE_log_spatial_intensity
            
            group_studywise_spatial_intensity = maps['Group_'+group+'_Studywise_Spatial_Intensity'].reshape((-1))
            SE_spatial_intensity = group_studywise_spatial_intensity * SE_log_spatial_intensity
            spatial_intensity_se[group] = SE_spatial_intensity

        tables['Spatial_Regression_Coef_SE'] = pd.DataFrame.from_dict(spatial_regression_coef_se, orient='index')
        tables['Log_Spatial_Intensity_SE'] = pd.DataFrame.from_dict(log_spatial_intensity_se, orient='index')
        tables['Spatial_Intensity_SE'] = pd.DataFrame.from_dict(spatial_intensity_se, orient='index')

        # Inference on regression coefficient of moderators
        if self.moderators:
            moderators_dim = gamma.shape[1]
            nll = lambda gamma: -GLMPoisson._log_likelihood_single_group(group_beta_linear_weight, gamma, Coef_spline_bases, group_moderators, group_foci_per_voxel, group_foci_per_study, self.device)
            params = (gamma)
            F_moderators_coef = torch.autograd.functional.hessian(nll, params, create_graph=False, vectorize=True, outer_jacobian_strategy='forward-mode')
            F_moderators_coef = F_moderators_coef.reshape((moderators_dim, moderators_dim))
            Cov_moderators_coef = np.linalg.inv(F_moderators_coef.detach().numpy())
            Var_moderators = np.diag(Cov_moderators_coef).reshape((1, moderators_dim))
            SE_moderators = np.sqrt(Var_moderators)
            tables['Moderators_Regression_SE'] = pd.DataFrame(SE_moderators, columns=self.moderators)

        return maps, tables

class CBMRInference(object):
    def __init__(self, CBMRResults, t_con_group=None, t_con_moderator=None, device='cpu'):
        self.device = device
        self.CBMRResults = CBMRResults
        self.t_con_group = t_con_group
        self.t_con_moderator = t_con_moderator
        self.group_names = self.CBMRResults.tables['Spatial_Regression_Coef'].index.values.tolist()
        self.n_groups = len(self.group_names)
        if self.t_con_group is not False:
            # Conduct group-wise spatial homogeneity test by default
            self.t_con_group = [np.eye(self.n_groups)] if not self.t_con_group else [np.array(con_group) for con_group in self.t_con_group]
            self.t_con_group = [con_group.reshape((1,-1)) if len(con_group.shape)==1 else con_group for con_group in self.t_con_group] # 2D contrast matrix/vector
            if np.any([con_group.shape[1] != self.n_groups for con_group in self.t_con_group]):
                wrong_con_group_idx = np.where([con_group.shape[1] != self.n_groups for con_group in self.t_con_group])[0].tolist()
                raise ValueError("The shape of {}th contrast vector(s) in group-wise intensity contrast matrix doesn't match with groups".format(str(wrong_con_group_idx)))
            con_group_zero_row = [np.where(np.sum(np.abs(con_group), axis=1) == 0)[0] for con_group in self.t_con_group]
            if np.any([len(zero_row)>0 for zero_row in con_group_zero_row]): # remove zero rows in contrast matrix
                self.t_con_group = [np.delete(self.t_con_group[i], con_group_zero_row[i], axis=0) for i in range(len(self.t_con_group))]
                if np.any([con_group.shape[0]== 0 for con_group in self.t_con_group]):
                    raise ValueError('One or more of contrast vectors(s) in group-wise intensity contrast matrix are all zeros')
            n_contrasts_group = [con_group.shape[0] for con_group in self.t_con_group]
            self._Name_of_con_group()
            # standardization
            self.t_con_group = [con_group / np.sum(np.abs(con_group), axis=1).reshape((-1,1)) for con_group in self.t_con_group]
        
        if self.t_con_moderator is not False:
            if self.CBMRResults.estimator.moderators:
                self.moderator_names = self.CBMRResults.estimator.moderators
                self.n_moderators = len(self.moderator_names)
                self.t_con_moderator = [np.eye(self.n_moderators)] if not self.t_con_moderator else [np.array(con_moderator) for con_moderator in self.t_con_moderator]
                self.t_con_moderator = [con_moderator.reshape((1,-1)) if len(con_moderator.shape)==1 else con_moderator for con_moderator in self.t_con_moderator]
                # test the existence of effect of moderators
                if np.any([con_moderator.shape[1] != self.n_moderators for con_moderator in self.t_con_moderator]):
                    wrong_con_moderator_idx = np.where([con_moderator.shape[1] != self.n_moderators for con_moderator in self.t_con_moderator])[0].tolist()
                    raise ValueError("The shape of {}th contrast vector(s) in moderators contrast matrix doesn't match with moderators".format(str(wrong_con_moderator_idx)))
                con_moderator_zero_row = [np.where(np.sum(np.abs(con_modereator), axis=1)==0)[0] for con_modereator in self.t_con_moderator]
                if np.any([len(zero_row)>0 for zero_row in con_moderator_zero_row]): # remove zero rows in contrast matrix
                    self.t_con_moderator = [np.delete(self.t_con_moderator[i], con_moderator_zero_row[i], axis=0) for i in range(len(self.t_con_moderator))]
                    if np.any([con_moderator.shape[0]== 0 for con_moderator in self.t_con_moderator]):
                        raise ValueError('One or more of contrast vectors(s) in modereators contrast matrix are all zeros')
                n_contrasts_moderator = [con_moderator.shape[0] for con_moderator in self.t_con_moderator] 
                self._Name_of_con_moderator()
                self.t_con_moderator = [con_moderator / np.sum(np.abs(con_moderator), axis=1).reshape((-1,1)) for con_moderator in self.t_con_moderator]
            else:
                self.t_con_moderator = False
        if self.device == 'cuda' and not torch.cuda.is_available(): 
            LGR.debug(f"cuda not found, use device 'cpu'")
            self.device = 'cpu'

    def _Name_of_con_group(self):
        self.t_con_group_name = list()
        for con_group in self.t_con_group:
            con_group_name = list()
            for num, idx in enumerate(con_group): 
                if np.sum(idx) != 0: # homogeneity test
                    nonzero_con_group_info = str()
                    nonzero_group_index = np.where(idx!=0)[0].tolist()
                    nonzero_group_name = [self.group_names[i] for i in nonzero_group_index]
                    nonzero_con = [int(idx[i]) for i in nonzero_group_index]
                    for i in range(len(nonzero_group_index)):
                        nonzero_con_group_info += str(abs(nonzero_con[i])) + 'x' + str(nonzero_group_name[i])
                    con_group_name.append('homo_test_' + nonzero_con_group_info)
                else: # group-comparison test
                    pos_group_idx, neg_group_idx = np.where(idx>0)[0].tolist(), np.where(idx<0)[0].tolist()
                    pos_group_name, neg_group_name = [self.group_names[i] for i in pos_group_idx], [self.group_names[i] for i in neg_group_idx]
                    pos_group_con, neg_group_con = [int(idx[i]) for i in pos_group_idx], [int(idx[i]) for i in neg_group_idx]
                    pos_con_group_info, neg_con_group_info = str(), str()
                    for i in range(len(pos_group_idx)):
                        pos_con_group_info += str(pos_group_con[i]) + 'x' + str(pos_group_name[i])
                    for i in range(len(neg_group_idx)):
                        neg_con_group_info += str(abs(neg_group_con[i])) + 'x' + str(neg_group_name[i])
                    con_group_name.append(pos_con_group_info + 'VS' + neg_con_group_info)
            self.t_con_group_name.append(con_group_name)
        return
    
    def _Name_of_con_moderator(self):
        self.t_con_moderator_name = list()
        for con_moderator in self.t_con_moderator:
            con_moderator_name = list()
            for num, idx in enumerate(con_moderator): 
                if np.sum(idx) != 0: # homogeneity test
                    nonzero_con_moderator_info = str()
                    nonzero_moderator_index = np.where(idx!=0)[0].tolist()
                    nonzero_moderator_name = [self.moderator_names[i] for i in nonzero_moderator_index]
                    nonzero_con = [int(idx[i]) for i in nonzero_moderator_index]
                    for i in range(len(nonzero_moderator_index)):
                        nonzero_con_moderator_info += str(abs(nonzero_con[i])) + 'x' + str(nonzero_moderator_name[i])
                    con_moderator_name.append('Effect_of_' + nonzero_con_moderator_info)
                else: # group-comparison test
                    pos_moderator_idx, neg_moderator_idx = np.where(idx>0)[0].tolist(), np.where(idx<0)[0].tolist()
                    pos_moderator_name, neg_moderator_name = [self.moderator_names[i] for i in pos_moderator_idx], [self.moderator_names[i] for i in neg_moderator_idx]
                    pos_moderator_con, neg_moderator_con = [int(idx[i]) for i in pos_moderator_idx], [int(idx[i]) for i in neg_moderator_idx]
                    pos_con_moderator_info, neg_con_moderator_info = str(), str()
                    for i in range(len(pos_moderator_idx)):
                        pos_con_moderator_info += str(pos_moderator_con[i]) + 'x' + str(pos_moderator_name[i])
                    for i in range(len(neg_moderator_idx)):
                        neg_con_moderator_info += str(abs(neg_moderator_con[i])) + 'x' + str(neg_moderator_name[i])
                    con_moderator_name.append(pos_con_moderator_info + 'VS' + neg_con_moderator_info)
            self.t_con_moderator_name.append(con_moderator_name)
        return

    def _Fisher_info_spatial_coef(self, GLH_involved_index):
        Coef_spline_bases = torch.tensor(self.CBMRResults.estimator.inputs_['Coef_spline_bases'], dtype=torch.float64, device=self.device)
        GLH_involved = [self.group_names[i] for i in GLH_involved_index]
        involved_group_foci_per_voxel = [torch.tensor(self.CBMRResults.estimator.inputs_['all_foci_per_voxel'][group], dtype=torch.float64, device=self.device) for group in GLH_involved]
        involved_group_foci_per_study = [torch.tensor(self.CBMRResults.estimator.inputs_['all_foci_per_study'][group], dtype=torch.float64, device=self.device) for group in GLH_involved]
        if 'Overdispersion_Coef' in self.CBMRResults.tables.keys():
            involved_overdispersion_coef = torch.tensor([self.CBMRResults.tables['Overdispersion_Coef'].to_numpy()[i, :] for i in GLH_involved_index], dtype=torch.float64, device=self.device)
        involved_spatial_coef = torch.tensor([self.CBMRResults.tables['Spatial_Regression_Coef'].to_numpy()[i, :].reshape((-1,1)) for i in GLH_involved_index], dtype=torch.float64, device=self.device)
        n_involved_groups, spatial_coef_dim, _ = involved_spatial_coef.shape
        if self.CBMRResults.estimator.moderators:
            involved_group_moderators = [torch.tensor(self.CBMRResults.estimator.inputs_['all_group_moderators'][group], dtype=torch.float64, device=self.device) for group in GLH_involved]
            involved_moderator_coef = torch.tensor(self.CBMRResults.tables['Moderators_Regression_Coef'].to_numpy().T, dtype=torch.float64, device=self.device)
        else:
            involved_group_moderators, involved_moderator_coef = None, None
        # a = GLMPoisson._log_likelihood_mult_group(involved_spatial_coef, Coef_spline_bases,  involved_group_foci_per_voxel, involved_group_foci_per_study, involved_moderator_coef, involved_group_moderators, self.device)
        if self.CBMRResults.estimator.model == 'Poisson':
            nll = lambda all_spatial_coef: -GLMPoisson._log_likelihood_mult_group(all_spatial_coef, Coef_spline_bases, involved_group_foci_per_voxel, involved_group_foci_per_study, involved_moderator_coef, involved_group_moderators)
        elif self.CBMRResults.estimator.model == 'NB':
            nll = lambda all_spatial_coef: -GLMNB._log_likelihood_mult_group(involved_overdispersion_coef, all_spatial_coef, Coef_spline_bases, involved_group_foci_per_voxel, involved_group_foci_per_study, involved_moderator_coef, involved_group_moderators)
        elif self.CBMRResults.estimator.model == 'clustered_NB':
            nll = lambda all_spatial_coef: -GLMCNB._log_likelihood_mult_group(involved_overdispersion_coef, all_spatial_coef, Coef_spline_bases, involved_group_foci_per_voxel, involved_group_foci_per_study, involved_moderator_coef, involved_group_moderators)
        h = functorch.hessian(nll)(involved_spatial_coef)
        h = h.view(n_involved_groups*spatial_coef_dim, -1)

        return h.detach().cpu().numpy()

    def _Fisher_info_moderator_coef(self):
        Coef_spline_bases = torch.tensor(self.CBMRResults.estimator.inputs_['Coef_spline_bases'], dtype=torch.float64, device=self.device)
        all_group_foci_per_voxel = [torch.tensor(self.CBMRResults.estimator.inputs_['all_foci_per_voxel'][group], dtype=torch.float64, device=self.device) for group in self.group_names]
        all_group_foci_per_study = [torch.tensor(self.CBMRResults.estimator.inputs_['all_foci_per_study'][group], dtype=torch.float64, device=self.device) for group in self.group_names]
        all_spatial_coef = torch.tensor([self.CBMRResults.tables['Spatial_Regression_Coef'].to_numpy()[i, :].reshape((-1,1)) for i in range(self.n_groups)], dtype=torch.float64, device=self.device)
        
        all_moderator_coef = torch.tensor(self.CBMRResults.tables['Moderators_Regression_Coef'].to_numpy().T, dtype=torch.float64, device=self.device)
        moderator_coef_dim, _ = all_moderator_coef.shape
        all_group_moderators = [torch.tensor(self.CBMRResults.estimator.inputs_['all_group_moderators'][group], dtype=torch.float64, device=self.device) for group in self.group_names]
        
        if 'Overdispersion_Coef' in self.CBMRResults.tables.keys():
            all_overdispersion_coef = torch.tensor(self.CBMRResults.tables['Overdispersion_Coef'].to_numpy(), dtype=torch.float64, device=self.device)
            
        if self.CBMRResults.estimator.model == 'Poisson':
            nll = lambda all_moderator_coef: -GLMPoisson._log_likelihood_mult_group(all_spatial_coef, Coef_spline_bases, all_group_foci_per_voxel, all_group_foci_per_study, all_moderator_coef, all_group_moderators)
        elif self.CBMRResults.estimator.model == 'NB':
            nll = lambda all_moderator_coef: -GLMNB._log_likelihood_mult_group(all_overdispersion_coef, all_spatial_coef, Coef_spline_bases, all_group_foci_per_voxel, all_group_foci_per_study, all_moderator_coef, all_group_moderators)
        elif self.CBMRResults.estimator.model == 'clustered_NB':
            nll = lambda all_moderator_coef: -GLMCNB._log_likelihood_mult_group(all_overdispersion_coef, all_spatial_coef, Coef_spline_bases, all_group_foci_per_voxel, all_group_foci_per_study, all_moderator_coef, all_group_moderators)
        h = functorch.hessian(nll)(all_moderator_coef)
        h = h.view(moderator_coef_dim, moderator_coef_dim)
        
        return h.detach().cpu().numpy()

    def _contrast(self):
        Log_Spatial_Intensity_SE = self.CBMRResults.tables['Log_Spatial_Intensity_SE']
        if self.t_con_group is not False:
            con_group_count = 0
            for con_group in self.t_con_group: 
                con_group_involved_index = np.where(np.any(con_group!=0, axis=0))[0].tolist()
                con_group_involved = [self.group_names[i] for i in con_group_involved_index]
                n_con_group_involved = len(con_group_involved)
                simp_con_group = con_group[:,~np.all(con_group == 0, axis = 0)] # contrast matrix of involved groups only
                if np.all(np.count_nonzero(con_group, axis=1)==1): # GLH: homogeneity test
                    involved_log_intensity_per_voxel = list()
                    for group in con_group_involved:
                        group_foci_per_voxel = self.CBMRResults.estimator.inputs_['all_foci_per_voxel'][group]
                        group_foci_per_study = self.CBMRResults.estimator.inputs_['all_foci_per_study'][group]
                        n_voxels, n_study = group_foci_per_voxel.shape[0], group_foci_per_study.shape[0]
                        group_null_log_spatial_intensity = np.log(np.sum(group_foci_per_voxel) / (n_voxels * n_study))
                        group_log_intensity_per_voxel = np.log(self.CBMRResults.maps['Group_'+group+'_Studywise_Spatial_Intensity'])
                        group_log_intensity_per_voxel = group_log_intensity_per_voxel - group_null_log_spatial_intensity
                        involved_log_intensity_per_voxel.append(group_log_intensity_per_voxel)
                    involved_log_intensity_per_voxel = np.stack(involved_log_intensity_per_voxel, axis=0)
                else: # GLH: group-comparison
                    involved_log_intensity_per_voxel = list()
                    for group in con_group_involved:
                        group_log_intensity_per_voxel = np.log(self.CBMRResults.maps['Group_'+group+'_Studywise_Spatial_Intensity'])
                        involved_log_intensity_per_voxel.append(group_log_intensity_per_voxel)
                    involved_log_intensity_per_voxel = np.stack(involved_log_intensity_per_voxel, axis=0)
                Contrast_log_intensity = np.matmul(simp_con_group, involved_log_intensity_per_voxel) 
                m, n_brain_voxel = Contrast_log_intensity.shape
                # Correlation of involved group-wise spatial coef
                F_spatial_coef = self._Fisher_info_spatial_coef(con_group_involved_index)
                Cov_spatial_coef = np.linalg.inv(F_spatial_coef)
                spatial_coef_dim = self.CBMRResults.tables['Spatial_Regression_Coef'].to_numpy().shape[1]
                Cov_log_intensity = np.empty(shape=(0,n_brain_voxel))
                for k in range(n_con_group_involved):
                    for s in range(n_con_group_involved):
                        Cov_beta_ks = Cov_spatial_coef[k*spatial_coef_dim: (k+1)*spatial_coef_dim, s*spatial_coef_dim: (s+1)*spatial_coef_dim]
                        Cov_group_log_intensity = np.empty(shape=(1, 0))
                        for j in range(n_brain_voxel):
                            x_j = self.CBMRResults.estimator.inputs_['Coef_spline_bases'][j, :].reshape((1, spatial_coef_dim))
                            Cov_group_log_intensity_j = x_j @ Cov_beta_ks @ x_j.T
                            Cov_group_log_intensity = np.concatenate((Cov_group_log_intensity, Cov_group_log_intensity_j), axis=1)
                        Cov_log_intensity = np.concatenate((Cov_log_intensity, Cov_group_log_intensity), axis=0) # (m^2, n_voxels)
                # GLH on log_intensity (eta)
                chi_sq_spatial = np.empty(shape=(0, ))
                for j in range(n_brain_voxel):
                    Contrast_log_intensity_j = Contrast_log_intensity[:, j].reshape(m, 1)
                    V_j = Cov_log_intensity[:, j].reshape((n_con_group_involved, n_con_group_involved))
                    CV_jC = simp_con_group @ V_j @ simp_con_group.T
                    CV_jC_inv = np.linalg.inv(CV_jC)
                    chi_sq_spatial_j = Contrast_log_intensity_j.T @ CV_jC_inv @ Contrast_log_intensity_j
                    chi_sq_spatial = np.concatenate((chi_sq_spatial, chi_sq_spatial_j.reshape(1,)), axis=0)
                p_vals_spatial = 1 - scipy.stats.chi2.cdf(chi_sq_spatial, df=m)

                con_group_name = self.t_con_group_name[con_group_count]
                if len(con_group_name) == 1:
                    self.CBMRResults.maps[con_group_name[0] +'_chi_sq'] = chi_sq_spatial
                    self.CBMRResults.maps[con_group_name[0] +'_p'] = p_vals_spatial
                else:
                    self.CBMRResults.maps['spatial_coef_GLH_' + str(con_group_count) +'_chi_sq'] = chi_sq_spatial
                    self.CBMRResults.maps['spatial_coef_GLH_' + str(con_group_count) +'_p'] = p_vals_spatial
                    self.CBMRResults.metadata['spatial_coef_GLH_' + str(con_group_count)] = con_group_name
                con_group_count += 1
        
        if self.t_con_moderator is not False: 
            con_moderator_count = 0
            for con_moderator in self.t_con_moderator: 
                m_con_moderator, _ = con_moderator.shape
                moderator_coef = self.CBMRResults.tables['Moderators_Regression_Coef'].to_numpy().T
                Contrast_moderator_coef = np.matmul(con_moderator, moderator_coef) 
                F_moderator_coef = self._Fisher_info_moderator_coef()
                Cov_moderator_coef = np.linalg.inv(F_moderator_coef)
                chi_sq_moderator = Contrast_moderator_coef.T @ np.linalg.inv(con_moderator @ Cov_moderator_coef @ con_moderator.T) @ Contrast_moderator_coef
                chi_sq_moderator = chi_sq_moderator.item()
                p_vals_moderator = 1 - scipy.stats.chi2.cdf(chi_sq_moderator, df=m_con_moderator)
            
                con_moderator_name = self.t_con_moderator_name[con_moderator_count]
                if len(con_moderator_name) == 1:
                    self.CBMRResults.tables[con_moderator_name[0] +'_chi_sq'] = chi_sq_moderator
                    self.CBMRResults.tables[con_moderator_name[0] +'_p'] = p_vals_moderator
                else:
                    self.CBMRResults.tables['moderator_coef_GLH_' + str(con_moderator_count) +'_chi_sq'] = chi_sq_moderator
                    self.CBMRResults.tables['moderator_coef_GLH_' + str(con_moderator_count) +'_p'] = p_vals_moderator
                    self.CBMRResults.metadata['moderator_coef_GLH_' + str(con_moderator_count)] = con_moderator_name
                con_moderator_count += 1
            
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
    
    def _log_likelihood_single_group(beta, gamma, Coef_spline_bases, moderators, foci_per_voxel, foci_per_study, device='cpu'):
        log_mu_spatial = torch.matmul(Coef_spline_bases, beta.T)
        mu_spatial = torch.exp(log_mu_spatial)
        if gamma is not None:
            log_mu_moderators = torch.matmul(moderators, gamma.T)
            mu_moderators = torch.exp(log_mu_moderators)
        else:
            n_study, _ = foci_per_study.shape
            log_mu_moderators = torch.tensor([0]*n_study, dtype=torch.float64, device=device).reshape((-1,1))
            mu_moderators = torch.exp(log_mu_moderators)
        log_l = torch.sum(torch.mul(foci_per_voxel, log_mu_spatial)) + torch.sum(torch.mul(foci_per_study, log_mu_moderators)) \
                        - torch.sum(mu_spatial) * torch.sum(mu_moderators)

        return log_l

    def _log_likelihood_mult_group(all_spatial_coef, Coef_spline_bases,  all_foci_per_voxel, all_foci_per_study, moderator_coef=None, all_moderators=None, device='cpu'):
        n_groups = len(all_spatial_coef)
        all_log_spatial_intensity = [torch.matmul(Coef_spline_bases, all_spatial_coef[i, :, :]) for i in range(n_groups)]
        all_spatial_intensity = [torch.exp(log_spatial_intensity) for log_spatial_intensity in all_log_spatial_intensity]
        if moderator_coef is not None:
            all_log_moderator_effect = [torch.matmul(moderator, moderator_coef) for moderator in all_moderators]
            all_moderator_effect = [torch.exp(log_moderator_effect) for log_moderator_effect in all_log_moderator_effect]
        else:
            all_log_moderator_effect = [torch.tensor([0]*foci_per_study.shape[0], dtype=torch.float64, device=device).reshape((-1,1)) for foci_per_study in all_foci_per_study]
            all_moderator_effect = [torch.exp(log_moderator_effect) for log_moderator_effect in all_log_moderator_effect]
        l = 0
        for i in range(n_groups):
            l +=  torch.sum(all_foci_per_voxel[i] * all_log_spatial_intensity[i]) + torch.sum(all_foci_per_study[i] * all_log_moderator_effect[i]) - torch.sum(all_spatial_intensity[i]) * torch.sum(all_moderator_effect[i])
        return l
    
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
            group_foci_per_voxel = all_foci_per_voxel[group]
            group_foci_per_study = all_foci_per_study[group]
            if self.study_level_moderators:
                log_mu_moderators = all_log_mu_moderators[group]
                mu_moderators = torch.exp(log_mu_moderators)
            else:
                n_group_study, _ = group_foci_per_study.shape
                log_mu_moderators = torch.tensor([0]*n_group_study, device=self.device).reshape((-1,1))
                mu_moderators = torch.exp(log_mu_moderators)
            # Under the assumption that Y_ij is either 0 or 1
            # l = [Y_g]^T * log(mu^X) + [Y^t]^T * log(mu^Z) - [1^T mu_g^X]*[1^T mu_g^Z]
            group_log_l = torch.sum(torch.mul(group_foci_per_voxel, log_mu_spatial)) + torch.sum(torch.mul(group_foci_per_study, log_mu_moderators)) - torch.sum(mu_spatial) * torch.sum(mu_moderators)
            log_l += group_log_l
        
        if self.penalty:
            # Firth-type penalty 
            for group in all_foci_per_voxel.keys(): 
                beta = self.all_beta_linears[group].weight.T
                beta_dim = beta.shape[0]
                group_foci_per_voxel = all_foci_per_voxel[group]
                group_foci_per_study = all_foci_per_study[group] 
                if self.study_level_moderators:
                    gamma = self.gamma_linear.weight.T
                    group_moderators = all_moderators[group]
                    gamma, group_moderators = [gamma], [group_moderators]
                else: 
                    gamma, group_moderators = None, None
                
                all_spatial_coef = torch.stack([beta])
                all_foci_per_voxel, all_foci_per_study = torch.stack([group_foci_per_voxel]), torch.stack([group_foci_per_study])
                # a = -GLMPoisson._log_likelihood(all_spatial_coef, Coef_spline_bases, all_foci_per_voxel, all_foci_per_study, gamma, group_moderators)
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
    
    def _log_likelihood_single_group(alpha, beta, gamma, Coef_spline_bases, group_moderators, group_foci_per_voxel, group_foci_per_study, device='cpu'):
        v = 1 / alpha
        log_mu_spatial = torch.matmul(Coef_spline_bases, beta.T)
        mu_spatial = torch.exp(log_mu_spatial)
        if gamma is not None: 
            log_mu_moderators = torch.matmul(group_moderators, gamma.T)
            mu_moderators = torch.exp(log_mu_moderators)
        else:
            n_study, _ = group_foci_per_study.shape
            log_mu_moderators = torch.tensor([0]*n_study, dtype=torch.float64, device=device).reshape((-1,1))
            mu_moderators = torch.exp(log_mu_moderators)
        numerator = mu_spatial**2 * torch.sum(mu_moderators**2)
        denominator = mu_spatial**2 * torch.sum(mu_moderators)**2
        estimated_sum_alpha = alpha * numerator / denominator

        p = numerator / (v * mu_spatial * torch.sum(mu_moderators) + numerator)
        r = v * denominator / numerator

        log_l = GLMNB._three_term(group_foci_per_voxel,r, device=device) + torch.sum(r*torch.log(1-p) + group_foci_per_voxel*torch.log(p))

        return log_l
    
    def _log_likelihood_mult_group(all_overdispersion_coef, all_spatial_coef, Coef_spline_bases,  all_foci_per_voxel, all_foci_per_study, moderator_coef=None, all_moderators=None, device='cpu'):
        all_v = 1 / all_overdispersion_coef
        n_groups = len(all_foci_per_voxel)
        all_log_spatial_intensity = [torch.matmul(Coef_spline_bases, all_spatial_coef[i, :, :]) for i in range(n_groups)]
        all_spatial_intensity = [torch.exp(log_spatial_intensity) for log_spatial_intensity in all_log_spatial_intensity]
        if moderator_coef is not None:
            all_log_moderator_effect = [torch.matmul(moderator, moderator_coef) for moderator in all_moderators]
            all_moderator_effect = [torch.exp(log_moderator_effect) for log_moderator_effect in all_log_moderator_effect]
        else:
            all_log_moderator_effect = [torch.tensor([0]*foci_per_study.shape[0], dtype=torch.float64, device=device).reshape((-1,1)) for foci_per_study in all_foci_per_study]
            all_moderator_effect = [torch.exp(log_moderator_effect) for log_moderator_effect in all_log_moderator_effect]
        
        all_numerator = [all_spatial_intensity[i]**2 * torch.sum(all_moderator_effect[i]**2) for i in range(n_groups)]
        all_denominator = [all_spatial_intensity[i]**2 * torch.sum(all_moderator_effect[i])**2 for i in range(n_groups)]
        all_estimated_sum_alpha = [all_overdispersion_coef[i,:] * all_numerator[i] / all_denominator[i] for i in range(n_groups)]
        
        p = [all_numerator[i] / (all_v[i] * all_spatial_intensity[i] * torch.sum(all_moderator_effect[i]) + all_denominator[i]) for i in range(n_groups)]
        r = [all_v[i] * all_denominator[i] / all_numerator[i] for i in range(n_groups)]
        
        l = 0
        for i in range(n_groups):
            l += GLMNB._three_term(all_foci_per_voxel[i],r[i], device=device) + torch.sum(r[i]*torch.log(1-p[i]) + all_foci_per_voxel[i]*torch.log(p[i]))
    
        return l
    
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
            if self.study_level_moderators:
                log_mu_moderators = all_log_mu_moderators[group]
                mu_moderators = torch.exp(log_mu_moderators)
            else:
                n_group_study, _ = all_foci_per_study[group].shape
                log_mu_moderators = torch.tensor([0]*n_group_study, device=self.device).reshape((-1,1))
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
        self.device = device
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
    
    def _log_likelihood_single_group(alpha, beta, gamma, Coef_spline_bases, group_moderators, group_foci_per_voxel, group_foci_per_study, device='cpu'):
        v = 1 / alpha
        log_mu_spatial = torch.matmul(Coef_spline_bases, beta.T)
        mu_spatial = torch.exp(log_mu_spatial)
        if gamma is not None:
            log_mu_moderators = torch.matmul(group_moderators, gamma.T)
            mu_moderators = torch.exp(log_mu_moderators)
        else: 
            n_study, _ = group_foci_per_study.shape
            log_mu_moderators = torch.tensor([0]*n_study, dtype=torch.float64, device=device).reshape((-1,1))
            mu_moderators = torch.exp(log_mu_moderators)
        mu_sum_per_study = torch.sum(mu_spatial) * mu_moderators
        group_n_study, _ = group_foci_per_study.shape

        log_l = group_n_study * v * torch.log(v) - group_n_study * torch.lgamma(v) + torch.sum(torch.lgamma(group_foci_per_study + v)) - torch.sum((group_foci_per_study + v) * torch.log(mu_sum_per_study + v)) \
            + torch.sum(group_foci_per_voxel * log_mu_spatial) + torch.sum(group_foci_per_study * log_mu_moderators)

        return log_l

    def _log_likelihood_mult_group(all_overdispersion_coef, all_spatial_coef, Coef_spline_bases,  all_foci_per_voxel, all_foci_per_study, moderator_coef=None, all_moderators=None, device='cpu'):
        n_groups = len(all_foci_per_voxel)
        all_log_spatial_intensity = [torch.matmul(Coef_spline_bases, all_spatial_coef[i, :, :]) for i in range(n_groups)]
        all_spatial_intensity = [torch.exp(log_spatial_intensity) for log_spatial_intensity in all_log_spatial_intensity]
        if moderator_coef is not None:
            all_log_moderator_effect = [torch.matmul(moderator, moderator_coef) for moderator in all_moderators]
            all_moderator_effect = [torch.exp(log_moderator_effect) for log_moderator_effect in all_log_moderator_effect]
        else:
            all_log_moderator_effect = [torch.tensor([0]*foci_per_study.shape[0], dtype=torch.float64, device=device).reshape((-1,1)) for foci_per_study in all_foci_per_study]
            all_moderator_effect = [torch.exp(log_moderator_effect) for log_moderator_effect in all_log_moderator_effect]
        
        all_mu_sum_per_study = [torch.sum(all_spatial_intensity[i]) * all_moderator_effect[i] for i in range(n_groups)]
        l = 0
        for i in range(n_groups):
            l +=  torch.sum(all_foci_per_voxel[i] * all_log_spatial_intensity[i]) + torch.sum(all_foci_per_study[i] * all_log_moderator_effect[i]) - torch.sum(all_spatial_intensity[i]) * torch.sum(all_moderator_effect[i])
        return l

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
            group_foci_per_voxel = all_foci_per_voxel[group]
            group_foci_per_study = all_foci_per_study[group]
            if self.study_level_moderators:
                log_mu_moderators = all_log_mu_moderators[group]
                mu_moderators = torch.exp(log_mu_moderators)
            else:
                n_group_study, _ = group_foci_per_study.shape
                log_mu_moderators = torch.tensor([0]*n_group_study, device=self.device).reshape((-1,1))
                mu_moderators = torch.exp(log_mu_moderators)
            group_n_study, _ = group_foci_per_study.shape
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