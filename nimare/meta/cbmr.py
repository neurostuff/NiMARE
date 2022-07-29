import string
from attr import has
from numpy import spacing
from nimare.base import Estimator
from nimare.utils import get_template, get_masker, B_spline_bases
import nibabel as nib
import numpy as np
from nimare.utils import mm2vox, vox2idx, intensity2voxel
import torch
import logging

LGR = logging.getLogger(__name__)
class CBMREstimator(Estimator):
    _required_inputs = {"coordinates": ("coordinates", None)}

    def __init__(self, multiple_groups=False, moderators=None, moderators_center=True, moderators_scale=True, mask=None, 
                spline_spacing=5, model='Poisson', penalty=False, n_iter=1000, lr=1e-2, tol=1e-2, device='cpu', **kwargs):
        super().__init__(**kwargs)
        if mask is not None:
            mask = get_masker(mask)
        self.masker = mask

        self.multiple_groups = multiple_groups
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
                study_id_annotations = dataset.annotations.set_index('study_id').index
                study_id_coordinates = dataset.coordinates.set_index('study_id').index
                # remove study_id without any coordinates
                valid_study_bool = study_id_annotations.isin(study_id_coordinates)
                dataset_annotations = dataset.annotations[valid_study_bool]
                all_group_study_id = dict()
                if self.multiple_groups:
                    if 'group_id' not in dataset_annotations.columns: 
                        raise ValueError("group_id must exist in the dataset in group-wise CBMR")
                    else:
                        group_names = list(dataset_annotations['group_id'].unique())
                        if len(group_names) == 1:
                            raise ValueError('Only a single group exists in the dataset')
                        for group_name in group_names:
                            group_study_id_bool = dataset_annotations['group_id'] == group_name
                            group_study_id = dataset_annotations.loc[group_study_id_bool]['study_id']
                            all_group_study_id[group_name] = group_study_id.unique().tolist()
                else:
                    all_group_study_id['single_group'] = dataset_annotations['study_id'].unique().tolist()
                self.inputs_['all_group_study_id'] = all_group_study_id
                # collect studywise moderators if specficed
                if hasattr(self, "moderators"):
                    all_group_moderators = dict()
                    for group_name in all_group_study_id.keys():
                        df_group = dataset_annotations.loc[dataset_annotations['study_id'].isin(all_group_study_id[group_name])] 
                        group_moderators = np.stack([df_group[moderator_name] for moderator_name in self.moderators], axis=1)
                        group_moderators = group_moderators.astype(np.float64)
                        # standardize mean
                        if isinstance(self.moderators_center, bool):
                            if self.moderators_center: 
                                group_moderators -= np.mean(group_moderators, axis=0)
                        elif isinstance(self.moderators_center, str):
                            index_moderators_center = self.moderators.index(self.moderators_center)
                            group_moderators[:,index_moderators_center] -= np.mean(group_moderators[:, index_moderators_center], axis=0)
                        elif isinstance(self.moderators_center, list):
                            index_moderators_center = [self.moderators.index(moderator_name) for moderator_name in self.moderators_center]
                            for i in index_moderators_center:
                                group_moderators[:,i] -= np.mean(group_moderators[:, i], axis=0)
                        # standardize var
                        if isinstance(self.moderators_scale, bool):
                            if self.moderators_scale: 
                                group_moderators /= np.std(group_moderators, axis=0)
                        elif isinstance(self.moderators_scale, str):
                            index_moderators_scale = self.moderators.index(self.moderators_scale)
                            group_moderators[:,index_moderators_scale] /= np.std(group_moderators[:, index_moderators_scale], axis=0)
                        elif isinstance(self.moderators_scale, list):
                            index_moderators_scale = [self.moderators.index(moderator_name) for moderator_name in self.moderators_scale]
                            for i in index_moderators_scale:
                                group_moderators[:,i] /= np.std(group_moderators[:, i], axis=0)
                        all_group_moderators[group_name] = group_moderators
                    self.inputs_["all_group_moderators"] = all_group_moderators
                # Calculate IJK matrix indices for target mask
                # Mask space is assumed to be the same as the Dataset's space
                # These indices are used directly by any KernelTransformer
                all_foci_per_voxel, all_foci_per_study = dict(), dict()
                for group_name in all_group_study_id.keys():
                    group_study_id = all_group_study_id[group_name]
                    group_coordinates = dataset.coordinates.loc[dataset.coordinates['study_id'].isin(group_study_id)] 

                    group_xyz = group_coordinates[['x', 'y', 'z']].values
                    group_ijk = mm2vox(group_xyz, mask_img.affine)
                    group_foci_idx = vox2idx(group_ijk, mask_img._dataobj)
                    
                    n_group_study = len(group_study_id)
                    masker_voxels = np.sum(mask_img._dataobj).astype(int)
                    group_foci_per_voxel = np.zeros((masker_voxels, 1))
                    group_foci_per_voxel[group_foci_idx, :] += 1
                    group_foci_per_study = np.array([(group_coordinates['study_id']==i).sum() for i in group_study_id])
                    group_foci_per_study = group_foci_per_study.reshape((n_group_study, 1))

                    all_foci_per_voxel[group_name] = group_foci_per_voxel
                    all_foci_per_study[group_name] = group_foci_per_study
                
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
        self.n_groups = len(self.inputs_["all_group_study_id"])
        if model == 'Poisson':
            cbmr_model = GLMPoisson(beta_dim=beta_dim, gamma_dim=gamma_dim, n_groups=self.n_groups, study_level_moderators=study_level_moderators, penalty=penalty)
        if 'cuda' in device:
            cbmr_model = cbmr_model.cuda()
        
        return cbmr_model

    def _update(self, model, optimizer, Coef_spline_bases, moderators_array, n_foci_per_voxel, n_foci_per_study, prev_loss, gamma=0.99):
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=gamma) # learning rate decay
        scheduler.step()
        
        self.iter += 1
        def closure():
            optimizer.zero_grad()
            loss = model(Coef_spline_bases, moderators_array, n_foci_per_voxel, n_foci_per_study)
            loss.backward()
            return loss
        loss = optimizer.step(closure)
                
        return loss

    def _optimizer(self, model, lr, tol, n_iter, device): 
        optimizer = torch.optim.LBFGS(model.parameters(), lr)
        # load dataset info to torch.tensor
        Coef_spline_bases = torch.tensor(self.inputs_['Coef_spline_bases'], dtype=torch.float64, device=device)
        if hasattr(self, "moderators"):
            for group_name in self.inputs_['all_group_study_id'].keys():
                moderators_array = torch.tensor(self.inputs_['all_group_moderators'][group_name], dtype=torch.float64, device=device)
        n_foci_per_voxel = torch.tensor(self.inputs_['n_foci_per_voxel'], dtype=torch.float64, device=device)
        n_foci_per_study = torch.tensor(self.inputs_['n_foci_per_study'], dtype=torch.float64, device=device)
        
        if self.iter == 0:
            prev_loss = torch.tensor(float('inf')) # initialization loss difference

        for i in range(n_iter):
            loss = self._update(model, optimizer, Coef_spline_bases, moderators_array, n_foci_per_voxel, n_foci_per_study, prev_loss)
            loss_diff = loss - prev_loss
            LGR.debug(f"Iter {self.iter:04d}: log-likelihood {loss:.4f}")
            if torch.abs(loss_diff) < tol:
                break
            prev_loss = loss
        
        return

    def _fit(self, dataset):
        masker_voxels = self.inputs_['mask_img']._dataobj
        Coef_spline_bases = B_spline_bases(masker_voxels=masker_voxels, spacing=self.spline_spacing)
        self.inputs_['Coef_spline_bases'] = Coef_spline_bases

        cbmr_model = self._model_structure(self.model, self.penalty, self.device)
        optimisation = self._optimizer(cbmr_model, self.lr, self.tol, self.n_iter, self.device)

        # beta: regression coef of spatial effec        
        self._beta = cbmr_model.beta_linear.weight
        self._beta = self._beta.detach().numpy().T
        
        studywise_spatial_intensity = np.exp(np.matmul(Coef_spline_bases, self._beta))
        studywise_spatial_intensity = intensity2voxel(studywise_spatial_intensity, self.inputs_['mask_img']._dataobj)

        if hasattr(self, "moderators"):
            self._gamma = cbmr_model.gamma_linear.weight
            self._gamma = self._gamma.detach().numpy().T

            moderator_results = np.exp(np.matmul(self.inputs_["moderators_array"], self._gamma))


        return

    

class GLMPoisson(torch.nn.Module):
    def __init__(self, beta_dim=None, gamma_dim=None, n_groups=None, study_level_moderators=False, penalty='No'):
        super().__init__()
        self.n_groups = n_groups
        self.study_level_moderators = study_level_moderators
        # initialization for beta
        beta_linear_weights = list()
        for i in range(self.n_groups):
            beta_linear_i = torch.nn.Linear(beta_dim, 1, bias=False).double()
            torch.nn.init.uniform_(beta_linear_i.weight, a=-0.01, b=0.01)
            beta_linear_weights.append(beta_linear_i.weight)
        beta_linear_weights = torch.stack(beta_linear_weights)
        self.beta_linear_weights = torch.nn.Parameter(beta_linear_weights, requires_grad=True)
        # gamma 
        if self.study_level_moderators:
            self.gamma_linear = torch.nn.Linear(gamma_dim, 1, bias=False).double()
            torch.nn.init.uniform_(self.gamma_linear.weight, a=-0.01, b=0.01)
    
    def forward(self, Coef_spline_bases, moderators_array, n_foci_per_voxel, n_foci_per_study):
        # spatial effect: mu^X = exp(X * beta)
        log_mu_spatial = self.beta_linear(Coef_spline_bases) 
        mu_spatial = torch.exp(log_mu_spatial)
        if torch.is_tensor(moderators_array):
            # mu^Z = exp(Z * gamma)
            log_mu_moderators = self.gamma_linear(moderators_array)
            mu_moderators = torch.exp(log_mu_moderators)
        # Under the assumption that Y_ij is either 0 or 1
        # l = [Y_g]^T * log(mu^X) + [Y^t]^T * log(mu^Z) - [1^T mu_g^X]*[1^T mu_g^Z]
        log_l = torch.sum(torch.mul(n_foci_per_voxel, log_mu_spatial)) + torch.sum(torch.mul(n_foci_per_study, log_mu_moderators)) - torch.sum(mu_spatial) * torch.sum(mu_moderators) 

        return -log_l
