from attr import has
from numpy import spacing
from nimare.base import Estimator
from nimare.utils import get_template, get_masker, B_spline_bases
import nibabel as nib
import numpy as np
from nimare.utils import mm2vox, vox2idx, intensity2voxel
import torch

class CBMREstimator(Estimator):
    _required_inputs = {"coordinates": ("coordinates", None)}

    def __init__(self, groups=False, moderators=None, moderators_center=True, moderators_scale=True, mask=None, **kwargs):
        super().__init__(**kwargs)
        if mask is not None:
            mask = get_masker(mask)
        self.masker = mask

        self.groups = groups
        self.moderators = moderators 
        self.moderators_center = moderators_center # either boolean or a list of strings
        self.moderators_scale = moderators_scale

    def _preprocess_input(self, dataset):
        masker = self.masker or dataset.masker

        mask_img = masker.mask_img or masker.labels_img
        if isinstance(mask_img, str):
            mask_img = nib.load(mask_img)
        
        ma_values = self._collect_inputs(dataset, drop_invalid=True)
        self.inputs_['mask_img'] = mask_img

        for name, (type_, _) in self._required_inputs.items():
            if type_ == "coordinates":
                if hasattr(self, "groups"):
                    ## to do: raise an error if group column doesn't exist in dataset.annotations
                    group_names = dataset.annotations['group_id'].unique()
                    gb = dataset.annotations.groupby('group_id')
                    multiple_groups = [gb.get_group(x)['study_id'] for x in gb.groups]
                if hasattr(self, "moderators"):
                    study_id_moderators = dataset.annotations.set_index('study_id').index
                    study_id_coordinates = dataset.coordinates.set_index('study_id').index
                    moderators_with_coordinates = dataset.annotations[study_id_moderators.isin(study_id_coordinates)] # moderators dataframe where foci exist in selected studies
                    moderators_array = np.stack([moderators_with_coordinates[moderator_name] for moderator_name in self.moderators], axis=1)
                    moderators_array = moderators_array.astype(np.float64)
                    if isinstance(self.moderators_center, bool):
                        ## to do: if moderators_center & moderators_array is a list of moderators names, only operate on the chosen moderators
                        if self.moderators_center: 
                            moderators_array -= np.mean(moderators_array, axis=0)
                    if isinstance(self.moderators_scale, bool):
                        if self.moderators_scale: 
                            moderators_array /= np.var(moderators_array, axis=0)
                    self.inputs_["moderators_array"] = moderators_array
                # Calculate IJK matrix indices for target mask
                # Mask space is assumed to be the same as the Dataset's space
                # These indices are used directly by any KernelTransformer
                xyz = dataset.coordinates[['x', 'y', 'z']].values
                ijk = mm2vox(xyz, mask_img.affine)
                
                study_id = dataset.coordinates['study_id']
                study_index = [np.where(study_id.unique()==i)[0].item() for i in study_id]
                self.inputs_["coordinates"]["study_index"] = study_index 
                self.inputs_["coordinates"][["i", "j", "k"]] = ijk
                foci_idx = vox2idx(ijk, mask_img._dataobj)
                self.inputs_["coordinates"]['foci_idx'] = foci_idx
                
                n_study = np.shape(study_id.unique())[0]
                masker_voxels = np.sum(mask_img._dataobj).astype(int)
                n_foci_per_voxel = np.zeros((masker_voxels, 1))
                n_foci_per_voxel[foci_idx, :] += 1
                self.inputs_['n_foci_per_voxel'] = n_foci_per_voxel
                n_foci_per_study = np.zeros((n_study, 1))
                n_foci_per_study[study_index, :] += 1
                self.inputs_['n_foci_per_study'] = n_foci_per_study

    def _model_structure(self, model, penalty, device):
        beta_dim = self.inputs_['Coef_spline_bases'].shape[1] # regression coef of spatial effect
        if hasattr(self, "moderators"):
            gamma_dim = self.inputs_["moderators_array"].shape[1]
            study_level_covariates = True
        else:
            gamma_dim = None
            study_level_covariates = False
        if model == 'Poisson':
            cbmr_model = GLMPoisson(beta_dim=beta_dim, gamma_dim=gamma_dim, study_level_covariates=study_level_covariates, penalty=penalty)
        if 'cuda' in device:
            cbmr_model = cbmr_model.cuda()
        
        return cbmr_model

    def _update(self, model, penalty, optimizer, Coef_spline_bases, moderators_array, n_foci_per_voxel, n_foci_per_study, prev_loss, gamma=0.99):
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=gamma) # learning rate decay
        scheduler.step()
        def closure():
            optimizer.zero_grad()
            loss = model(penalty, Coef_spline_bases, moderators_array, n_foci_per_voxel, n_foci_per_study)
            loss.backward()
            return loss
        loss = optimizer.step(closure)
                
        return loss

    def _optimizer(self, model, penalty, lr, tol, n_iter, device): 
        optimizer = torch.optim.LBFGS(model.parameters(), lr)
        # load dataset info to torch.tensor
        Coef_spline_bases = torch.tensor(self.inputs_['Coef_spline_bases'], dtype=torch.float64, device=device)
        if hasattr(self, "moderators"):
            moderators_array = torch.tensor(self.inputs_['moderators_array'], dtype=torch.float64, device=device)
        n_foci_per_voxel = torch.tensor(self.inputs_['n_foci_per_voxel'], dtype=torch.float64, device=device)
        n_foci_per_study = torch.tensor(self.inputs_['n_foci_per_study'], dtype=torch.float64, device=device)
        prev_loss = torch.tensor(float('inf')) # initialization loss difference
        for i in range(n_iter):
            loss = self._update(model, penalty, optimizer, Coef_spline_bases, moderators_array, n_foci_per_voxel, n_foci_per_study, prev_loss)
            loss_diff = loss - prev_loss
            if torch.abs(loss_diff) < tol:
                break
            prev_loss = loss
        
        return

    def _fit(self, dataset, spline_spacing=5, model='Poisson', penalty=False, n_iter=1000, lr=1e-2, tol=1e-2, device='cpu'):
        self.model = model
        masker_voxels = self.inputs_['mask_img']._dataobj
        Coef_spline_bases = B_spline_bases(masker_voxels=masker_voxels, spacing=spline_spacing)
        self.inputs_['Coef_spline_bases'] = Coef_spline_bases

        cbmr_model = self._model_structure(model, penalty, device)
        optimisation = self._optimizer(cbmr_model, penalty, lr, tol, n_iter, device)

        # beta: regression coef of spatial effect
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
    def __init__(self, beta_dim=None, gamma_dim=None, study_level_covariates=False, penalty='No'):
        super().__init__()
        self.study_level_covariates = study_level_covariates
        # initialization for beta
        self.beta_linear = torch.nn.Linear(beta_dim, 1, bias=False).double()
        torch.nn.init.uniform_(self.beta_linear.weight, a=-0.01, b=0.01)
        # gamma 
        if self.study_level_covariates:
            self.gamma_linear = torch.nn.Linear(gamma_dim, 1, bias=False).double()
            torch.nn.init.uniform_(self.gamma_linear.weight, a=-0.01, b=0.01)
    
    def forward(self, penalty, Coef_spline_bases, moderators_array, n_foci_per_voxel, n_foci_per_study):
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
