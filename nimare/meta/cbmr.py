from attr import has
from numpy import spacing
from nimare.base import Estimator
from nimare.utils import get_template, get_masker, B_spline_bases
import nibabel as nib
import numpy as np
from nimare.utils import mm2vox, vox2idx
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
                    moderators_array = np.stack([dataset.annotations[moderator_name] for moderator_name in self.moderators], axis=1)
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

    def _fit(self, dataset, spline_spacing=5, model='Poisson', penalty=False, n_iter=1000, lr=1e-2, tol=1e-2, device='cpu'):
        self.model = model
        masker_voxels = self.inputs_['mask_img']._dataobj
        Coef_spline_bases = B_spline_bases(masker_voxels=masker_voxels, spacing=spline_spacing)
        self.inputs_['Coef_spline_bases'] = Coef_spline_bases

        cbmr_model = self._model_structure(model, penalty, device)
        optimisation = self._optimizer(cbmr_model, penalty, lr, tol, n_iter, device)

        return

    def _update(self, model, penalty, optimizer, Coef_spline_bases, moderators_array, n_foci_per_voxel, n_foci_per_study, gamma=0.99):
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=gamma) # learning rate decay
        scheduler.step()
        def closure():
            optimizer.zero_grad()
            loss = model(penalty, Coef_spline_bases, moderators_array, n_foci_per_voxel, n_foci_per_study)
            loss.backward()
            return loss
        loss = optimizer.step(closure)

        
        
        pass

    def _optimizer(self, model, penalty, lr, tol, n_iter, device): 
        optimizer = torch.optim.LBFGS(model.parameters(), lr)
        # load dataset info to torch.tensor
        Coef_spline_bases = torch.tensor(self.inputs_['Coef_spline_bases'], dtype=torch.float64, device=device)
        if hasattr(self, "moderators"):
            moderators_array = torch.tensor(self.inputs_['moderators_array'], dtype=torch.float64, device=device)
        n_foci_per_voxel = torch.tensor(self.inputs_['n_foci_per_voxel'], dtype=torch.float64, device=device)
        n_foci_per_study = torch.tensor(self.inputs_['n_foci_per_study'], dtype=torch.float64, device=device)
        for i in range(n_iter):
            self._update(model, penalty, optimizer, Coef_spline_bases, moderators_array, n_foci_per_voxel, n_foci_per_study)


        # while torch.abs(loss_diff) > tol: 
        #     if step <= n_iter:
        #         scheduler.step()
        #         def closure():
        #             optimizer.zero_grad()
        #             loss = model(self.X, y, Z, y_t)
        #             loss.backward()
        #             return loss
        #         loss = optimizer.step(closure)
        #         # reset L_BFGS if NAN appears
        #         if torch.any(torch.isnan(model.beta_linear.weight)):
        #             print("Reset lbfgs optimiser ......")
        #             count += 1
        #             if count > 10:
        #                 print('optimisation failed')
        #                 break
        #             model.beta_linear.weight = torch.nn.Parameter(last_state['beta_linear.weight'])
        #             if self.covariates == True:
        #                 model.gamma_linear.weight = torch.nn.Parameter(last_state['gamma_linear.weight'])
        #             if self.model == 'NB':
        #                 model.theta = torch.nn.Parameter(last_state['theta'])
        #             if self.model == 'Clustered_NB':
        #                 model.alpha = torch.nn.Parameter(last_state['alpha'])
        #             loss_diff = torch.tensor(float('inf'))
        #             optimizer = torch.optim.LBFGS(model.parameters(), lr)
        #             continue
        #         else:
        #             last_state = copy.deepcopy(model.state_dict())
        #         print("step {0}: loss {1}".format(step, loss))
        #         loss_diff = loss - prev_loss
        #         prev_loss = loss
        #         step = step + 1
        #     else:
        #         print('it did not converge \n')
        #         print('The difference of loss in the current and previous iteration is', loss_diff)
        #         exit()
        # return 


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
        log_mu_X = self.beta_linear(Coef_spline_bases) 
        mu_X = torch.exp(log_mu_X)
        # if self.covariates == True:
        #     # mu^Z = exp(Z * gamma)
        #     log_mu_Z = self.gamma_linear(Z)
        #     mu_Z = torch.exp(log_mu_Z)
        # else:
        #     log_mu_Z = torch.zeros(n_study, 1, device='cuda')
        #     mu_Z = torch.ones(n_study, 1, device='cuda')
        # # Under the assumption that Y_ij is either 0 or 1
        # # l = [Y_g]^T * log(mu^X) + [Y^t]^T * log(mu^Z) - [1^T mu_g^X]*[1^T mu_g^Z]
        # log_l = torch.sum(torch.mul(y, log_mu_X)) + torch.sum(torch.mul(y_t, log_mu_Z)) - torch.sum(mu_X) * torch.sum(mu_Z) 
        # if self.penalty == 'No':
        #     l = log_l

        return -l
