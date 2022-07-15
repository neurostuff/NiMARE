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
                        if self.moderators_scale: 
                            moderators_array /= np.var(moderators_array, axis=0)
                # Calculate IJK matrix indices for target mask
                # Mask space is assumed to be the same as the Dataset's space
                # These indices are used directly by any KernelTransformer
                xyz = dataset.coordinates[['x', 'y', 'z']].values
                ijk = mm2vox(xyz, mask_img.affine)
                if hasattr(self, "moderators"):   
                    study_id = dataset.coordinates['study_id']
                    study_index = [np.where(study_id.unique()==i)[0].item() for i in study_id]
                    self.inputs_["coordinates"]["study_index"] = study_index 
                self.inputs_["coordinates"][["i", "j", "k"]] = ijk
                foci_idx = vox2idx(ijk, mask_img._dataobj)
                self.inputs_["coordinates"]['foci_idx'] = foci_idx
                # Y & y & y_t
                n_study = np.shape(study_id.unique())[0]
                masker_voxels = np.sum(mask_img._dataobj).astype(int)
                Y = np.zeros((n_study, masker_voxels))
                
                y = np.sum(Y, axis=0)
                y_t = np.sum(Y, axis=1)



    def _fit(self, dataset, spline_spacing):
        masker_voxels = self.inputs_['mask_img']._dataobj
        X = B_spline_bases(masker_voxels=masker_voxels, spacing=spline_spacing)

        pass


    def _optimizer(self, model, y, Z, y_t, penalty, lr, tol, iter):
        # optimization 
        optimizer = torch.optim.LBFGS(model.parameters(), lr)
        prev_loss = torch.tensor(float('inf'))
        loss_diff = torch.tensor(float('inf'))
        step = 0
        count = 0
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.995)
        while torch.abs(loss_diff) > tol: 
            if step <= iter:
                scheduler.step()
                def closure():
                    optimizer.zero_grad()
                    loss = model(self.X, y, Z, y_t)
                    loss.backward()
                    return loss
                loss = optimizer.step(closure)
                # reset L_BFGS if NAN appears
                if torch.any(torch.isnan(model.beta_linear.weight)):
                    print("Reset lbfgs optimiser ......")
                    count += 1
                    if count > 10:
                        break
                    model.beta_linear.weight = torch.nn.Parameter(last_state['beta_linear.weight'])
                    if self.covariates == True:
                        model.gamma_linear.weight = torch.nn.Parameter(last_state['gamma_linear.weight'])
                    if self.model == 'NB':
                        model.theta = torch.nn.Parameter(last_state['theta'])
                    if self.model == 'Clustered_NB':
                        model.alpha = torch.nn.Parameter(last_state['alpha'])
                    loss_diff = torch.tensor(float('inf'))
                    optimizer = torch.optim.LBFGS(model.parameters(), lr)
                    continue
                else:
                    last_state = copy.deepcopy(model.state_dict())
                print("step {0}: loss {1}".format(step, loss))
                loss_diff = loss - prev_loss
                prev_loss = loss
                step = step + 1
            else:
                print('it did not converge \n')
                print('The difference of loss in the current and previous iteration is', loss_diff)
                exit()
        return 
        
    def train(self, model, penalty, covariates, iter=1500, lr=0.01, tol=1e-4):
        self.model = model
        self.penalty = penalty
        self.covariates = covariates
        # model & optimization process
        for i in range(100):
            model = self.model_structure(model=self.model, penalty=self.penalty, covariates=self.covariates)
            optimization = self._optimizer(model=model, y=self.y, Z=self.Z, y_t=self.y_t, penalty=self.penalty, lr=lr, tol=tol, iter=iter)
            # beta
            beta = model.beta_linear.weight
            beta = beta.detach().cpu().numpy().T
            print(np.all(np.isnan(beta)))
            if np.all(np.isnan(beta)): 
                print('restart the optimisation!')
                continue
            else:
