from numpy import spacing
from nimare.base import Estimator
from nimare.utils import get_template, get_masker, B_spline_bases
import nibabel as nib


class CBMREstimator(Estimator):
    def __init__(self, model="Poisson", penalty=None, spline_knots_spacing=5, mask=None, **kwargs):
        super().__init__(**kwargs)
        if mask is not None:
            mask = get_masker(mask)
        self.masker = mask

        self.model = model
        self.penalty = penalty
        self.spline_knots_spacing = spline_knots_spacing

    def _preprocess_input(self, dataset):
        masker = self.masker or dataset.masker

        mask_img = masker.mask_img or masker.labels_img
        if isinstance(mask_img, str):
            mask_img = nib.load(mask_img)
        masker_voxels = mask_img._dataobj
        design_matrix = B_spline_bases(
            masker_voxels=masker_voxels, spacing=self.spline_knots_spacing
        )

        return design_matrix

    def _fit(self, dataset):

        pass
