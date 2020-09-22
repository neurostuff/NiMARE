"""Utilities for generating data for testing"""
import os

import numpy as np
import nibabel as nib
from scipy import ndimage

from .dataset import Dataset
from .utils import get_resource_path
from .meta.utils import compute_ma, get_ale_kernel
from .transforms import vox2mm


def create_coordinate_dataset(foci, studies, sample_size_mean,
                              sample_size_variance, foci_variance=None, rng=None):
    """Create a Dataset object with simulated data"""
    if rng is None:
        rng = np.random.RandomState(seed=1939)
    # Dataset()
    sample_size_lower_limit = int(sample_size_mean - sample_size_variance)
    sample_size_upper_limit = int(sample_size_mean + sample_size_variance)
    sample_sizes = rng.randint(sample_size_lower_limit, sample_size_upper_limit, size=studies)
    foci_dict = create_foci(foci, studies, sample_sizes=sample_sizes, rng=rng)
    study_foci = [
        {a: [c[s][i] for c in foci_dict.values()] for i, a in enumerate(["x", "y", "z"])}
        for s in range(studies)
    ]

    source_dict = create_source(study_foci, sample_sizes)
    
    dataset = Dataset(source_dict)
    ground_truth_foci = list(foci_dict.keys())
    # [(-10.0, 54.0, -42.0), (52.0, -20.0, -60.0)]
    return ground_truth_foci, dataset


def create_source(foci, sample_sizes, space="MNI"):
    """create dictionary with nimads specification"""
    source = {}
    for study_idx, (sample_size, study_foci)  in enumerate(zip(sample_sizes, foci)):
        source[f'study-{study_idx}'] = {
            "contrasts": {
                "1": {
                    'coords': {
                        'space': "MNI",
                        'x': study_foci['x'],
                        'y': study_foci['y'],
                        'z': study_foci['z'],
                    },
                    'sample_sizes': [sample_size],
                }
            }
        }
        
    return source


def create_foci(foci, studies, space="MNI", rng=None, sample_sizes=None, fwhm=None):
    if rng is None:
        rng = np.random.RandomState(seed=1939)

    if space == "MNI":
        template_img = nib.load(
            os.path.join(get_resource_path(), 'templates', 'MNI152_2x2x2_brainmask.nii.gz')
        )
    else:
        raise NotImplementedError("Only coordinates for the MNI atlas has been defined")

    if (sample_sizes is None and fwhm is None) or (sample_sizes is not None and fwhm is not None):
        raise ValueError("either sample_sizes or fwhm must be set (mutually exclusive)")

    template_data = template_img.get_data()
    possible_i, possible_j, possible_k = np.nonzero(template_data)
    max_i = possible_i.max()
    max_j = possible_j.max()
    max_k = possible_k.max()
    if isinstance(foci, np.ndarray):
        foci_n = foci.shape[0]
        ground_truth_foci = foci
    elif isinstance(foci, int):
        foci_n = foci
        ground_truth_i = rng.choice(possible_i, foci_n)
        ground_truth_j = rng.choice(possible_j, foci_n)
        ground_truth_k = rng.choice(possible_k, foci_n)
        ground_truth_foci = np.vstack([ground_truth_i, ground_truth_j, ground_truth_k]).T

    if isinstance(sample_sizes, int):
        sample_sizes = [sample_sizes] * foci_n 
    elif isinstance(sample_sizes, list):
        if sample_sizes != foci_n:
            raise ValueError(("sample_sizes must be a list the same size as the"
                             " number of foci or an integer"))

    foci_dict = {}
    for ground_truth_focus, sample_size in zip(ground_truth_foci, sample_sizes):
        _, kernel = get_ale_kernel(template_img, sample_size=sample_size, fwhm=fwhm)
        template_data_dilated = tuple([s + kernel.shape[0] for s in template_data.shape])
        prob_map = compute_ma(template_data_dilated, np.atleast_2d(ground_truth_focus), kernel)
        prob_map_ijk = np.argwhere(prob_map)
        filtered_idxs = np.where((prob_map_ijk[:, 0] <= max_i) & (prob_map_ijk[:, 1] <= max_j) & (prob_map_ijk[:, 2] <= max_k))
        usable_ijk = prob_map_ijk[filtered_idxs]
        usable_prob_map = prob_map[[tuple(c) for c in usable_ijk.T]]
        usable_prob_map = usable_prob_map / usable_prob_map.sum()
        ijk_idxs = rng.choice(usable_ijk.shape[0], studies, p=usable_prob_map, replace=False)

        focus_ijks = prob_map_ijk[ijk_idxs]
        focus_xyzs = [vox2mm(ijk, template_img.affine) for ijk in focus_ijks]
        foci_dict[tuple(vox2mm(ground_truth_focus, template_img.affine))] = focus_xyzs

    return foci_dict
