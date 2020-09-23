"""Utilities for generating data for testing"""
import os

import numpy as np
import nibabel as nib

from .dataset import Dataset
from .utils import get_resource_path
from .meta.utils import compute_ma, get_ale_kernel
from .transforms import vox2mm


def create_coordinate_dataset(foci, fwhm, sample_size_mean,
                              sample_size_variance, studies, rng=None):
    """Generate coordinate based dataset for meta analysis.

    Parameters
    ----------
    foci : :obj:`int` or :obj:`list` of three item array_like objects
        Either the number of foci to be generated or a list of foci in ijk
        coordinates.
    fwhm : :obj:`float`
        Full width at half maximum to define the probability
        spread of the foci.
    sample_size_mean : :obj:`int`
        Mean number of participants in each study.
    sample_size_variance : :obj:`int`
        Variance of the number of participants in each study.
        (must be >=0).
    studies : :obj:`int`
        Number of studies to generate.
    rng : :class:`numpy.random.RandomState` (Optional)
        Random state to reproducibly initialize random numbers.

    Returns
    -------
    ground_truth_foci : :obj:`list`
        generated foci in xyz (mm) coordinates
    dataset : :class:`nimare.Dataset`
    """
    if rng is None:
        rng = np.random.RandomState(seed=1939)
    sample_size_lower_limit = int(sample_size_mean - sample_size_variance)
    # add 1 since randint upper limit is a closed interval
    sample_size_upper_limit = int(sample_size_mean + sample_size_variance + 1)
    sample_sizes = rng.randint(sample_size_lower_limit, sample_size_upper_limit, size=studies)
    foci_dict = create_foci(foci, studies, fwhm=fwhm, rng=rng)
    # re-arrange foci_dict into a list of dictionaries which represent individual studies
    study_foci = [
        {a: [c[s][i] for c in foci_dict.values()] for i, a in enumerate(["x", "y", "z"])}
        for s in range(studies)
    ]

    source_dict = create_source(study_foci, sample_sizes)
    dataset = Dataset(source_dict)
    ground_truth_foci = list(foci_dict.keys())

    return ground_truth_foci, dataset


def create_source(foci, sample_sizes, space="MNI"):
    """Create dictionary according to nimads(ish) specification

    Parameters
    ----------
    foci : :obj:`list` of three item array_like objects
        A list of foci in xyz (mm) coordinates
    sample_sizes : :obj:`list`
        The sample size for each study
    space : :obj:`str` (Default="MNI")
        The template space the coordinates are reported in

    Returns
    -------
    source : :obj:`dict`
        study information in nimads format
    """
    source = {}
    for study_idx, (sample_size, study_foci) in enumerate(zip(sample_sizes, foci)):
        source[f"study-{study_idx}"] = {
            "contrasts": {
                "1": {
                    "coords": {
                        "space": space,
                        "x": study_foci["x"],
                        "y": study_foci["y"],
                        "z": study_foci["z"],
                    },
                    "metadata": {
                        "sample_sizes": [sample_size],
                    }
                }
            }
        }

    return source


def create_foci(foci, studies, fwhm, rng=None, space="MNI"):
    """Generate study specific foci.

    Parameters
    ----------
    foci : :obj:`int` or :obj:`list` of three item array_like objects
        Either the number of foci to be generated or a list of foci in ijk
        coordinates.
    studies : :obj:`int`
        Number of studies to generate.
    fwhm : :obj:`float`
        Full width at half maximum to define the probability
        spread of the foci.
    rng : :class:`numpy.random.RandomState` (Optional)
        Random state to reproducibly initialize random numbers.
    space : :obj:`str` (Default="MNI")
        The template space the coordinates are reported in.

    Returns
    -------
    foci_dict : :obj:`dict`
        Dictionary with keys representing the ground truth foci, and
        whose values represent the study specific foci.
    """
    if rng is None:
        rng = np.random.RandomState(seed=1939)

    if space == "MNI":
        template_img = nib.load(
            os.path.join(get_resource_path(), "templates", "MNI152_2x2x2_brainmask.nii.gz")
        )
    else:
        raise NotImplementedError("Only coordinates for the MNI atlas has been defined")

    # use a template to find all "valid" coordinates
    template_data = template_img.get_data()
    possible_i, possible_j, possible_k = np.nonzero(template_data)
    max_i = possible_i.max()
    max_j = possible_j.max()
    max_k = possible_k.max()
    # foci were specified by the caller
    if isinstance(foci, np.ndarray):
        foci_n = foci.shape[0]
        ground_truth_foci = foci
    # foci are to be generated randomly
    elif isinstance(foci, int):
        foci_n = foci
        ground_truth_i = rng.choice(possible_i, foci_n, replace=False)
        ground_truth_j = rng.choice(possible_j, foci_n, replace=False)
        ground_truth_k = rng.choice(possible_k, foci_n, replace=False)
        ground_truth_foci = np.vstack([ground_truth_i, ground_truth_j, ground_truth_k]).T

    if isinstance(fwhm, float):
        fwhms = [fwhm] * foci_n
    elif isinstance(fwhm, list):
        fwhms = fwhm
        if len(fwhms) != foci_n:
            raise ValueError(("fwhm must be a list the same size as the"
                             " number of foci or a float"))
    else:
        try:
            fwhms = [float(fwhm)] * foci_n
        except TypeError:
            raise TypeError("fwhm must be a float or a list")

    foci_dict = {}
    # generate study specific foci for each ground truth focus
    for ground_truth_focus, fwhm in zip(ground_truth_foci, fwhms):
        _, kernel = get_ale_kernel(template_img, fwhm=fwhm)
        # dilate template so coordinates can be generated near an edge
        template_data_dilated = tuple([s + kernel.shape[0] for s in template_data.shape])
        # create the probability map to select study specific foci
        prob_map = compute_ma(template_data_dilated, np.atleast_2d(ground_truth_focus), kernel)
        # extract all viable coordinates from prob_map
        # and filter them based on the boundaries of the brain
        prob_map_ijk = np.argwhere(prob_map)
        filtered_idxs = np.where(((prob_map_ijk[:, 0] <= max_i)
                                 & (prob_map_ijk[:, 1] <= max_j)
                                 & (prob_map_ijk[:, 2] <= max_k)))
        usable_ijk = prob_map_ijk[filtered_idxs]
        usable_prob_map = prob_map[[tuple(c) for c in usable_ijk.T]]
        # normalize the probability map so it sums to 1
        usable_prob_map = usable_prob_map / usable_prob_map.sum()
        # select the foci for the number of studies specified
        ijk_idxs = rng.choice(usable_ijk.shape[0], studies, p=usable_prob_map, replace=False)
        focus_ijks = prob_map_ijk[ijk_idxs]
        # transform ijk voxel coordinates to xyz mm coordinates
        focus_xyzs = [vox2mm(ijk, template_img.affine) for ijk in focus_ijks]
        foci_dict[tuple(vox2mm(ground_truth_focus, template_img.affine))] = focus_xyzs

    return foci_dict
