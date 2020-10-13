"""Utilities for generating data for testing"""
import os

import numpy as np
import nibabel as nib

from .dataset import Dataset
from .utils import get_resource_path
from .meta.utils import compute_ma, get_ale_kernel
from .transforms import vox2mm, mm2vox

DEF_TINY = np.finfo(float).eps


def create_coordinate_dataset(
    n_foci,
    fwhm,
    sample_size,
    n_studies,
    foci_coords=None,
    n_noise_foci=0,
    foci_weights=None,
    seed=42,
    sample_size_interval=None,
    space="MNI",
):
    """Generate coordinate based dataset for meta analysis.

    Parameters
    ----------
    n_foci : :obj:`int` or :obj:`list`
        The number of foci to be generated per study
    fwhm : :obj:`float`
        Full width at half maximum to define the probability
        spread of the foci.
    sample_size : :obj:`int` or :obj:`list`
        Either mean number of participants in each study
        or a list specifying the sample size for each
        study.
    n_studies : :obj:`int`
        Number of n_studies to generate.
    foci_coords : :obj:`list` of three item array_like objects, optional
        Ground truth xyz coordinates of foci.
        If not specified, coordinates will be randomly selected.
        If not the same length as the number of foci (i.e., `n_foci`),
        foci will either be over-sampled or under-sampled.
    n_noise_foci : :obj:`int` or :obj:`list`, optional
        Number of foci considered to be noise in each study
        or a list of integers representing how many noise foci
        there should be in each study. (Default=0)
    foci_weights : :obj:`list`, optional
        Weighting of each peak representing the probability of
        that peak being sampled in a study.
    seed : :obj:`int`, optional
        Random state to reproducibly initialize random numbers.
        (Default=42)
    sample_size_interval : :obj:`int`, optional
        Half of the interval for the uniform distribution centered on
        sample_size, U(sample_size - interval, sample_size + interval).
        sample_size_interval must be >= 0.
    space : :obj:`str`
        The template space the coordinates are reported in. (Default='MNI')

    Returns
    -------
    ground_truth_foci : :obj:`list`
        generated foci in xyz (mm) coordinates
    dataset : :class:`nimare.Dataset`
    """
    rng = np.random.RandomState(seed=seed)
    if isinstance(sample_size, list):
        sample_sizes = sample_size
        if len(sample_sizes) != n_studies:
            raise ValueError("sample_size must be the same length as n_studies")
    else:
        if sample_size_interval is None:
            raise ValueError(
                (
                    "sample_size_interval must be specified if sample_size is not"
                    "provided for each study"
                )
            )
        sample_size_lower_limit = int(sample_size - sample_size_interval)
        # add 1 since randint upper limit is a closed interval
        sample_size_upper_limit = int(sample_size + sample_size_interval + 1)
        sample_sizes = rng.randint(
            sample_size_lower_limit, sample_size_upper_limit, size=n_studies
        )
    ground_truth_foci, foci_dict = create_foci(
        n_foci,
        fwhm,
        n_studies,
        foci_coords,
        n_noise_foci,
        foci_weights,
        seed=seed,
        space="MNI",
    )

    source_dict = create_source(foci_dict, sample_sizes)
    dataset = Dataset(source_dict)

    return ground_truth_foci, dataset


def create_source(foci, sample_sizes, space="MNI"):
    """Create dictionary according to nimads(ish) specification

    Parameters
    ----------
    foci : :obj:`dict`
        A dictionary of foci in xyz (mm) coordinates whose keys represent
        different n_studies.
    sample_sizes : :obj:`list`
        The sample size for each study
    space : :obj:`str`
        The template space the coordinates are reported in. (Default='MNI')

    Returns
    -------
    source : :obj:`dict`
        study information in nimads format
    """
    source = {}
    for sample_size, (study, study_foci) in zip(sample_sizes, foci.items()):
        source[f"study-{study}"] = {
            "contrasts": {
                "1": {
                    "coords": {
                        "space": space,
                        "x": [c[0] for c in study_foci],
                        "y": [c[1] for c in study_foci],
                        "z": [c[2] for c in study_foci],
                    },
                    "metadata": {
                        "sample_sizes": [sample_size],
                    },
                }
            }
        }

    return source


def create_foci(
    n_foci,
    fwhm,
    n_studies,
    foci_coords=None,
    n_noise_foci=0,
    foci_weights=None,
    seed=42,
    space="MNI",
):
    """Generate study specific foci.

    Parameters
    ----------
    n_foci : :obj:`int` or :obj:`list`
        The number of foci to be generated per study.
    fwhm : :obj:`float` or :obj:`list`
        Full width at half maximum (fwhm) to define the probability
        spread of the foci or list of fwhms identifying the fwhm
        for each foci.
    n_studies : :obj:`int`
        Number of n_studies to generate.
    foci_coords : :obj:`list` of three item array_like objects, optional
        Ground truth xyz coordinates of foci.
        If not specified, coordinates will be randomly selected.
        If not the same length as the number of foci (i.e., `n_foci`),
        foci will either be over-sampled or under-sampled.
    n_noise_foci : :obj:`int` or :obj:`list`, optional
        Number of foci considered to be noise in each study
        or a list of integers representing how many noise foci
        there should be in each study. (Default=0)
    foci_weights : :obj:`list`, optional
        Weighting of each peak representing the probability of
        that peak being sampled in a study.
    seed : :obj:`int`, optional
        Random state to reproducibly initialize random numbers.
        (Default=42)
    space : :obj:`str`
        The template space the coordinates are reported in. (Default='MNI')

    Returns
    -------
    ground_truth_foci : :obj:`list`
        List of 3-item tuples containing x, y, z coordinates
        of the ground truth foci.
    foci_dict : :obj:`dict`
        Dictionary with keys representing the ground truth foci, and
        whose values represent the study specific foci.
    """
    rng = np.random.RandomState(seed=seed)

    if space == "MNI":
        template_img = nib.load(
            os.path.join(get_resource_path(), "templates", "MNI152_2x2x2_brainmask.nii.gz")
        )
    else:
        raise NotImplementedError("Only coordinates for the MNI atlas has been defined")

    # use a template to find all "valid" coordinates
    template_data = template_img.get_fdata()
    possible_ijks = np.argwhere(template_data)
    # number of "convergent" foci each study should report
    if isinstance(n_foci, int):
        n_focis = [n_foci] * n_studies
    else:
        if len(n_foci) != n_studies:
            raise ValueError("n_foci must be the same length as n_studies")
        n_focis = n_foci
    # foci_coords are to be generated randomly
    if foci_coords is None:
        foci_idxs = rng.choice(range(possible_ijks.shape[0]), max(n_focis), replace=False)
        # if there are no foci_idxs, give a dummy coordinate (0, 0, 0)
        ground_truth_foci_ijk = (
            possible_ijks[foci_idxs] if foci_idxs.size else np.array([(0, 0, 0)])
        )
        foci_coords_num = ground_truth_foci_ijk.shape[0]
    # foci_coords were specified by the caller
    elif isinstance(foci_coords, list):
        ground_truth_foci_ijk = [mm2vox(coord, template_img.affine) for coord in foci_coords]
        foci_coords_num = len(ground_truth_foci_ijk)
    else:
        raise ValueError("foci_coords must be a list or None")

    if not foci_weights:
        foci_weights = [1] * foci_coords_num
    if not isinstance(foci_weights, list):
        raise ValueError("foci_weights must be a list")
    elif len(foci_weights) != foci_coords_num:
        raise ValueError("foci_weights must be the same length as foci")

    if isinstance(fwhm, float):
        fwhms = [fwhm] * foci_coords_num
    elif isinstance(fwhm, list):
        fwhms = [float(f) for f in fwhm]
        if len(fwhms) != foci_coords_num:
            raise ValueError(
                ("fwhm must be a list the same size as the number of foci or a float")
            )
    else:
        try:
            fwhms = [float(fwhm)] * foci_coords_num
        except ValueError:
            raise ValueError("fwhm must be a float or a list")

    foci_dict = {}
    # generate study specific foci for each ground truth focus

    kernels = [get_ale_kernel(template_img, fwhm)[1] for fwhm in fwhms]
    weighted_prob_map = sum(
        [
            compute_ma(template_data.shape, np.atleast_2d(ground_truth_focus), kernel) * weight
            for kernel, ground_truth_focus, weight in zip(
                kernels, ground_truth_foci_ijk, foci_weights
            )
        ]
    )
    # handle case when sum is 0 by adding a small amount
    weighted_prob_map = weighted_prob_map / np.sum(weighted_prob_map) + DEF_TINY
    weighted_prob_map_ijks = np.argwhere(weighted_prob_map)
    weighted_prob_vector = weighted_prob_map[np.nonzero(weighted_prob_map)]

    if isinstance(n_noise_foci, int):
        n_noise_foci_list = [n_noise_foci] * n_studies
    else:
        n_noise_foci_list = n_noise_foci

    for f_num, f_noise, study in zip(n_focis, n_noise_foci_list, range(n_studies)):
        ijk_idxs = rng.choice(
            weighted_prob_map_ijks.shape[0], f_num, p=weighted_prob_vector, replace=False
        )
        foci_ijks = weighted_prob_map_ijks[ijk_idxs]

        if f_noise > 0:
            weighted_noise_map_ijks = np.delete(weighted_prob_map_ijks, ijk_idxs, axis=0)
            noise_ijk_idxs = rng.choice(weighted_noise_map_ijks.shape[0], f_noise, replace=False)
            # add the noise foci ijks to the existing foci ijks
            foci_ijks = np.vstack([foci_ijks, weighted_noise_map_ijks[noise_ijk_idxs]])

        # transform ijk voxel coordinates to xyz mm coordinates
        foci_xyzs = [vox2mm(ijk, template_img.affine) for ijk in foci_ijks]
        foci_dict[study] = foci_xyzs

    ground_truth_foci_xyz = [
        tuple(vox2mm(ijk, template_img.affine)) for ijk in ground_truth_foci_ijk
    ]
    return ground_truth_foci_xyz, foci_dict
