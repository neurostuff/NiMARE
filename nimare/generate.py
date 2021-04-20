"""Utilities for generating data for testing."""
from itertools import zip_longest

import nilearn
import numpy as np

from .dataset import Dataset
from .io import convert_neurovault_to_dataset
from .meta.utils import compute_ale_ma, get_ale_kernel
from .transforms import transform_images
from .utils import mm2vox, vox2mm

# defaults for creating a neurovault dataset
NEUROVAULT_IDS = (8836, 8838, 8893, 8895, 8892, 8891, 8962, 8894, 8956, 8854, 9000)
CONTRAST_OF_INTEREST = {"animal": "as-Animal"}


def create_coordinate_dataset(
    foci=1,
    foci_percentage="100%",
    fwhm=10,
    sample_size=30,
    n_studies=30,
    n_noise_foci=0,
    seed=None,
    space="MNI",
):
    """Generate coordinate based dataset for meta analysis.

    Parameters
    ----------
    foci : :obj:`int` or :obj:`list`
        The number of foci to be generated per study or the
        x,y,z coordinates of the ground truth foci. (Default=1)
    foci_percentage : :obj:`float`
        Percentage of studies where the foci appear. (Default="100%")
    fwhm : :obj:`float`
        Full width at half maximum (fwhm) to define the probability
        spread of the foci. (Default=10)
    sample_size : :obj:`int` or :obj:`list`
        Either mean number of participants in each study
        or a list specifying the sample size for each
        study. If a list of two numbers and n_studies is
        not two, then the first number will represent a lower
        bound and the second number will represent an upper bound
        of a uniform sample. (Default=30)
    n_studies : :obj:`int`
        Number of studies to generate. (Default=30)
    n_noise_foci : :obj:`int`
        Number of foci considered to be noise in each study. (Default=0)
    seed : :obj:`int` or None
        Random state to reproducibly initialize random numbers.
        If seed is None, then the random state will try to be initialized
        with data from /dev/urandom (or the Windows analogue) if available
        or will initialize from the clock otherwise. (Default=None)
    space : :obj:`str`
        The template space the coordinates are reported in. (Default='MNI')

    Returns
    -------
    ground_truth_foci : :obj:`list`
        generated foci in xyz (mm) coordinates
    dataset : :class:`nimare.Dataset`
    """
    # set random state
    rng = np.random.RandomState(seed=seed)

    # check foci argument
    if not isinstance(foci, int) and not _array_like(foci):
        raise ValueError("foci must be a positive integer or array like")

    # check foci_percentage argument
    if (
        (not isinstance(foci_percentage, (float, str)))
        or (isinstance(foci_percentage, str) and foci_percentage[-1] != "%")
        or (isinstance(foci_percentage, float) and not (0.0 <= foci_percentage <= 1.0))
    ):
        raise ValueError(
            "foci_percentage must be a string (example '96%') or a float between 0 and 1"
        )

    # check sample_size argument
    if _array_like(sample_size) and len(sample_size) != n_studies and len(sample_size) != 2:
        raise ValueError("sample_size must be the same length as n_studies or list of 2 items")
    elif not _array_like(sample_size) and not isinstance(sample_size, int):
        raise ValueError("sample_size must be array like or integer")

    # check space argument
    if space != "MNI":
        raise NotImplementedError("Only coordinates for the MNI atlas has been defined")

    # process foci_percentage argument
    if isinstance(foci_percentage, str) and foci_percentage[-1] == "%":
        foci_percentage = float(foci_percentage[:-1]) / 100

    # process sample_size argument
    if isinstance(sample_size, int):
        sample_size = [sample_size] * n_studies
    elif _array_like(sample_size) and len(sample_size) == 2 and n_studies != 2:
        sample_size_lower_limit = sample_size[0]
        sample_size_upper_limit = sample_size[1]
        sample_size = rng.randint(sample_size_lower_limit, sample_size_upper_limit, size=n_studies)

    ground_truth_foci, foci_dict = _create_foci(
        foci, foci_percentage, fwhm, n_studies, n_noise_foci, rng, space
    )

    source_dict = _create_source(foci_dict, sample_size, space)
    dataset = Dataset(source_dict)

    return ground_truth_foci, dataset


def create_neurovault_dataset(
    collection_ids=NEUROVAULT_IDS,
    contrasts=CONTRAST_OF_INTEREST,
    img_dir=None,
    map_type_conversion=None,
    **dset_kwargs,
):
    """Download images from NeuroVault and use them to create a dataset.

    This function will also attempt to generate Z images for any contrasts
    for which this is possible.

    Parameters
    ----------
    collection_ids : :obj:`list` of :obj:`int` or :obj:`dict`, optional
        A list of collections on neurovault specified by their id.
        The collection ids can accessed through the neurovault API
        (i.e., https://neurovault.org/api/collections) or
        their main website (i.e., https://neurovault.org/collections).
        For example, in this URL https://neurovault.org/collections/8836/,
        `8836` is the collection id.
        collection_ids can also be a dictionary whose keys are the informative
        study name and the values are collection ids to give the collections
        human readable names in the dataset.
    contrasts : :obj:`dict`, optional
        Dictionary whose keys represent the name of the contrast in
        the dataset and whose values represent a regular expression that would
        match the names represented in NeuroVault.
        For example, under the ``Name`` column in this URL
        https://neurovault.org/collections/8836/,
        a valid contrast could be "as-Animal", which will be called "animal" in the created
        dataset if the contrasts argument is ``{'animal': "as-Animal"}``.
    img_dir : :obj:`str` or None, optional
        Base path to save all the downloaded images, by default the images
        will be saved to a temporary directory with the prefix "neurovault"
    map_type_conversion : :obj:`dict` or None, optional
        Dictionary whose keys are what you expect the `map_type` name to
        be in neurovault and the values are the name of the respective
        statistic map in a nimare dataset. Default = None.
    **dset_kwargs : keyword arguments passed to Dataset
        Keyword arguments to pass in when creating the Dataset object.
        see :obj:`nimare.dataset.Dataset` for details.

    Returns
    -------
    :obj:`nimare.dataset.Dataset`
        Dataset object containing experiment information from neurovault.
    """
    dataset = convert_neurovault_to_dataset(
        collection_ids, contrasts, img_dir, map_type_conversion, **dset_kwargs
    )
    dataset.images = transform_images(
        dataset.images, target="z", masker=dataset.masker, metadata_df=dataset.metadata
    )

    return dataset


def _create_source(foci, sample_sizes, space="MNI"):
    """Create dictionary according to nimads(ish) specification.

    Parameters
    ----------
    foci : :obj:`dict`
        A dictionary of foci in xyz (mm) coordinates whose keys represent
        different studies.
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
                    "metadata": {"sample_sizes": [sample_size]},
                }
            }
        }

    return source


def _create_foci(foci, foci_percentage, fwhm, n_studies, n_noise_foci, rng, space):
    """Generate study specific foci.

    Parameters
    ----------
    foci : :obj:`int` or :obj:`list`
        The number of foci to be generated per study or the
        x,y,z coordinates of the ground truth foci.
    foci_percentage : :obj:`float`
        Percentage of studies where the foci appear.
    fwhm : :obj:`float`
        Full width at half maximum (fwhm) to define the probability
        spread of the foci.
    n_studies : :obj:`int`
        Number of n_studies to generate.
    n_noise_foci : :obj:`int`
        Number of foci considered to be noise in each study.
    rng : :class:`numpy.random.RandomState`
        Random state to reproducibly initialize random numbers.
    space : :obj:`str`
        The template space the coordinates are reported in.

    Returns
    -------
    ground_truth_foci : :obj:`list`
        List of 3-item tuples containing x, y, z coordinates
        of the ground truth foci or an empty list if
        there are no ground_truth_foci.
    foci_dict : :obj:`dict`
        Dictionary with keys representing the study, and
        whose values represent the study specific foci.
    """
    # convert foci_percentage to float between 0 and 1
    if isinstance(foci_percentage, str) and foci_percentage[-1] == "%":
        foci_percentage = float(foci_percentage[:-1]) / 100

    if space == "MNI":
        template_img = nilearn.datasets.load_mni152_brain_mask()

    # use a template to find all "valid" coordinates
    template_data = template_img.get_fdata()
    possible_ijks = np.argwhere(template_data)

    # number of "convergent" foci each study should report
    if isinstance(foci, int):
        foci_idxs = np.unique(rng.choice(range(possible_ijks.shape[0]), foci, replace=True))
        # if there are no foci_idxs, give a dummy coordinate (0, 0, 0)
        ground_truth_foci_ijks = possible_ijks[foci_idxs] if foci_idxs.size else np.array([[]])
    elif isinstance(foci, list):
        ground_truth_foci_ijks = np.array([mm2vox(coord, template_img.affine) for coord in foci])

    # create a probability map for each peak
    kernel = get_ale_kernel(template_img, fwhm)[1]
    foci_prob_maps = {
        tuple(peak): compute_ale_ma(template_data.shape, np.atleast_2d(peak), kernel)
        for peak in ground_truth_foci_ijks
        if peak.size
    }

    # get study specific instances of each foci
    signal_studies = int(round(foci_percentage * n_studies))
    signal_ijks = {
        peak: np.argwhere(prob_map)[
            rng.choice(
                np.argwhere(prob_map).shape[0],
                size=signal_studies,
                replace=True,
                p=prob_map[np.nonzero(prob_map)] / sum(prob_map[np.nonzero(prob_map)]),
            )
        ]
        for peak, prob_map in foci_prob_maps.items()
    }

    # reshape foci coordinates to be study specific
    paired_signal_ijks = (
        np.transpose(np.array(list(signal_ijks.values())), axes=(1, 0, 2))
        if signal_ijks
        else (None,)
    )

    foci_dict = {}
    for study_signal_ijks, study in zip_longest(paired_signal_ijks, range(n_studies)):
        if study_signal_ijks is None:
            study_signal_ijks = np.array([[]])
            n_noise_foci = max(1, n_noise_foci)

        if n_noise_foci > 0:
            noise_ijks = possible_ijks[
                rng.choice(possible_ijks.shape[0], n_noise_foci, replace=True)
            ]

            # add the noise foci ijks to the existing signal ijks
            foci_ijks = (
                np.unique(np.vstack([study_signal_ijks, noise_ijks]), axis=0)
                if np.any(study_signal_ijks)
                else noise_ijks
            )
        else:
            foci_ijks = study_signal_ijks

        # transform ijk voxel coordinates to xyz mm coordinates
        foci_xyzs = [vox2mm(ijk, template_img.affine) for ijk in foci_ijks]
        foci_dict[study] = foci_xyzs

    ground_truth_foci_xyz = [
        tuple(vox2mm(ijk, template_img.affine)) for ijk in ground_truth_foci_ijks if np.any(ijk)
    ]
    return ground_truth_foci_xyz, foci_dict


def _array_like(obj):
    """Test if obj is array-like."""
    return isinstance(obj, (list, tuple, np.ndarray))
