"""Utilities for generating data for testing."""

from itertools import zip_longest

import numpy as np
import sparse

from nimare.dataset import Dataset, _quiet_dataset_deprecation
from nimare.io import convert_neurovault_to_dataset
from nimare.meta.utils import compute_ale_ma, get_ale_kernel
from nimare.studyset import normalize_collection
from nimare.transforms import ImageTransformer
from nimare.utils import (
    DEFAULT_FLOAT_DTYPE,
    _mask_img_to_bool,
    get_template,
    mm2vox,
    vox2mm,
)

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

    .. warning::
        :class:`~nimare.dataset.Dataset` output is deprecated and will be removed in NiMARE
        1.0.0. Prefer :func:`~nimare.generate.create_coordinate_studyset`.

    .. versionadded:: 0.0.4

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
    dataset : :class:`~nimare.dataset.Dataset`
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


def create_coordinate_studyset(
    foci=1,
    foci_percentage="100%",
    fwhm=10,
    sample_size=30,
    n_studies=30,
    n_noise_foci=0,
    seed=None,
    space="MNI",
):
    """Generate a coordinate-based Studyset for meta-analysis.

    This is the Studyset-native companion to :func:`create_coordinate_dataset`
    and accepts the same arguments.

    Returns
    -------
    ground_truth_foci : :obj:`list`
        Generated foci in xyz (mm) coordinates.
    studyset : :class:`~nimare.nimads.Studyset`
    """
    with _quiet_dataset_deprecation():
        ground_truth_foci, dataset = create_coordinate_dataset(
            foci=foci,
            foci_percentage=foci_percentage,
            fwhm=fwhm,
            sample_size=sample_size,
            n_studies=n_studies,
            n_noise_foci=n_noise_foci,
            seed=seed,
            space=space,
        )
        return ground_truth_foci, normalize_collection(dataset)


def create_neurovault_dataset(
    collection_ids=NEUROVAULT_IDS,
    contrasts=CONTRAST_OF_INTEREST,
    img_dir=None,
    map_type_conversion=None,
    **dset_kwargs,
):
    """Download images from NeuroVault and use them to create a dataset.

    .. versionadded:: 0.0.8

    .. warning::
        :class:`~nimare.dataset.Dataset` output is deprecated and will be removed in NiMARE
        1.0.0. Prefer :func:`~nimare.generate.create_neurovault_studyset`.

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
        see :obj:`~nimare.dataset.Dataset` for details.

    Returns
    -------
    :obj:`~nimare.dataset.Dataset`
        Dataset object containing experiment information from neurovault.
    """
    # The image transformer works on -- and returns -- a studyset, so convert
    # back on the way out: this function promises a Dataset.
    return _neurovault_studyset(
        collection_ids, contrasts, img_dir, map_type_conversion, **dset_kwargs
    ).to_dataset()


def _neurovault_studyset(collection_ids, contrasts, img_dir, map_type_conversion, **dset_kwargs):
    """Download NeuroVault images and return them as a studyset with z maps."""
    with _quiet_dataset_deprecation():
        dataset = convert_neurovault_to_dataset(
            collection_ids, contrasts, img_dir, map_type_conversion, **dset_kwargs
        )
        return ImageTransformer(target="z").transform(dataset)


def create_neurovault_studyset(
    collection_ids=NEUROVAULT_IDS,
    contrasts=CONTRAST_OF_INTEREST,
    img_dir=None,
    map_type_conversion=None,
    **dset_kwargs,
):
    """Download images from NeuroVault and use them to create a Studyset.

    This is the Studyset-native companion to :func:`create_neurovault_dataset`
    and accepts the same arguments.

    Returns
    -------
    :obj:`~nimare.nimads.Studyset`
        Studyset object containing experiment information from NeuroVault.
    """
    return _neurovault_studyset(
        collection_ids, contrasts, img_dir, map_type_conversion, **dset_kwargs
    )


def _create_source(foci, sample_sizes, space="MNI"):
    """Create dictionary according to nimads(ish) specification.

    .. versionadded:: 0.0.4

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


#: Half-width, in mm, of the cube noise foci are drawn from, and the extent of the simulated
#: field when a caller asks for one without asking for noise. Named because the field simulator
#: takes it as its own extent, so the two must not drift apart.
DEFAULT_NOISE_EXTENT = 60.0


def _simulate_reported_peaks(
    ground_truth_foci,
    effect_sizes,
    n_subjects,
    threshold,
    smoothness_fwhm,
    blob_fwhm,
    field_zooms,
    field_extent,
    design,
    rng,
):
    """Simulate a study's statistic field and report the local maxima that clear its threshold.

    This is what makes peak-height inflation appear. Drawing a value at the ground-truth
    location and thresholding it, as the point simulator does, produces a reported statistic
    that is an unbiased estimate of the effect there -- so there is nothing for a peak-height
    correction to correct, and no simulator built that way can validate one.

    Here a smooth Gaussian noise field is added to the signal, and what gets reported is the
    position and height of a *local maximum* that cleared the threshold. Those maxima are
    selected for being large, and sit where the noise happened to help, so the reported height
    overstates the effect at that location and the reported position is displaced from the
    truth. Both fall out of the simulation rather than being imposed.

    Each peak carries the true effect at the voxel it was found in, so the inflation is a
    measurable quantity rather than something to be assumed.
    """
    from scipy.ndimage import gaussian_filter, maximum_filter

    zooms = np.full(3, float(field_zooms))
    half = int(np.ceil(field_extent / zooms[0]))
    shape = tuple(np.full(3, 2 * half + 1, dtype=int))
    origin = -zooms * half

    # Unit-variance smooth Gaussian noise: the field a null study would have.
    sigma = smoothness_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))) / zooms
    noise = gaussian_filter(rng.normal(size=shape), sigma)
    spread = noise.std()
    if spread <= 0:
        return []
    noise /= spread

    # Signal: a blob at each ground-truth focus, on the effect-size scale.
    grid = np.stack(np.indices(shape), axis=-1) * zooms + origin
    signal = np.zeros(shape, dtype=float)
    blob_sigma = blob_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    for focus, effect in zip(ground_truth_foci, effect_sizes):
        squared = ((grid - np.asarray(focus, dtype=float)) ** 2).sum(axis=-1)
        signal += float(effect) * np.exp(-squared / (2.0 * blob_sigma**2))

    scale = np.sqrt(n_subjects) if design == "one-sample" else np.sqrt(n_subjects / 4.0)
    # ``signal * scale`` is the noncentrality of the *t* statistic, so the field is built on
    # the t scale and then transformed, exactly as the point-based path does. Reporting
    # ``signal * scale + noise`` directly as a Z would label a t-scale statistic as a z: the
    # two agree closely near the threshold and diverge hard above it, because the t's heavier
    # tails mean a given tail probability sits at a much larger quantile. An estimator that
    # inverts a reported Z through the t -- which is what a paper's Z came from -- then
    # recovers a wildly inflated effect size, 8.8 against a true 1.6 at g = 1.6, and about
    # 13% too high even at g = 0.8.
    from nimare.transforms import t_to_z

    dof = n_subjects - 1 if design == "one-sample" else n_subjects - 2
    observed = t_to_z(signal * scale + noise, dof)

    # Local maxima of |observed| that clear the threshold, which is what a paper tabulates.
    magnitude = np.abs(observed)
    is_peak = (magnitude == maximum_filter(magnitude, size=3)) & (magnitude >= threshold)
    peaks = np.argwhere(is_peak)

    reported = []
    for index in peaks:
        position = tuple(index)
        reported.append(
            {
                "coordinates": [float(c) for c in grid[position]],
                "z": float(observed[position]),
                # The effect actually present where the peak was found, on the g scale.
                "true_g": float(signal[position]),
            }
        )
    return reported


def create_effect_size_coordinate_studyset(
    ground_truth_foci,
    effect_sizes=0.5,
    n_studies=20,
    sample_size=(20, 40),
    tau=0.0,
    prevalence=1.0,
    threshold_z=3.2905267314919255,
    spatial_sd=6.0,
    n_noise_foci=0,
    noise_extent=DEFAULT_NOISE_EXTENT,
    design="one-sample",
    seed=None,
    space="MNI",
    simulate_field=False,
    smoothness_fwhm=10.0,
    blob_fwhm=10.0,
    field_zooms=4.0,
):
    """Simulate a studyset whose coordinates carry reported z statistics.

    .. versionadded:: 0.13.0

    Unlike :func:`create_coordinate_dataset`, which only places foci, this simulates the whole
    reporting process that coordinate-based *effect-size* estimation has to invert: a true
    effect size at each ground-truth location, a study-level draw around it, a sampling draw
    around that, and finally a **within-study threshold** that decides whether the peak is
    reported at all. Studies whose local effect fails to clear the threshold contribute no
    focus there, which is exactly the censoring that makes the reported peaks a biased sample
    of the field.

    Parameters
    ----------
    ground_truth_foci : :obj:`list` of :obj:`tuple`
        xyz (mm) coordinates of the true effects.
    effect_sizes : :obj:`float` or :obj:`list`, default=0.5
        True population Hedges' g at each ground-truth focus. A scalar applies to all of them.
    n_studies : :obj:`int`, default=20
        Number of studies to simulate.
    sample_size : :obj:`int` or :obj:`tuple`, default=(20, 40)
        Per-study sample size, or the inclusive range to draw it from.
    tau : :obj:`float`, default=0.0
        Between-study standard deviation of the true effect at a focus.
    prevalence : :obj:`float`, default=1.0
        Probability that any given study has a non-null effect at a given focus. Below 1.0 the
        studies are a genuine mixture: some have an effect of the stated size, the rest have
        exactly zero. This is the situation a plain censored model cannot represent -- it has
        to explain a study's silence as a small common effect rather than as no effect -- and
        is what ``CBES(selection_model="zero-inflated")`` is built to recover.
    simulate_field : :obj:`bool`, default=False
        Simulate each study's whole statistic field and report the local maxima that clear its
        threshold, instead of drawing a value at each ground-truth location.

        This is the difference between a simulator that can validate a peak-height correction
        and one that cannot. The default draws a value *at* the focus, so the reported statistic
        is an unbiased estimate of the effect there and the true inflation is exactly 1 -- there
        is nothing for such a correction to recover. With a field, what gets reported is a local
        maximum selected for being large, sitting where the noise happened to help, so its
        height overstates the effect at its location and its position is displaced from the
        truth. Both emerge from the simulation rather than being imposed, and ``spatial_sd`` is
        then unused.

        Each reported point carries the true effect at the voxel it was found in, under the
        ``TRUEG`` value kind, so the inflation is measurable rather than assumed.
    smoothness_fwhm : :obj:`float`, default=10.0
        FWHM, in mm, of the simulated noise field. Only used when ``simulate_field`` is set.
    blob_fwhm : :obj:`float`, default=10.0
        FWHM, in mm, of the signal blob at each ground-truth focus. Only used when
        ``simulate_field`` is set.
    field_zooms : :obj:`float`, default=4.0
        Voxel size, in mm, of the simulated field. Only used when ``simulate_field`` is set.
    threshold_z : :obj:`float` or sequence of :obj:`float`, default=3.29
        Two-tailed reporting threshold on the z scale (p < .001 by default). A study reports a
        peak only where its observed statistic clears this. A sequence is drawn from at random,
        one threshold per study, which simulates a literature search over papers that did not
        agree on a threshold; each study then records its own under the ``reporting_threshold``
        metadata field, so estimators can be tested with and without knowing it.
    spatial_sd : :obj:`float`, default=6.0
        Standard deviation, in mm, of the localization error on a reported peak.
    n_noise_foci : :obj:`int`, default=0
        Number of null foci per study. Their locations are uniform in a cube of side
        ``2 * noise_extent`` and their statistics are drawn from the exponential
        peak-overshoot approximation for the height of a suprathreshold local maximum, so
        noise peaks look like real reported peaks rather than like implausibly large ones.
    noise_extent : :obj:`float`, default=60.0
        Half-width, in mm, of the cube noise foci are drawn from.
    design : {"one-sample", "two-sample"}, default="one-sample"
        Design to simulate. Only affects the statistic/effect-size conversion.
    seed : :obj:`int` or None, optional
        Random seed.
    space : :obj:`str`, default="MNI"
        Coordinate space label recorded on each point.

    Returns
    -------
    :obj:`~nimare.studyset.Studyset`
        Studyset whose points carry a ``Z`` value and whose analyses carry ``sample_sizes``.

    Examples
    --------
    >>> studyset = create_effect_size_coordinate_studyset(
    ...     [(0, 0, 0)], effect_sizes=0.8, n_studies=10, seed=1
    ... )
    """
    from nimare.studyset import Studyset
    from nimare.transforms import t_to_z

    rng = np.random.default_rng(seed)

    ground_truth_foci = np.atleast_2d(np.asarray(ground_truth_foci, dtype=float))
    effect_sizes = np.broadcast_to(
        np.asarray(effect_sizes, dtype=float), (len(ground_truth_foci),)
    )

    if isinstance(sample_size, (int, np.integer)):
        sample_sizes = np.full(n_studies, int(sample_size))
    else:
        low, high = sample_size
        sample_sizes = rng.integers(int(low), int(high) + 1, size=n_studies)

    if np.ndim(threshold_z) == 0:
        thresholds = np.full(n_studies, float(threshold_z))
    else:
        thresholds = rng.choice(np.asarray(threshold_z, dtype=float), size=n_studies)

    studies = []
    for i_study, (n_subjects, threshold) in enumerate(zip(sample_sizes, thresholds)):
        dof = n_subjects - 1 if design == "one-sample" else n_subjects - 2
        scale = (
            np.sqrt(1.0 / n_subjects)
            if design == "one-sample"
            else np.sqrt(4.0 / n_subjects)  # equal groups: sqrt(1/n1 + 1/n2)
        )

        points = []
        if simulate_field:
            present = [
                effect if prevalence >= 1.0 or rng.random() < prevalence else 0.0
                for effect in effect_sizes
            ]
            study_effects = [
                rng.normal(effect, tau) if tau and effect else effect for effect in present
            ]
            for peak in _simulate_reported_peaks(
                ground_truth_foci,
                study_effects,
                n_subjects,
                threshold,
                smoothness_fwhm,
                blob_fwhm,
                field_zooms,
                noise_extent if n_noise_foci or noise_extent else DEFAULT_NOISE_EXTENT,
                design,
                rng,
            ):
                points.append(
                    {
                        "space": space,
                        "coordinates": peak["coordinates"],
                        "values": [
                            {"kind": "Z", "value": peak["z"]},
                            # The effect where the peak was found, so the inflation a
                            # peak-height correction targets is measurable rather than assumed.
                            {"kind": "TRUEG", "value": peak["true_g"]},
                        ],
                    }
                )
            studies.append(
                {
                    "id": f"study-{i_study}",
                    "name": f"study-{i_study}",
                    "metadata": {
                        "sample_sizes": [int(n_subjects)],
                        "reporting_threshold": float(threshold),
                    },
                    "analyses": [
                        {
                            "id": f"study-{i_study}-1",
                            "name": "1",
                            "metadata": {
                                "sample_sizes": [int(n_subjects)],
                                "reporting_threshold": float(threshold),
                            },
                            "points": points,
                        }
                    ],
                }
            )
            continue

        for focus, true_g in zip(ground_truth_foci, effect_sizes):
            if prevalence < 1.0 and rng.random() >= prevalence:
                continue  # this study simply has no effect here
            study_effect = rng.normal(true_g, tau) if tau else true_g
            sampling_sd = np.sqrt(scale**2 + study_effect**2 / (2.0 * n_subjects))
            observed_d = rng.normal(study_effect, sampling_sd)
            observed_z = t_to_z(np.array([observed_d / scale]), dof)[0]
            if np.abs(observed_z) < threshold:
                continue
            reported = focus + rng.normal(0, spatial_sd, size=3)
            points.append(
                {
                    "space": space,
                    "coordinates": [float(c) for c in reported],
                    "values": [{"kind": "Z", "value": float(observed_z)}],
                }
            )

        for _ in range(n_noise_foci):
            # Height of a null suprathreshold local maximum: P(Z > z | Z > u) ~= exp(-u(z - u)).
            overshoot = rng.exponential(1.0 / threshold)
            noise_z = (threshold + overshoot) * rng.choice([-1.0, 1.0])
            points.append(
                {
                    "space": space,
                    "coordinates": [
                        float(c) for c in rng.uniform(-noise_extent, noise_extent, size=3)
                    ],
                    "values": [{"kind": "Z", "value": float(noise_z)}],
                }
            )

        studies.append(
            {
                "id": f"study-{i_study}",
                "name": f"study-{i_study}",
                "metadata": {
                    "sample_sizes": [int(n_subjects)],
                    "reporting_threshold": float(threshold),
                },
                "analyses": [
                    {
                        "id": f"study-{i_study}-1",
                        "name": "1",
                        "metadata": {
                            "sample_sizes": [int(n_subjects)],
                            "reporting_threshold": float(threshold),
                        },
                        "points": points,
                    }
                ],
            }
        )

    return Studyset({"id": "simulated", "name": "simulated", "studies": studies})


def _create_foci(foci, foci_percentage, fwhm, n_studies, n_noise_foci, rng, space):
    """Generate study specific foci.

    .. versionadded:: 0.0.4

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
        template_img = get_template(space="mni152_2mm", mask="brain")

    # use a template to find all "valid" coordinates
    template_data = template_img.get_fdata(dtype=DEFAULT_FLOAT_DTYPE)
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
    template_mask = _mask_img_to_bool(template_img)

    def _peak_probability_map(peak):
        peak_ma_map, _, _ = compute_ale_ma(template_img, np.atleast_2d(peak), kernel=kernel)
        prob_map = np.zeros(template_mask.size, dtype=DEFAULT_FLOAT_DTYPE)
        prob_map[template_mask.reshape(-1)] = peak_ma_map.toarray().ravel()
        return sparse.COO.from_numpy(prob_map.reshape(template_data.shape))

    foci_prob_maps = {
        tuple(peak): _peak_probability_map(peak) for peak in ground_truth_foci_ijks if peak.size
    }

    # get study specific instances of each foci
    signal_studies = int(round(foci_percentage * n_studies))
    signal_ijks = {
        peak: sparse.argwhere(prob_map)[
            rng.choice(
                sparse.argwhere(prob_map).shape[0],
                size=signal_studies,
                replace=True,
                p=(prob_map[prob_map.nonzero()] / sum(prob_map[prob_map.nonzero()])).todense(),
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
