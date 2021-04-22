"""Miscellaneous spatial and statistical transforms."""

import copy
import logging
import os.path as op

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.reporting import get_clusters_table
from scipy import stats

from . import references
from .base import Transformer
from .due import due
from .utils import dict_to_coordinates, dict_to_df, get_masker

LGR = logging.getLogger(__name__)


def transform_images(images_df, target, masker, metadata_df=None, out_dir=None):
    """Generate images of a given type from other image types and write out to files.

    Parameters
    ----------
    images_df : :class:`pandas.DataFrame`
        DataFrame with paths to images for studies in Dataset.
    target : {'z', 'p', 'beta', 'varcope'}
        Target data type.
    masker : :class:`nilearn.input_data.NiftiMasker` or similar
        Masker used to define orientation and resolution of images.
        Specific voxels defined in mask will not be used, and a new masker
        with _all_ voxels in acquisition matrix selected will be created.
    metadata_df : :class:`pandas.DataFrame` or :obj:`None`, optional
        DataFrame with metadata. Rows in this DataFrame must match those in
        ``images_df``, including the ``'id'`` column.
    out_dir : :obj:`str` or :obj:`None`, optional
        Path to output directory. If None, use folder containing first image
        for each study in ``images_df``.

    Returns
    -------
    images_df : :class:`pandas.DataFrame`
        DataFrame with paths to new images added.
    """
    images_df = images_df.copy()

    valid_targets = ["z", "p", "beta", "varcope"]
    if target not in valid_targets:
        raise ValueError("Target type must be one of: {}".format(", ".join(valid_targets)))
    mask_img = masker.mask_img
    new_mask = np.ones(mask_img.shape, int)
    new_mask = nib.Nifti1Image(new_mask, mask_img.affine, header=mask_img.header)
    new_masker = get_masker(new_mask)
    res = masker.mask_img.header.get_zooms()
    res = "x".join([str(r) for r in res])
    if target not in images_df.columns:
        target_ids = images_df["id"].values
    else:
        target_ids = images_df.loc[images_df[target].isnull(), "id"]

    for id_ in target_ids:
        row = images_df.loc[images_df["id"] == id_].iloc[0]

        # Determine output filename, if file can be generated
        if out_dir is None:
            options = [r for r in row.values if isinstance(r, str) and op.isfile(r)]
            id_out_dir = op.dirname(options[0])
        else:
            id_out_dir = out_dir
        new_file = op.join(
            id_out_dir, "{id_}_{res}_{target}.nii.gz".format(id_=id_, res=res, target=target)
        )

        # Grab columns with actual values
        available_data = row[~row.isnull()].to_dict()
        if metadata_df is not None:
            metadata_row = metadata_df.loc[metadata_df["id"] == id_].iloc[0]
            metadata = metadata_row[~metadata_row.isnull()].to_dict()
            for k, v in metadata.items():
                if k not in available_data.keys():
                    available_data[k] = v

        # Get converted data
        img = resolve_transforms(target, available_data, new_masker)
        if img is not None:
            img.to_filename(new_file)
            images_df.loc[images_df["id"] == id_, target] = new_file
        else:
            images_df.loc[images_df["id"] == id_, target] = None
    return images_df


def resolve_transforms(target, available_data, masker):
    """Determine and apply the appropriate transforms to a target image type from available data.

    Parameters
    ----------
    target : {'z', 'p', 't', 'beta', 'varcope'}
        Target image type.
    available_data : dict
        Dictionary mapping data types to their values. Images in the dictionary
        are paths to files.
    masker : nilearn Masker
        Masker used to convert images to arrays and back. Preferably, this mask
        should cover the full acquisition matrix (rather than an ROI), given
        that the calculated images will be saved and used for the full Dataset.

    Returns
    -------
    img_like or None
        Image object with the desired data type, if it can be generated.
        Otherwise, None.
    """
    if target in available_data.keys():
        LGR.warning('Target "{}" already available.'.format(target))
        return available_data[target]

    if target == "z":
        if ("t" in available_data.keys()) and ("sample_sizes" in available_data.keys()):
            dof = sample_sizes_to_dof(available_data["sample_sizes"])
            t = masker.transform(available_data["t"])
            z = t_to_z(t, dof)
        elif "p" in available_data.keys():
            p = masker.transform(available_data["p"])
            z = p_to_z(p)
        else:
            return None
        z = masker.inverse_transform(z.squeeze())
        return z
    elif target == "t":
        # will return none given no transform/target exists
        temp = resolve_transforms("z", available_data, masker)
        if temp is not None:
            available_data["z"] = temp

        if ("z" in available_data.keys()) and ("sample_sizes" in available_data.keys()):
            dof = sample_sizes_to_dof(available_data["sample_sizes"])
            z = masker.transform(available_data["z"])
            t = z_to_t(z, dof)
            t = masker.inverse_transform(t.squeeze())
            return t
        else:
            return None
    elif target == "beta":
        if "t" not in available_data.keys():
            # will return none given no transform/target exists
            temp = resolve_transforms("t", available_data, masker)
            if temp is not None:
                available_data["t"] = temp

        if "varcope" not in available_data.keys():
            temp = resolve_transforms("varcope", available_data, masker)
            if temp is not None:
                available_data["varcope"] = temp

        if ("t" in available_data.keys()) and ("varcope" in available_data.keys()):
            t = masker.transform(available_data["t"])
            varcope = masker.transform(available_data["varcope"])
            beta = t_and_varcope_to_beta(t, varcope)
            beta = masker.inverse_transform(beta.squeeze())
            return beta
        else:
            return None
    elif target == "varcope":
        if "se" in available_data.keys():
            se = masker.transform(available_data["se"])
            varcope = se_to_varcope(se)
        elif ("samplevar_dataset" in available_data.keys()) and (
            "sample_sizes" in available_data.keys()
        ):
            sample_size = sample_sizes_to_sample_size(available_data["sample_sizes"])
            samplevar_dataset = masker.transform(available_data["samplevar_dataset"])
            varcope = samplevar_dataset_to_varcope(samplevar_dataset, sample_size)
        elif ("sd" in available_data.keys()) and ("sample_sizes" in available_data.keys()):
            sample_size = sample_sizes_to_sample_size(available_data["sample_sizes"])
            sd = masker.transform(available_data["sd"])
            varcope = sd_to_varcope(sd, sample_size)
            varcope = masker.inverse_transform(varcope)
        elif ("t" in available_data.keys()) and ("beta" in available_data.keys()):
            t = masker.transform(available_data["t"])
            beta = masker.transform(available_data["beta"])
            varcope = t_and_beta_to_varcope(t, beta)
        else:
            return None
        varcope = masker.inverse_transform(varcope.squeeze())
        return varcope
    elif target == "p":
        if ("t" in available_data.keys()) and ("sample_sizes" in available_data.keys()):
            dof = sample_sizes_to_dof(available_data["sample_sizes"])
            t = masker.transform(available_data["t"])
            z = t_to_z(t, dof)
            p = z_to_p(z)
        elif "z" in available_data.keys():
            z = masker.transform(available_data["z"])
            p = z_to_p(z)
        else:
            return None
        p = masker.inverse_transform(p.squeeze())
        return p
    else:
        return None


class ImagesToCoordinates(Transformer):
    """Transformer from images to coordinates.

    Parameters
    ----------
    merge_strategy : {'fill', 'replace', 'demolish'}, optional
        strategy on how to incorporate the generated coordinates
        with possible pre-existing coordinates.
        The three different strategies are 'fill', 'replace',
        and 'demolish'. Default='fill'.
        - 'fill': only add coordinates to study contrasts that
          do not have coordinates. If a study contrast has both
          image and coordinate data, the original coordinate data
          will be kept.
        - 'replace': replace existing coordinates with coordinates
          generated by this function. If a study contrast only has
          coordinate data and no images or if the statistical
          threshold is too high for nimare to detect any peaks
          the original coordinates will be kept.
        - 'demolish': only keep generated coordinates and discard
          any study contrasts with coordinate data, but no images.
    cluster_threshold : :obj:`int` or `None`, optional
        Cluster size threshold, in voxels. Default=None.
    remove_subpeaks : :obj:`bool`, optional
        If True, removes subpeaks from the cluster results. Default=False.
    two_sided : :obj:`bool`, optional
        Whether to employ two-sided thresholding or to evaluate positive values
        only. Default=False.
    min_distance : :obj:`float`, optional
        Minimum distance between subpeaks in mm. Default=8mm.
    z_threshold : :obj:`float`
        Cluster forming z-scale threshold. Default=3.1.

    Notes
    -----
    The raw Z and/or P maps are not corrected for multiple comparisons,
    uncorrected z-values and/or p-values are used for thresholding.
    """

    def __init__(
        self,
        merge_strategy="fill",
        cluster_threshold=None,
        remove_subpeaks=False,
        two_sided=False,
        min_distance=8.0,
        z_threshold=3.1,
    ):
        self.merge_strategy = merge_strategy
        self.cluster_threshold = cluster_threshold
        self.remove_subpeaks = remove_subpeaks
        self.min_distance = min_distance
        self.two_sided = two_sided
        self.z_threshold = z_threshold

    def transform(self, dataset):
        """Create coordinate peaks from statistical images.

        Parameters
        ----------
        dataset : :obj:`nimare.dataset.Dataset`
            Dataset with z maps and/or p maps
            that can be converted to coordinates.

        Returns
        -------
        dataset : :obj:`nimare.dataset.Dataset`
            Dataset with coordinates generated from
            images and metadata indicating origin
            of coordinates ('original' or 'nimare').
        """
        # relevant variables from dataset
        space = dataset.space
        masker = dataset.masker
        images_df = dataset.images
        metadata = dataset.metadata.copy()

        # conform space specification
        if "mni" in space.lower() or "ale" in space.lower():
            coordinate_space = "MNI"
        elif "tal" in space.lower():
            coordinate_space = "TAL"
        else:
            coordinate_space = None

        coordinates_dict = {}
        for _, row in images_df.iterrows():

            if row["id"] in list(dataset.coordinates["id"]) and self.merge_strategy == "fill":
                continue

            if row.get("z"):
                clusters = get_clusters_table(
                    nib.funcs.squeeze_image(nib.load(row.get("z"))),
                    self.z_threshold,
                    self.cluster_threshold,
                    self.two_sided,
                    self.min_distance,
                )
            elif row.get("p"):
                LGR.info(
                    (
                        f"No Z map for {row['id']}, using p map "
                        "(p-values will be treated as positive z-values)"
                    )
                )
                if self.two_sided:
                    LGR.warning(f"Cannot use two_sided threshold using a p map for {row['id']}")
                p_threshold = 1 - z_to_p(self.z_threshold)
                nimg = nib.funcs.squeeze_image(nib.load(row.get("p")))
                inv_nimg = nib.Nifti1Image(1 - nimg.get_fdata(), nimg.affine, nimg.header)
                clusters = get_clusters_table(
                    inv_nimg,
                    p_threshold,
                    self.cluster_threshold,
                    self.min_distance,
                )
                # Peak stat p-values are reported as 1 - p in get_clusters_table
                clusters["Peak Stat"] = p_to_z(1 - clusters["Peak Stat"])
            else:
                LGR.warning(f"No Z or p map for {row['id']}, skipping...")
                continue

            # skip entry if no clusters are found
            if clusters.empty:
                LGR.warning(
                    f"No clusters were found for {row['id']} at a threshold of {self.z_threshold}"
                )
                continue

            if self.remove_subpeaks:
                # subpeaks are identified as 1a, 1b, etc
                # while peaks are kept as 1, 2, 3, etc,
                # so removing all non-int rows will
                # keep main peaks while removing subpeaks
                clusters = clusters[clusters["Cluster ID"].apply(lambda x: isinstance(x, int))]

            coordinates_dict[row["study_id"]] = {
                "contrasts": {
                    row["contrast_id"]: {
                        "coords": {
                            "space": coordinate_space,
                            "x": list(clusters["X"]),
                            "y": list(clusters["Y"]),
                            "z": list(clusters["Z"]),
                            "z_stat": list(clusters["Peak Stat"]),
                        },
                        "metadata": {"coordinate_source": "nimare"},
                    }
                }
            }

        # only the generated coordinates ('demolish')
        coordinates_df = dict_to_coordinates(coordinates_dict, masker, space)
        meta_df = dict_to_df(
            pd.DataFrame(dataset._ids),
            coordinates_dict,
            "metadata",
        )

        if "coordinate_source" in meta_df.columns:
            metadata["coordinate_source"] = meta_df["coordinate_source"]
        else:
            # nimare did not overwrite any coordinates
            metadata["coordinate_source"] = ["original"] * metadata.shape[0]

        if self.merge_strategy != "demolish":
            original_idxs = ~dataset.coordinates["id"].isin(coordinates_df["id"])
            old_coordinates_df = dataset.coordinates[original_idxs]
            coordinates_df = coordinates_df.append(old_coordinates_df, ignore_index=True)

            # specify original coordinates
            original_ids = set(old_coordinates_df["id"])
            metadata[metadata["id"].isin(original_ids)]["coordinate_source"] = "original"

        # ensure z_stat is treated as float
        if "z_stat" in coordinates_df.columns:
            coordinates_df["z_stat"] = coordinates_df["z_stat"].astype(float)

        new_dataset = copy.deepcopy(dataset)
        new_dataset.coordinates = coordinates_df
        new_dataset.metadata = metadata

        return new_dataset


def sample_sizes_to_dof(sample_sizes):
    """Calculate degrees of freedom from a list of sample sizes using a simple heuristic.

    Parameters
    ----------
    sample_sizes : array_like
        A list of sample sizes for different groups in the study.

    Returns
    -------
    dof : int
        An estimate of degrees of freedom. Number of participants minus number
        of groups.
    """
    dof = np.sum(sample_sizes) - len(sample_sizes)
    return dof


def sample_sizes_to_sample_size(sample_sizes):
    """Calculate appropriate sample size from a list of sample sizes using a simple heuristic.

    Parameters
    ----------
    sample_sizes : array_like
        A list of sample sizes for different groups in the study.

    Returns
    -------
    sample_size : int
        Total (sum) sample size.
    """
    sample_size = np.sum(sample_sizes)
    return sample_size


def sd_to_varcope(sd, sample_size):
    """Convert standard deviation to sampling variance.

    Parameters
    ----------
    sd : array_like
        Standard deviation of the sample
    sample_size : int
        Sample size

    Returns
    -------
    varcope : array_like
        Sampling variance of the parameter
    """
    se = sd / np.sqrt(sample_size)
    varcope = se_to_varcope(se)
    return varcope


def se_to_varcope(se):
    """Convert standard error values to sampling variance.

    Parameters
    ----------
    se : array_like
        Standard error of the sample parameter

    Returns
    -------
    varcope : array_like
        Sampling variance of the parameter

    Notes
    -----
    Sampling variance is standard error squared.
    """
    varcope = se ** 2
    return varcope


def samplevar_dataset_to_varcope(samplevar_dataset, sample_size):
    """Convert "sample variance of the dataset" to "sampling variance".

    Parameters
    ----------
    samplevar_dataset : array_like
        Sample variance of the dataset (i.e., variance of the individual observations in a single
        sample). Can be calculated with ``np.var``.
    sample_size : int
        Sample size

    Returns
    -------
    varcope : array_like
        Sampling variance of the parameter (i.e., variance of sampling distribution for the
        parameter).

    Notes
    -----
    Sampling variance is sample variance divided by sample size.
    """
    varcope = samplevar_dataset / sample_size
    return varcope


def t_and_varcope_to_beta(t, varcope):
    """Convert t-statistic to parameter estimate using sampling variance.

    Parameters
    ----------
    t : array_like
        T-statistics of the parameter
    varcope : array_like
        Sampling variance of the parameter

    Returns
    -------
    beta : array_like
        Parameter estimates
    """
    beta = t * np.sqrt(varcope)
    return beta


def t_and_beta_to_varcope(t, beta):
    """Convert t-statistic to sampling variance using parameter estimate.

    Parameters
    ----------
    t : array_like
        T-statistics of the parameter
    beta : array_like
        Parameter estimates

    Returns
    -------
    varcope : array_like
        Sampling variance of the parameter
    """
    varcope = (beta / t) ** 2
    return varcope


def z_to_p(z, tail="two"):
    """Convert z-values to p-values.

    Parameters
    ----------
    z : array_like
        Z-statistics
    tail : {'one', 'two'}, optional
        Whether p-values come from one-tailed or two-tailed test. Default is
        'two'.

    Returns
    -------
    p : array_like
        P-values
    """
    z = np.array(z)
    if tail == "two":
        p = stats.norm.sf(abs(z)) * 2
    elif tail == "one":
        p = stats.norm.sf(abs(z))
    else:
        raise ValueError('Argument "tail" must be one of ["one", "two"]')

    if p.shape == ():
        p = p[()]
    return p


def p_to_z(p, tail="two"):
    """Convert p-values to (unsigned) z-values.

    Parameters
    ----------
    p : array_like
        P-values
    tail : {'one', 'two'}, optional
        Whether p-values come from one-tailed or two-tailed test. Default is
        'two'.

    Returns
    -------
    z : array_like
        Z-statistics (unsigned)
    """
    p = np.array(p)
    if tail == "two":
        z = stats.norm.isf(p / 2)
    elif tail == "one":
        z = stats.norm.isf(p)
        z = np.array(z)
        z[z < 0] = 0
    else:
        raise ValueError('Argument "tail" must be one of ["one", "two"]')

    if z.shape == ():
        z = z[()]
    return z


@due.dcite(references.T2Z_TRANSFORM, description="Introduces T-to-Z transform.")
@due.dcite(references.T2Z_IMPLEMENTATION, description="Python implementation of T-to-Z transform.")
def t_to_z(t_values, dof):
    """Convert t-statistics to z-statistics.

    An implementation of [1]_ from Vanessa Sochat's TtoZ package [2]_.

    Parameters
    ----------
    t_values : array_like
        T-statistics
    dof : int
        Degrees of freedom

    Returns
    -------
    z_values : array_like
        Z-statistics

    References
    ----------
    .. [1] Hughett, P. (2007). Accurate Computation of the F-to-z and t-to-z
           Transforms for Large Arguments. Journal of Statistical Software,
           23(1), 1-5.
    .. [2] Sochat, V. (2015, October 21). TtoZ Original Release. Zenodo.
           http://doi.org/10.5281/zenodo.32508
    """
    # Select just the nonzero voxels
    nonzero = t_values[t_values != 0]

    # We will store our results here
    z_values_nonzero = np.zeros(len(nonzero))

    # Select values less than or == 0, and greater than zero
    c = np.zeros(len(nonzero))
    k1 = nonzero <= c
    k2 = nonzero > c

    # Subset the data into two sets
    t1 = nonzero[k1]
    t2 = nonzero[k2]

    # Calculate p values for <=0
    p_values_t1 = stats.t.cdf(t1, df=dof)
    p_values_t1[p_values_t1 < np.finfo(p_values_t1.dtype).eps] = np.finfo(p_values_t1.dtype).eps
    z_values_t1 = stats.norm.ppf(p_values_t1)

    # Calculate p values for > 0
    p_values_t2 = stats.t.cdf(-t2, df=dof)
    p_values_t2[p_values_t2 < np.finfo(p_values_t2.dtype).eps] = np.finfo(p_values_t2.dtype).eps
    z_values_t2 = -stats.norm.ppf(p_values_t2)
    z_values_nonzero[k1] = z_values_t1
    z_values_nonzero[k2] = z_values_t2

    z_values = np.zeros(t_values.shape)
    z_values[t_values != 0] = z_values_nonzero
    return z_values


def z_to_t(z_values, dof):
    """Convert z-statistics to t-statistics.

    An inversion of the t_to_z implementation of [1]_ from Vanessa Sochat's
    TtoZ package [2]_.

    Parameters
    ----------
    z_values : array_like
        Z-statistics
    dof : int
        Degrees of freedom

    Returns
    -------
    t_values : array_like
        T-statistics

    References
    ----------
    .. [1] Hughett, P. (2007). Accurate Computation of the F-to-z and t-to-z
           Transforms for Large Arguments. Journal of Statistical Software,
           23(1), 1-5.
    .. [2] Sochat, V. (2015, October 21). TtoZ Original Release. Zenodo.
           http://doi.org/10.5281/zenodo.32508
    """
    # Select just the nonzero voxels
    nonzero = z_values[z_values != 0]

    # We will store our results here
    t_values_nonzero = np.zeros(len(nonzero))

    # Select values less than or == 0, and greater than zero
    c = np.zeros(len(nonzero))
    k1 = nonzero <= c
    k2 = nonzero > c

    # Subset the data into two sets
    z1 = nonzero[k1]
    z2 = nonzero[k2]

    # Calculate p values for <=0
    p_values_z1 = stats.norm.cdf(z1)
    t_values_z1 = stats.t.ppf(p_values_z1, df=dof)

    # Calculate p values for > 0
    p_values_z2 = stats.norm.cdf(-z2)
    t_values_z2 = -stats.t.ppf(p_values_z2, df=dof)
    t_values_nonzero[k1] = t_values_z1
    t_values_nonzero[k2] = t_values_z2

    t_values = np.zeros(z_values.shape)
    t_values[z_values != 0] = t_values_nonzero
    return t_values
