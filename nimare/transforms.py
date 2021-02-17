"""Miscellaneous spatial and statistical transforms
"""
import logging
import os.path as op

import nibabel as nib
import numpy as np
from scipy import stats

from . import references, utils
from .due import due

LGR = logging.getLogger(__name__)


def transform_images(images_df, target, masker, metadata_df=None, out_dir=None):
    """
    Generate images of a given type, depending on compatible images of other
    types, and write out to files.

    Parameters
    ----------
    images_df : :class:`pandas.DataFrame`
        DataFrame with paths to images for studies in Dataset.
    target : {'z', 'beta', 'varcope'}
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

    valid_targets = ["z", "beta", "varcope"]
    if target not in valid_targets:
        raise ValueError("Target type must be one of: {}".format(", ".join(valid_targets)))
    mask_img = masker.mask_img
    new_mask = np.ones(mask_img.shape, int)
    new_mask = nib.Nifti1Image(new_mask, mask_img.affine, header=mask_img.header)
    new_masker = utils.get_masker(new_mask)
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
    """Figure out the appropriate set of transforms for given available data
    to a target image type, and apply them.

    Parameters
    ----------
    target : {'z', 't', 'beta', 'varcope'}
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
    else:
        return None


def sample_sizes_to_dof(sample_sizes):
    """A simple heuristic for calculating degrees of freedom from a list of
    sample sizes.

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
    """A simple heuristic for appropriate sample size from a list of sample
    sizes.

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
    """Convert "sample variance of the dataset" (variance of the individual
    observations in a single sample) to "sampling variance" (variance of
    sampling distribution for the parameter).

    Parameters
    ----------
    samplevar_dataset : array_like
        Sample variance of the dataset (e.g., ``np.var(values)``).
    sample_size : int
        Sample size

    Returns
    -------
    varcope : array_like
        Sampling variance of the parameter

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
    """
    Convert t-statistics to z-statistics.

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
    z_values_t1 = stats.norm.ppf(p_values_t1)

    # Calculate p values for > 0
    p_values_t2 = stats.t.cdf(-t2, df=dof)
    z_values_t2 = -stats.norm.ppf(p_values_t2)
    z_values_nonzero[k1] = z_values_t1
    z_values_nonzero[k2] = z_values_t2

    z_values = np.zeros(t_values.shape)
    z_values[t_values != 0] = z_values_nonzero
    return z_values


def z_to_t(z_values, dof):
    """
    Convert z-statistics to t-statistics.

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


def vox2mm(ijk, affine):
    """
    Convert matrix subscripts to coordinates.

    Parameters
    ----------
    ijk : (X, 3) :obj:`numpy.ndarray`
        Matrix subscripts for coordinates being transformed.
        One row for each coordinate, with three columns: i, j, and k.
    affine : (4, 4) :obj:`numpy.ndarray`
        Affine matrix from image.

    Returns
    -------
    xyz : (X, 3) :obj:`numpy.ndarray`
        Coordinates in image-space.

    Notes
    -----
    From here:
    http://blog.chrisgorgolewski.org/2014/12/how-to-convert-between-voxel-and-mm.html
    """
    xyz = nib.affines.apply_affine(affine, ijk)
    return xyz


def mm2vox(xyz, affine):
    """
    Convert coordinates to matrix subscripts.

    Parameters
    ----------
    xyz : (X, 3) :obj:`numpy.ndarray`
        Coordinates in image-space.
        One row for each coordinate, with three columns: x, y, and z.
    affine : (4, 4) :obj:`numpy.ndarray`
        Affine matrix from image.

    Returns
    -------
    ijk : (X, 3) :obj:`numpy.ndarray`
        Matrix subscripts for coordinates being transformed.

    Notes
    -----
    From here:
    http://blog.chrisgorgolewski.org/2014/12/how-to-convert-between-voxel-and-mm.html
    """
    ijk = nib.affines.apply_affine(np.linalg.inv(affine), xyz).astype(int)
    return ijk


@due.dcite(
    references.LANCASTER_TRANSFORM,
    description="Introduces the Lancaster MNI-to-Talairach transform, "
    "as well as its inverse, the Talairach-to-MNI "
    "transform.",
)
@due.dcite(
    references.LANCASTER_TRANSFORM_VALIDATION,
    description="Validates the Lancaster MNI-to-Talairach and " "Talairach-to-MNI transforms.",
)
def tal2mni(coords):
    """
    Convert coordinates from Talairach space to MNI space.

    Parameters
    ----------
    coords : (X, 3) :obj:`numpy.ndarray`
        Coordinates in Talairach space to convert.
        Each row is a coordinate, with three columns.

    Returns
    -------
    coords : (X, 3) :obj:`numpy.ndarray`
        Coordinates in MNI space.
        Each row is a coordinate, with three columns.

    Notes
    -----
    Python version of BrainMap's tal2icbm_other.m.

    This function converts coordinates from Talairach space to MNI
    space (normalized using templates other than those contained
    in SPM and FSL) using the tal2icbm transform developed and
    validated by Jack Lancaster at the Research Imaging Center in
    San Antonio, Texas.
    http://www3.interscience.wiley.com/cgi-bin/abstract/114104479/ABSTRACT
    """
    # Find which dimensions are of size 3
    shape = np.array(coords.shape)
    if all(shape == 3):
        LGR.info("Input is an ambiguous 3x3 matrix.\nAssuming coords are row " "vectors (Nx3).")
        use_dim = 1
    elif not any(shape == 3):
        raise AttributeError("Input must be an Nx3 or 3xN matrix.")
    else:
        use_dim = np.where(shape == 3)[0][0]

    # Transpose if necessary
    if use_dim == 1:
        coords = coords.transpose()

    # Transformation matrices, different for each software package
    icbm_other = np.array(
        [
            [0.9357, 0.0029, -0.0072, -1.0423],
            [-0.0065, 0.9396, -0.0726, -1.3940],
            [0.0103, 0.0752, 0.8967, 3.6475],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ]
    )

    # Invert the transformation matrix
    icbm_other = np.linalg.inv(icbm_other)

    # Apply the transformation matrix
    coords = np.concatenate((coords, np.ones((1, coords.shape[1]))))
    coords = np.dot(icbm_other, coords)

    # Format the output, transpose if necessary
    out_coords = coords[:3, :]
    if use_dim == 1:
        out_coords = out_coords.transpose()
    return out_coords


@due.dcite(
    references.LANCASTER_TRANSFORM,
    description="Introduces the Lancaster MNI-to-Talairach transform, "
    "as well as its inverse, the Talairach-to-MNI "
    "transform.",
)
@due.dcite(
    references.LANCASTER_TRANSFORM_VALIDATION,
    description="Validates the Lancaster MNI-to-Talairach and " "Talairach-to-MNI transforms.",
)
def mni2tal(coords):
    """
    Convert coordinates from MNI space Talairach space.

    Parameters
    ----------
    coords : (X, 3) :obj:`numpy.ndarray`
        Coordinates in MNI space to convert.
        Each row is a coordinate, with three columns.

    Returns
    -------
    coords : (X, 3) :obj:`numpy.ndarray`
        Coordinates in Talairach space.
        Each row is a coordinate, with three columns.

    Notes
    -----
    Python version of BrainMap's icbm_other2tal.m.
    This function converts coordinates from MNI space (normalized using
    templates other than those contained in SPM and FSL) to Talairach space
    using the icbm2tal transform developed and validated by Jack Lancaster at
    the Research Imaging Center in San Antonio, Texas.
    http://www3.interscience.wiley.com/cgi-bin/abstract/114104479/ABSTRACT
    """
    # Find which dimensions are of size 3
    shape = np.array(coords.shape)
    if all(shape == 3):
        LGR.info("Input is an ambiguous 3x3 matrix.\nAssuming coords are row " "vectors (Nx3).")
        use_dim = 1
    elif not any(shape == 3):
        raise AttributeError("Input must be an Nx3 or 3xN matrix.")
    else:
        use_dim = np.where(shape == 3)[0][0]

    # Transpose if necessary
    if use_dim == 1:
        coords = coords.transpose()

    # Transformation matrices, different for each software package
    icbm_other = np.array(
        [
            [0.9357, 0.0029, -0.0072, -1.0423],
            [-0.0065, 0.9396, -0.0726, -1.3940],
            [0.0103, 0.0752, 0.8967, 3.6475],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ]
    )

    # Apply the transformation matrix
    coords = np.concatenate((coords, np.ones((1, coords.shape[1]))))
    coords = np.dot(icbm_other, coords)

    # Format the output, transpose if necessary
    out_coords = coords[:3, :]
    if use_dim == 1:
        out_coords = out_coords.transpose()
    return out_coords
