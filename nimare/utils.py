"""Utility functions for NiMARE."""
import contextlib
import datetime
import inspect
import json
import logging
import multiprocessing as mp
import os
import os.path as op
import re
import warnings
from functools import wraps
from string import ascii_lowercase
from tempfile import mkstemp

import joblib
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn._utils import check_niimg_3d
from nilearn._utils.niimg import _safe_get_data
from nilearn.image import new_img_like, resampling, threshold_img
from nilearn.input_data import NiftiMasker
from scipy.ndimage import (
    center_of_mass,
    generate_binary_structure,
    label,
    maximum_filter,
    minimum_filter,
)

LGR = logging.getLogger(__name__)


def _check_ncores(n_cores):
    """Check number of cores used for method.

    .. versionadded:: 0.0.12
        Moved from Estimator._check_ncores into its own function.
    """
    if n_cores <= 0:
        n_cores = mp.cpu_count()
    elif n_cores > mp.cpu_count():
        LGR.warning(
            f"Desired number of cores ({n_cores}) greater than number "
            f"available ({mp.cpu_count()}). Setting to {mp.cpu_count()}."
        )
        n_cores = mp.cpu_count()
    return n_cores


def get_resource_path():
    """Return the path to general resources, terminated with separator.

    Resources are kept outside package folder in "datasets".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return op.abspath(op.join(op.dirname(__file__), "resources") + op.sep)


def get_template(space="mni152_2mm", mask=None):
    """Load template file.

    .. versionchanged:: 0.0.11

        - Remove the ``mask="gm"`` option.
        - Replace the nilearn templates with ones downloaded directly from TemplateFlow.

    Parameters
    ----------
    space : {'mni152_1mm', 'mni152_2mm', 'ale_2mm'}, optional
        Template to load. Default is 'mni152_2mm'.
        The options are:

            -   mni152_1mm: The MNI152NLin6Asym template at 1mm3 resolution,
                downloaded from TemplateFlow. The shape of this template is 182x218x182 voxels.
            -   mni152_2mm: The MNI152NLin6Asym template at 2mm3 resolution,
                downloaded from TemplateFlow. The shape of this template is 91x109x91 voxels.
            -   ale_2mm: The template used is the MNI152NLin6Asym template at 2mm3 resolution,
                but if ``mask='brain'``, then a brain mask taken from GingerALE will be used.
                The brain mask corresponds to GingerALE's "more conservative" mask.
                The shape of this template is 91x109x91 voxels.
    mask : {None, 'brain'}, optional
        Whether to return the raw T1w template (None) or a brain mask ('brain').
        Default is None.

    Returns
    -------
    img : :obj:`~nibabel.nifti1.Nifti1Image`
        Template image object.
    """
    template_dir = op.join(get_resource_path(), "templates")
    if space == "mni152_1mm":
        if mask is None:
            img = nib.load(op.join(template_dir, "tpl-MNI152NLin6Asym_res-01_T1w.nii.gz"))
        elif mask == "brain":
            img = nib.load(
                op.join(template_dir, "tpl-MNI152NLin6Asym_res-01_desc-brain_mask.nii.gz")
            )
        else:
            raise ValueError(f"Mask option '{mask}' not supported")
    elif space == "mni152_2mm":
        if mask is None:
            img = nib.load(op.join(template_dir, "tpl-MNI152NLin6Asym_res-02_T1w.nii.gz"))
        elif mask == "brain":
            img = nib.load(
                op.join(template_dir, "tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz")
            )
        else:
            raise ValueError(f"Mask option '{mask}' not supported")
    elif space == "ale_2mm":
        if mask is None:
            img = nib.load(op.join(template_dir, "tpl-MNI152NLin6Asym_res-02_T1w.nii.gz"))
        elif mask == "brain":
            # Not the same as the nilearn brain mask, but should correspond to
            # the default "more conservative" MNI152 mask in GingerALE.
            img = nib.load(op.join(template_dir, "MNI152_2x2x2_brainmask.nii.gz"))
        else:
            raise ValueError(f"Mask option '{mask}' not supported")
    else:
        raise ValueError(f"Space '{space}' not supported")

    # Coerce to array-image
    img = nib.Nifti1Image(img.get_fdata(), affine=img.affine, header=img.header)
    return img


def get_masker(mask):
    """Get an initialized, fitted nilearn Masker instance from passed argument.

    Parameters
    ----------
    mask : str, :class:`nibabel.nifti1.Nifti1Image`, or any nilearn Masker

    Returns
    -------
    masker : an initialized, fitted instance of a subclass of
        `nilearn.input_data.base_masker.BaseMasker`
    """
    if isinstance(mask, str):
        mask = nib.load(mask)

    if isinstance(mask, nib.nifti1.Nifti1Image):
        # Coerce to array-image
        mask = nib.Nifti1Image(mask.get_fdata(), affine=mask.affine, header=mask.header)

        mask = NiftiMasker(mask)

    if not (hasattr(mask, "transform") and hasattr(mask, "inverse_transform")):
        raise ValueError(
            "mask argument must be a string, a nibabel image, or a Nilearn Masker instance."
        )

    # Fit the masker if needed
    if not hasattr(mask, "mask_img_"):
        mask.fit()

    return mask


def vox2mm(ijk, affine):
    """Convert matrix subscripts to coordinates.

    .. versionchanged:: 0.0.8

        * [ENH] This function was part of `nimare.transforms` in previous versions (0.0.3-0.0.7)

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
    """Convert coordinates to matrix subscripts.

    .. versionchanged:: 0.0.8

        * [ENH] This function was part of `nimare.transforms` in previous versions (0.0.3-0.0.7)

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


def tal2mni(coords):
    """Convert coordinates from Talairach space to MNI space.

    .. versionchanged:: 0.0.8

        * [ENH] This function was part of `nimare.transforms` in previous versions (0.0.3-0.0.7)

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
        LGR.info("Input is an ambiguous 3x3 matrix.\nAssuming coords are row vectors (Nx3).")
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


def mni2tal(coords):
    """Convert coordinates from MNI space Talairach space.

    .. versionchanged:: 0.0.8

        * [ENH] This function was part of `nimare.transforms` in previous versions (0.0.3-0.0.7)

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
        LGR.info("Input is an ambiguous 3x3 matrix.\nAssuming coords are row vectors (Nx3).")
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


def _dict_to_df(id_df, data, key="labels"):
    """Load a given data type in NIMADS-format dictionary into DataFrame.

    Parameters
    ----------
    id_df : :obj:`pandas.DataFrame`
        DataFrame with columns for identifiers. Index is [studyid]-[expid].
    data : :obj:`dict`
        NIMADS-format dictionary storing the raw dataset, from which
        relevant data are loaded into DataFrames.
    key : {'labels', 'metadata', 'text', 'images'}
        Which data type to load.

    Returns
    -------
    df : :obj:`pandas.DataFrame`
        DataFrame with id columns from id_df and new columns for the
        requested data type.
    """
    exp_dict = {}
    for pid in data.keys():
        for expid in data[pid]["contrasts"].keys():
            exp = data[pid]["contrasts"][expid]
            id_ = f"{pid}-{expid}"

            if key not in data[pid]["contrasts"][expid].keys():
                continue
            exp_dict[id_] = exp[key]

    temp_df = pd.DataFrame.from_dict(exp_dict, orient="index")
    df = pd.merge(id_df, temp_df, left_index=True, right_index=True, how="outer")
    df = df.reset_index(drop=True)
    df = df.replace(to_replace="None", value=np.nan)
    # replace nan with none
    df = df.where(pd.notnull(df), None)
    return df


def _dict_to_coordinates(data, masker, space):
    """Load coordinates in NIMADS-format dictionary into DataFrame."""
    # Required columns
    columns = ["id", "study_id", "contrast_id", "x", "y", "z", "space"]
    core_columns = columns.copy()  # Used in contrast for loop

    all_dfs = []
    for pid in data.keys():
        for expid in data[pid]["contrasts"].keys():
            if "coords" not in data[pid]["contrasts"][expid].keys():
                continue

            exp_columns = core_columns.copy()
            exp = data[pid]["contrasts"][expid]

            # Required info (ids, x, y, z, space)
            n_coords = len(exp["coords"]["x"])
            rep_id = np.array([[f"{pid}-{expid}", pid, expid]] * n_coords).T

            space_arr = exp["coords"].get("space")
            space_arr = np.array([space_arr] * n_coords)
            temp_data = np.vstack(
                (
                    rep_id,
                    np.array(exp["coords"]["x"]),
                    np.array(exp["coords"]["y"]),
                    np.array(exp["coords"]["z"]),
                    space_arr,
                )
            )

            # Optional information
            for k in list(set(exp["coords"].keys()) - set(core_columns)):
                k_data = exp["coords"][k]
                if not isinstance(k_data, list):
                    k_data = np.array([k_data] * n_coords)
                exp_columns.append(k)

                if k not in columns:
                    columns.append(k)
                temp_data = np.vstack((temp_data, k_data))

            # Place data in list of dataframes to merge
            con_df = pd.DataFrame(temp_data.T, columns=exp_columns)
            all_dfs.append(con_df)

    if not all_dfs:
        return pd.DataFrame(
            {
                "id": [],
                "study_id": [],
                "contrast_id": [],
                "x": [],
                "y": [],
                "z": [],
                "space": [],
            },
        )

    df = pd.concat(all_dfs, axis=0, join="outer", sort=False)
    df = df[columns].reset_index(drop=True)
    df = df.replace(to_replace="None", value=np.nan)
    # replace nan with none
    df = df.where(pd.notnull(df), None)
    df[["x", "y", "z"]] = df[["x", "y", "z"]].astype(float)
    df = _transform_coordinates_to_space(df, masker, space)
    return df


def _transform_coordinates_to_space(df, masker, space):
    """Convert xyz coordinates in a DataFrame to ijk indices for a given target space.

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
    masker : :class:`~nilearn.input_data.NiftiMasker` or similar
        Masker object defining the space and location of the area of interest
        (e.g., 'brain').
    space : :obj:`str`
        String describing the stereotactic space and resolution of the masker.

    Returns
    -------
    df : :obj:`pandas.DataFrame`
        DataFrame with IJK columns either added or overwritten.
    """
    # Now to apply transformations!
    if "mni" in space.lower() or "ale" in space.lower():
        transform = {"MNI": None, "TAL": tal2mni, "Talairach": tal2mni}
    elif "tal" in space.lower():
        transform = {"MNI": mni2tal, "TAL": None, "Talairach": None}
    else:
        raise ValueError(f"Unrecognized space: {space}")

    found_spaces = df["space"].unique()
    for found_space in found_spaces:
        if found_space not in transform.keys():
            LGR.warning(
                f"Not applying transforms to coordinates in unrecognized space '{found_space}'"
            )
        alg = transform.get(found_space, None)
        idx = df["space"] == found_space
        if alg:
            df.loc[idx, ["x", "y", "z"]] = alg(df.loc[idx, ["x", "y", "z"]].values)
        df.loc[idx, "space"] = space

    return df


def _validate_df(df):
    """Check that an input is a DataFrame and has a column for 'id'."""
    assert isinstance(df, pd.DataFrame)
    assert "id" in df.columns


def _validate_images_df(image_df):
    """Check and update image paths in DataFrame.

    Parameters
    ----------
    image_df : :class:`pandas.DataFrame`
        DataFrame with one row for each study and one column for each image
        type. Cells contain paths to image files.

    Returns
    -------
    image_df : :class:`pandas.DataFrame`
        DataFrame with updated paths and columns.
    """
    valid_suffixes = [".brik", ".head", ".nii", ".img", ".hed"]

    # Find columns in the DataFrame with images
    file_cols = []
    for col in image_df.columns:
        vals = [v for v in image_df[col].values if isinstance(v, str)]
        fc = any([any([vs in v for vs in valid_suffixes]) for v in vals])
        if fc:
            file_cols.append(col)

    # Clean up DataFrame
    # Find out which columns have full paths and which have relative paths
    abs_cols = []
    for col in file_cols:
        files = image_df[col].tolist()
        abspaths = [f == op.abspath(f) for f in files if isinstance(f, str)]
        if all(abspaths):
            abs_cols.append(col)
        elif not any(abspaths):
            if not col.endswith("__relative"):
                image_df = image_df.rename(columns={col: col + "__relative"})
        else:
            raise ValueError(
                f"Mix of absolute and relative paths detected for images in column '{col}'"
            )

    # Set relative paths from absolute ones
    if len(abs_cols):
        all_files = list(np.ravel(image_df[abs_cols].values))
        all_files = [f for f in all_files if isinstance(f, str)]

        if len(all_files) == 1:
            # In the odd case where there's only one absolute path
            shared_path = op.dirname(all_files[0]) + op.sep
        else:
            shared_path = _find_stem(all_files)

        # Get parent *directory* if shared path includes common prefix.
        if not shared_path.endswith(op.sep):
            shared_path = op.dirname(shared_path) + op.sep
        LGR.info(f"Shared path detected: '{shared_path}'")

        image_df_out = image_df.copy()  # To avoid SettingWithCopyWarning
        for abs_col in abs_cols:
            image_df_out[abs_col + "__relative"] = image_df[abs_col].apply(
                lambda x: x.split(shared_path)[1] if isinstance(x, str) else x
            )

        image_df = image_df_out

    return image_df


def _listify(obj):
    """Wrap all non-list or tuple objects in a list.

    This provides a simple way to accept flexible arguments.
    """
    return obj if isinstance(obj, (list, tuple, type(None), np.ndarray)) else [obj]


def _round2(ndarray):
    """Round X.5 to the nearest integer away from zero.

    Numpy rounds X.5 values to nearest even integer.
    """
    onedarray = ndarray.flatten()
    signs = np.sign(onedarray)  # pylint: disable=no-member
    idx = np.where(np.abs(onedarray - np.round(onedarray)) == 0.5)[0]
    x = np.abs(onedarray)
    y = np.round(x)
    y[idx] = np.ceil(x[idx])
    y *= signs
    rounded = y.reshape(ndarray.shape)
    return rounded.astype(int)


def _try_prepend(value, prefix):
    """Try to prepend a value to a string with a separator ('/').

    If not a string, will just return the original value.
    """
    if isinstance(value, str):
        return op.join(prefix, value)
    else:
        return value


def _find_stem(arr):
    """Find longest common substring in array of strings.

    From https://www.geeksforgeeks.org/longest-common-substring-array-strings/
    """
    # Determine size of the array
    n_items_in_array = len(arr)

    # Take first word from array as reference
    reference_string = arr[0]
    n_chars_in_first_item = len(reference_string)

    res = ""
    for i_char in range(n_chars_in_first_item):
        # Generate all starting substrings of our reference string
        stem = reference_string[:i_char]

        j_item = 1  # Retained in case of an array with only one item
        for j_item in range(1, n_items_in_array):
            # Check if the generated stem is common to to all words
            if not arr[j_item].startswith(stem):
                break

        # If current substring is present in all strings and its length is
        # greater than current result
        if (j_item + 1 == n_items_in_array) and (len(res) < len(stem)):
            res = stem

    return res


def _uk_to_us(text):
    """Convert UK spellings to US based on a converter.

    .. versionadded:: 0.0.2

    Parameters
    ----------
    text : :obj:`str`

    Returns
    -------
    text : :obj:`str`

    Notes
    -----
    The english_spellings.csv file is from http://www.tysto.com/uk-us-spelling-list.html.
    """
    SPELL_DF = pd.read_csv(op.join(get_resource_path(), "english_spellings.csv"), index_col="UK")
    SPELL_DICT = SPELL_DF["US"].to_dict()

    if isinstance(text, str):
        # Convert British to American English
        pattern = re.compile(r"\b(" + "|".join(SPELL_DICT.keys()) + r")\b")
        text = pattern.sub(lambda x: SPELL_DICT[x.group()], text)
    return text


def use_memmap(logger, n_files=1):
    """Memory-map array to a file, and perform cleanup after.

    .. versionadded:: 0.0.8

    Parameters
    ----------
    logger : :obj:`logging.Logger`
        A Logger with which to log information about the function.
    n_files : :obj:`int`, optional
        Number of memory-mapped files to create and manage.

    Notes
    -----
    This function is used as a decorator to methods in which memory-mapped arrays may be used.
    It will only be triggered if the class to which the method belongs has a ``memory_limit``
    attribute that is set to something other than ``None``.

    It will set an attribute within the method's class named ``memmap_filenames``, which is a list
    of filename strings, with ``n_files`` elements.
    If ``memory_limit`` is None, then it will be a list of ``Nones``.

    Files generated by this function will be stored in the NiMARE data directory and will be
    removed after the wrapped method finishes.
    """

    def inner_function(function):
        @wraps(function)
        def memmap_context(self, *args, **kwargs):
            if hasattr(self, "memory_limit") and self.memory_limit:
                self.memmap_filenames, filenames = [], []
                for i_file in range(n_files):
                    start_time = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
                    _, filename = mkstemp(prefix=self.__class__.__name__, suffix=start_time)
                    logger.debug(f"Temporary file written to {filename}")
                    self.memmap_filenames.append(filename)
                    filenames.append(filename)
            else:
                filenames = self.memmap_filenames = [None] * n_files

            try:
                return function(self, *args, **kwargs)
            except:
                for filename in filenames:
                    logger.error(f"{function.__name__} failed, removing {filename}")
                raise
            finally:
                if hasattr(self, "memory_limit") and self.memory_limit:
                    for filename in filenames:
                        if os.path.isfile(filename):
                            logger.debug(f"Removing temporary file: {filename}")
                            os.remove(filename)
                        else:
                            logger.debug(f"Temporary file DNE: {filename}")

        return memmap_context

    return inner_function


BYTE = 2
KILOBYTE = BYTE**10
BYTE_CONVERSION = {
    "kb": KILOBYTE,
    "mb": KILOBYTE**2,
    "gb": KILOBYTE**3,
    "tb": KILOBYTE**4,
}


def _determine_chunk_size(limit, arr, multiplier=1):
    """Determine how many arrays can be read into memory at once.

    Parameters
    ----------
    limit : :obj:`str`
        String representation of memory limit, can use:
        kb, mb, gb, and tb as suffix (e.g., "4gb").
    arr : :obj:`numpy.array`
        Representative numpy array.
    multiplier : :obj:`int`
        Adjustment for processes that have more or
        less overhead than expected.
    """
    limit = limit.lower()
    size, representation = re.search(r"([0-9]+)([a-z]+)", limit).groups()

    limit_bytes = float(size) * BYTE_CONVERSION[representation] * multiplier

    arr_bytes = arr.size * arr.itemsize

    chunk_size = int(limit_bytes // arr_bytes)

    if chunk_size == 0:
        arr_size = arr_bytes // BYTE_CONVERSION["mb"]
        raise RuntimeError(f"memory limit: {limit} too small for array with size {arr_size}mb")

    return chunk_size


def _safe_transform(imgs, masker, memory_limit="1gb", dtype="auto", memfile=None):
    """Apply a masker with limited memory usage.

    Parameters
    ----------
    imgs : list of niimgs
        List of images upon which to apply the masker.
    masker : nilearn masker
        Masker object to apply to images.
    memory_limit : :obj:`str`, optional
        String representation of memory limit, can use:
        kb, mb, gb, and tb as suffix (e.g., "4gb").
    dtype : :obj:`str`, optional
        Target datatype of masked array.
        Default is "auto", which uses the datatype of the niimgs.
    memfile : :obj:`str` or None, optional
        Name of a memory-mapped file. If None, memory-mapping will not be used.

    Returns
    -------
    masked_data : :obj:`numpy.ndarray` or :obj:`numpy.memmap`
        Masked data in a 2D array.
        Either an ndarray (if memfile is None) or a memmap array (if memfile is a string).
    """
    assert isinstance(memfile, (type(None), str))

    first_img_data = masker.transform(imgs[0])
    masked_shape = (len(imgs), first_img_data.size)
    if memfile:
        masked_data = np.memmap(
            memfile,
            dtype=first_img_data.dtype if dtype == "auto" else dtype,
            mode="w+",
            shape=masked_shape,
        )
    else:
        masked_data = np.empty(
            masked_shape,
            dtype=first_img_data.dtype if dtype == "auto" else dtype,
        )

    # perform transform on chunks of the input maps
    chunk_size = _determine_chunk_size(memory_limit, first_img_data)
    map_chunks = [imgs[i : i + chunk_size] for i in range(0, len(imgs), chunk_size)]
    idx = 0
    for map_chunk in map_chunks:
        end_idx = idx + len(map_chunk)
        map_chunk_data = masker.transform(map_chunk)
        masked_data[idx:end_idx, :] = map_chunk_data
        idx = end_idx

    return masked_data


def _add_metadata_to_dataframe(
    dataset,
    dataframe,
    metadata_field,
    target_column,
    filter_func=np.mean,
):
    """Add metadata from a Dataset to a DataFrame.

    .. versionadded:: 0.0.8

    This is particularly useful for kernel transformers or estimators where a given metadata field
    is necessary (e.g., ALEKernel with "sample_size"), but we want to just use the coordinates
    DataFrame instead of passing the full Dataset.

    Parameters
    ----------
    dataset : :obj:`~nimare.dataset.Dataset`
        Dataset containing study IDs and metadata to feed into dataframe.
    dataframe : :obj:`pandas.DataFrame`
        DataFrame containing study IDs, into which Dataset metadata will be merged.
    metadata_field : :obj:`str`
        Metadata field in ``dataset``.
    target_column : :obj:`str`
        Name of the column that will be added to ``dataframe``, containing information from the
        Dataset.
    filter_func : :obj:`function`, optional
        Function to apply to the metadata so that it fits as a column in a DataFrame.
        Default is ``numpy.mean``.

    Returns
    -------
    dataframe : :obj:`pandas.DataFrame`
        Updated DataFrame with ``target_column`` added.
    """
    dataframe = dataframe.copy()

    if metadata_field in dataset.get_metadata():
        # Collect metadata from Dataset
        metadata = dataset.get_metadata(field=metadata_field, ids=dataset.ids)
        metadata = [[m] for m in metadata]
        # Create a DataFrame with the metadata
        metadata = pd.DataFrame(
            index=dataset.ids,
            data=metadata,
            columns=[metadata_field],
        )
        # Reduce the metadata (if in list/array format) to single values
        metadata[target_column] = metadata[metadata_field].apply(filter_func)
        # Merge metadata df into coordinates df
        dataframe = dataframe.merge(
            right=metadata,
            left_on="id",
            right_index=True,
            sort=False,
            validate="many_to_one",
            suffixes=(False, False),
            how="left",
        )
    else:
        LGR.warning(
            f"Metadata field '{metadata_field}' not found. "
            "Set a constant value for this field as an argument, if possible."
        )

    return dataframe


def _check_type(obj, clss, **kwargs):
    """Check variable type and initialize if necessary.

    .. versionadded:: 0.0.8

    Parameters
    ----------
    obj
        Object to check and initialized if necessary.
    clss
        Target class of the object.
    kwargs
        Dictionary of keyword arguments that can be used when initializing the object.

    Returns
    -------
    obj
        Initialized version of the object.
    """
    # Allow both instances and classes for the input.
    if not issubclass(type(obj), clss) and not issubclass(obj, clss):
        raise ValueError(f"Argument {type(obj)} must be a kind of {clss}")
    elif not inspect.isclass(obj) and kwargs:
        LGR.warning(
            f"Argument {type(obj)} has already been initialized, so arguments "
            f"will be ignored: {', '.join(kwargs.keys())}"
        )
    elif inspect.isclass(obj):
        obj = obj(**kwargs)
    return obj


def _boolean_unmask(data_array, bool_array):
    """Unmask data based on a boolean array, with NaNs in empty voxels.

    Parameters
    ----------
    data_array : 1D or 2D :obj:`numpy.ndarray`
        Masked data array.
    bool_array : 1D :obj:`numpy.ndarray`
        Boolean mask array. Must have the same number of ``True`` entries as elements in the
        second dimension of ``data_array``.

    Returns
    -------
    unmasked_data : 1D or 2D :obj:`numpy.ndarray`
        Unmasked data array.
        If 1D, first dimension is the same size as the first (and only) dimension of
        ``boolean_array``.
        If 2D, first dimension is the same size as the first dimension of ``data_array``, while
        second dimension is the same size as the first (and only) dimension  of ``boolean_array``.
        All elements corresponding to ``False`` values in ``boolean_array`` will have NaNs.
    """
    assert data_array.ndim in (1, 2)
    assert bool_array.ndim == 1
    assert bool_array.sum() == data_array.shape[-1]

    unmasked_data = np.full(
        shape=bool_array.shape + data_array.T.shape[1:],
        fill_value=np.nan,
        dtype=data_array.dtype,
    )
    unmasked_data[bool_array] = data_array
    unmasked_data = unmasked_data.T
    return unmasked_data


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument.

    From https://stackoverflow.com/a/58936697/2589328.
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def unique_rows(ar, return_counts=False):
    """Remove repeated rows from a 2D array.

    In particular, if given an array of coordinates of shape
    (Npoints, Ndim), it will remove repeated points.

    Parameters
    ----------
    ar : 2-D ndarray
        The input array.
    return_counts : :obj:`bool`, optional
        If True, also return the number of times each unique item appears in ar.

    Returns
    -------
    ar_out : 2-D ndarray
        A copy of the input array with repeated rows removed.
    unique_counts : :obj:`np.ndarray`, optional
        The number of times each of the unique values comes up in the original array.
        Only provided if return_counts is True.

    Raises
    ------
    ValueError : if `ar` is not two-dimensional.

    Notes
    -----
    The function will generate a copy of `ar` if it is not
    C-contiguous, which will negatively affect performance for large
    input arrays.

    This is taken from skimage. See :func:`skimage.util.unique_rows`.

    Examples
    --------
    >>> ar = np.array([[1, 0, 1],
    ...                [0, 1, 0],
    ...                [1, 0, 1]], np.uint8)
    >>> unique_rows(ar)
    array([[0, 1, 0],
           [1, 0, 1]], dtype=uint8)

    License
    -------
    Copyright (C) 2019, the scikit-image team
    All rights reserved.
    """
    if ar.ndim != 2:
        raise ValueError("unique_rows() only makes sense for 2D arrays, " "got %dd" % ar.ndim)
    # the view in the next line only works if the array is C-contiguous
    ar = np.ascontiguousarray(ar)
    # np.unique() finds identical items in a raveled array. To make it
    # see each row as a single item, we create a view of each row as a
    # byte string of length itemsize times number of columns in `ar`
    ar_row_view = ar.view("|S%d" % (ar.itemsize * ar.shape[1]))
    if return_counts:
        _, unique_row_indices, counts = np.unique(
            ar_row_view, return_index=True, return_counts=True
        )

        return ar[unique_row_indices], counts
    else:
        _, unique_row_indices = np.unique(ar_row_view, return_index=True)

        return ar[unique_row_indices]


def find_braces(string):
    """Search a string for matched braces.

    This is used to identify pairs of braces in BibTeX files.
    The outside-most pairs should correspond to BibTeX entries.

    Parameters
    ----------
    string : :obj:`str`
        A long string to search for paired braces.

    Returns
    -------
    :obj:`list` of :obj:`tuple` of :obj:`int`
        A list of two-element tuples of indices of matched braces.
    """
    toret = {}
    pstack = []

    for idx, char in enumerate(string):
        if char == "{":
            pstack.append(idx)
        elif char == "}":
            if len(pstack) == 0:
                raise IndexError(f"No matching closing parens at: {idx}")

            toret[pstack.pop()] = idx

    if len(pstack) > 0:
        raise IndexError(f"No matching opening parens at: {pstack.pop()}")

    toret = list(toret.items())
    return toret


def reduce_idx(idx_list):
    """Identify outermost brace indices in list of indices.

    The purpose here is to find the brace pairs that correspond to BibTeX entries,
    while discarding brace pairs that appear within the entries
    (e.g., braces around article titles).

    Parameters
    ----------
    idx_list : :obj:`list` of :obj:`tuple` of :obj:`int`
        A list of two-element tuples of indices of matched braces.

    Returns
    -------
    reduced_idx_list : :obj:`list` of :obj:`tuple` of :obj:`int`
        A list of two-element tuples of indices of matched braces corresponding to BibTeX entries.
    """
    idx_list2 = [idx_item[0] for idx_item in idx_list]
    idx = np.argsort(idx_list2)
    idx_list = [idx_list[i] for i in idx]

    df = pd.DataFrame(data=idx_list, columns=["start", "end"])

    good_idx = []
    df["within"] = False
    for i, row in df.iterrows():
        df["within"] = df["within"] | ((df["start"] > row["start"]) & (df["end"] < row["end"]))
        if not df.iloc[i]["within"]:
            good_idx.append(i)

    idx_list = [idx_list[i] for i in good_idx]
    return idx_list


def index_bibtex_identifiers(string, idx_list):
    """Identify the BibTeX entry identifier before each entry.

    The purpose of this function is to take the raw BibTeX string and a list of indices of entries,
    starting and ending with the braces of each entry, and then extract the identifier before each.

    Parameters
    ----------
    string : :obj:`str`
        The full BibTeX file, as a string.
    idx_list : :obj:`list` of :obj:`tuple` of :obj:`int`
        A list of two-element tuples of indices of matched braces corresponding to BibTeX entries.

    Returns
    -------
    idx_list : :obj:`list` of :obj:`tuple` of :obj:`int`
        A list of two-element tuples of indices of BibTeX entries,
        from the starting @ to the final }.
    """
    at_idx = [(a.start(), a.end() - 1) for a in re.finditer("@[a-zA-Z0-9]+{", string)]
    df = pd.DataFrame(at_idx, columns=["real_start", "false_start"])
    df2 = pd.DataFrame(idx_list, columns=["false_start", "end"])
    df = pd.merge(left=df, right=df2, left_on="false_start", right_on="false_start")
    new_idx_list = list(zip(df.real_start, df.end))
    return new_idx_list


def find_citations(description):
    r"""Find citations in a text description.

    It looks for cases of \\citep{} and \\cite{} in a string.

    Parameters
    ----------
    description_ : :obj:`str`
        Description of a method, optionally with citations.

    Returns
    -------
    all_citations : :obj:`list` of :obj:`str`
        A list of all identifiers for citations.
    """
    paren_citations = re.findall(r"\\citep{([a-zA-Z0-9,/\.]+)}", description)
    intext_citations = re.findall(r"\\cite{([a-zA-Z0-9,/\.]+)}", description)
    inparen_citations = re.findall(r"\\citealt{([a-zA-Z0-9,/\.]+)}", description)
    all_citations = ",".join(paren_citations + intext_citations + inparen_citations)
    all_citations = all_citations.split(",")
    all_citations = sorted(list(set(all_citations)))
    return all_citations


def reduce_references(citations, reference_list):
    """Reduce the list of references to only include ones associated with requested citations.

    Parameters
    ----------
    citations : :obj:`list` of :obj:`str`
        A list of all identifiers for citations.
    reference_list : :obj:`list` of :obj:`str`
        List of all available BibTeX entries.

    Returns
    -------
    reduced_reference_list : :obj:`list` of :obj:`str`
        List of BibTeX entries for citations only.
    """
    reduced_reference_list = []
    for citation in citations:
        citation_found = False
        for reference in reference_list:
            check_string = "@[a-zA-Z]+{" + citation + ","
            if re.match(check_string, reference):
                reduced_reference_list.append(reference)
                citation_found = True
                continue

        if not citation_found:
            LGR.warning(f"Citation {citation} not found.")

    return reduced_reference_list


def get_description_references(description):
    """Find BibTeX references for citations in a methods description.

    Parameters
    ----------
    description_ : :obj:`str`
        Description of a method, optionally with citations.

    Returns
    -------
    bibtex_string : :obj:`str`
        A string containing BibTeX entries, limited only to the citations in the description.
    """
    bibtex_file = op.join(get_resource_path(), "references.bib")
    with open(bibtex_file, "r") as fo:
        bibtex_string = fo.read()

    braces_idx = find_braces(bibtex_string)
    red_braces_idx = reduce_idx(braces_idx)
    bibtex_idx = index_bibtex_identifiers(bibtex_string, red_braces_idx)
    citations = find_citations(description)
    reference_list = [bibtex_string[start : end + 1] for start, end in bibtex_idx]
    reduced_reference_list = reduce_references(citations, reference_list)

    bibtex_string = "\n".join(reduced_reference_list)
    return bibtex_string


def _local_max(data, affine, min_distance):
    """Find all local maxima of the array, separated by at least min_distance.

    Adapted from https://stackoverflow.com/a/22631583/2589328

    Parameters
    ----------
    data : array_like
        3D array of with masked values for cluster.

    affine : np.ndarray
        Square matrix specifying the position of the image array data
        in a reference space.

    min_distance : int
        Minimum distance between local maxima in ``data``, in terms of mm.

    Returns
    -------
    ijk : `numpy.ndarray`
        (n_foci, 3) array of local maxima indices for cluster.

    vals : `numpy.ndarray`
        (n_foci,) array of values from data at ijk.

    This function is partially derived from Nilearn's code.

    License
    -------
    New BSD License

    Copyright (c) 2007 - 2023 The nilearn developers.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    a. Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimer.
    b. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
    c. Neither the name of the nilearn developers nor the names of
        its contributors may be used to endorse or promote products
        derived from this software without specific prior written
        permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
    DAMAGE.
    """
    ijk, vals = _identify_subpeaks(data)
    xyz, ijk, vals = _sort_subpeaks(ijk, vals, affine)
    ijk, vals = _pare_subpeaks(xyz, ijk, vals, min_distance)
    return ijk, vals


def _identify_subpeaks(data):
    """Identify cluster peak and subpeaks based on minimum distance.

    Parameters
    ----------
    data : `numpy.ndarray`
        3D array of with masked values for cluster.

    Returns
    -------
    ijk : `numpy.ndarray`
        (n_foci, 3) array of local maximum indices for cluster.
    vals : `numpy.ndarray`
        (n_foci,) array of values from data at ijk.

    Notes
    -----
    When a cluster's local maximum corresponds to contiguous voxels with the
    same values (as in a binary cluster), this function determines the center
    of mass for those voxels. If the center of mass falls outside the cluster,
    we instead report the nearest cluster voxel.

    This function is partially derived from Nilearn's code.

    License
    -------
    New BSD License

    Copyright (c) 2007 - 2023 The nilearn developers.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    a. Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimer.
    b. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
    c. Neither the name of the nilearn developers nor the names of
        its contributors may be used to endorse or promote products
        derived from this software without specific prior written
        permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
    DAMAGE.
    """
    data_max = maximum_filter(data, 3)
    maxima = data == data_max
    zero_mask = data == 0
    maxima[zero_mask] = 0

    # Don't treat constant patches as maxima unless the entire cluster is
    # constant (as in a binary cluster).
    is_constant = np.isclose(data[~zero_mask].max(), data[~zero_mask].min())
    if not is_constant:
        data_min = minimum_filter(data, 3)
        diff = (data_max - data_min) > 0
        maxima[diff == 0] = 0

    labeled, n_subpeaks = label(maxima)
    labels_index = np.arange(1, n_subpeaks + 1)
    ijk = np.array(center_of_mass(data, labeled, labels_index))
    ijk = np.round(ijk).astype(int)
    # Determine if all subpeaks are within the cluster
    # They may not be if the cluster is binary and has a shape where the COM is
    # outside the cluster, like a donut.
    subpeaks_outside_cluster = labeled[ijk[:, 0], ijk[:, 1], ijk[:, 2]] != labels_index
    if np.any(subpeaks_outside_cluster):
        warnings.warn(
            "Attention: At least one of the (sub)peaks falls outside of the "
            "cluster body. Identifying the nearest in-cluster voxel."
        )
        # Replace centers of mass with their nearest neighbor points in the
        # corresponding clusters. Note this is also equivalent to computing the
        # centers of mass constrained to points within the cluster.
        ijk[subpeaks_outside_cluster] = _cluster_nearest_neighbor(
            ijk[subpeaks_outside_cluster],
            labels_index[subpeaks_outside_cluster],
            labeled,
        )
    vals = data[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    return ijk, vals


def _cluster_nearest_neighbor(ijk, labels_index, labeled):
    """Find the nearest neighbor for given points in the corresponding cluster.

    Parameters
    ----------
    ijk : :obj:`numpy.ndarray`
        (n_pts, 3) array of query points.
    labels_index : :obj:`numpy.ndarray`
        (n_pts,) array of corresponding cluster indices.
    labeled : :obj:`numpy.ndarray`
        3D array with voxels labeled according to cluster index.

    Returns
    -------
    nbrs : :obj:`numpy.ndarray`
        (n_pts, 3) nearest neighbor points.

    This function is partially derived from Nilearn's code.

    License
    -------
    New BSD License

    Copyright (c) 2007 - 2023 The nilearn developers.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    a. Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimer.
    b. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
    c. Neither the name of the nilearn developers nor the names of
        its contributors may be used to endorse or promote products
        derived from this software without specific prior written
        permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
    DAMAGE.
    """
    labels = labeled[labeled > 0]
    clusters_ijk = np.array(labeled.nonzero()).T
    nbrs = np.zeros_like(ijk)
    for ii, (lab, point) in enumerate(zip(labels_index, ijk)):
        lab_ijk = clusters_ijk[labels == lab]
        dist = np.linalg.norm(lab_ijk - point, axis=1)
        nbrs[ii] = lab_ijk[np.argmin(dist)]
    return nbrs


def _sort_subpeaks(ijk, vals, affine):
    """Sort subpeaks in cluster in descending order of stat value.

    Parameters
    ----------
    ijk : 2D numpy.ndarray
        The matrix indices of subpeaks to sort.
    vals : 1D numpy.ndarray
        The statistical value associated with each subpeak in ``ijk``.
    affine : (4x4) numpy.ndarray
        The affine of the img from which the subpeaks were extracted.
        Used to convert IJK indices to XYZ coordinates.

    Returns
    -------
    xyz : 2D numpy.ndarray
        The sorted coordinates of the subpeaks.
    ijk : 2D numpy.ndarray
        The sorted matrix indices of subpeaks.
    vals : 1D numpy.ndarray
        The sorted statistical value associated with each subpeak in ``ijk``.

    This function is partially derived from Nilearn's code.

    License
    -------
    New BSD License

    Copyright (c) 2007 - 2023 The nilearn developers.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    a. Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimer.
    b. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
    c. Neither the name of the nilearn developers nor the names of
        its contributors may be used to endorse or promote products
        derived from this software without specific prior written
        permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
    DAMAGE.
    """
    order = (-vals).argsort()
    vals = vals[order]
    ijk = ijk[order, :]
    xyz = nib.affines.apply_affine(affine, ijk)  # Convert to xyz in mm
    return xyz, ijk, vals


def _pare_subpeaks(xyz, ijk, vals, min_distance):
    """Reduce list of subpeaks based on distance.

    Parameters
    ----------
    xyz : 2D numpy.ndarray
        Subpeak coordinates to reduce. Rows correspond to peaks, columns
        correspond to x, y, and z dimensions.
    ijk : 2D numpy.ndarray
        The subpeak coordinates in ``xyz``, but converted to matrix indices.
    vals : 1D numpy.ndarray
        The statistical value associated with each subpeak in ``xyz``/``ijk``.
    min_distance : float
        The minimum distance between subpeaks, in millimeters.

    Returns
    -------
    ijk : 2D numpy.ndarray
        The reduced index of subpeaks.
    vals : 1D numpy.ndarray
        The statistical values associated with the reduced set of subpeaks.

    This function is partially derived from Nilearn's code.

    License
    -------
    New BSD License

    Copyright (c) 2007 - 2023 The nilearn developers.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    a. Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimer.
    b. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
    c. Neither the name of the nilearn developers nor the names of
        its contributors may be used to endorse or promote products
        derived from this software without specific prior written
        permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
    DAMAGE.
    """
    keep_idx = np.ones(xyz.shape[0]).astype(bool)
    for i in range(xyz.shape[0]):
        for j in range(i + 1, xyz.shape[0]):
            if keep_idx[i] == 1:
                dist = np.linalg.norm(xyz[i, :] - xyz[j, :])
                keep_idx[j] = dist > min_distance
    ijk = ijk[keep_idx, :]
    vals = vals[keep_idx]
    return ijk, vals


def _get_clusters_table(
    stat_img,
    stat_threshold,
    cluster_threshold=None,
    two_sided=False,
    min_distance=8.0,
    return_label_maps=False,
):
    """Creates pandas dataframe with img cluster statistics.

    This function should work on any statistical maps where more extreme values
    indicate greater statistical significance.
    For example, z-statistic or -log10(p) maps are valid inputs, but a p-value
    map is not.

    .. important::

        For binary clusters (clusters comprised of only one value),
        the table reports the center of mass of the cluster,
        rather than any peaks/subpeaks.

        This center of mass may, in some cases, appear outside of the cluster.

        .. versionchanged:: 0.9.2
            In this case, the cluster voxel nearest to the center of mass is
            reported.

    Parameters
    ----------
    stat_img : Niimg-like object
       Statistical image to threshold and summarize.

    stat_threshold : :obj:`float`
        Cluster forming threshold. This value must be in the same scale as
        ``stat_img``.

    cluster_threshold : :obj:`int` or None, optional
        Cluster size threshold, in :term:`voxels<voxel>`.
        If None, then no cluster size threshold will be applied. Default=None.

    two_sided : :obj:`bool`, optional
        Whether to employ two-sided thresholding or to evaluate positive values
        only. Default=False.

    min_distance : :obj:`float`, optional
        Minimum distance between subpeaks, in millimeters. Default=8.

        .. note::
            If two different clusters are closer than ``min_distance``, it can
            result in peaks closer than ``min_distance``.

    return_label_maps : :obj:`bool`, optional
        Whether or not to additionally output cluster label map images.
        Default=False.

        .. versionadded:: 0.10.1.dev

    Returns
    -------
    df : :obj:`pandas.DataFrame`
        Table with peaks and subpeaks from thresholded ``stat_img``.
        The columns in this table include:

        ================== ====================================================
        Cluster ID         The cluster number. Subpeaks have letters after the
                           number.
        X/Y/Z              The coordinate for the peak, in millimeters.
        Peak Stat          The statistical value associated with the peak.
                           The statistic type is dependent on the type of the
                           statistical image.
        Cluster Size (mm3) The size of the cluster, in millimeters cubed.
                           Rows corresponding to subpeaks will not have a value
                           in this column.
        ================== ====================================================

    label_maps : :obj:`list`
        Returned if return_label_maps=True
        List of Niimg-like objects of cluster label maps.
        If two_sided==True, first and second maps correspond
        to positive and negative tails.

        .. versionadded:: 0.10.1.dev

    This function is partially derived from Nilearn's code.

    License
    -------
    New BSD License

    Copyright (c) 2007 - 2023 The nilearn developers.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    a. Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimer.
    b. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
    c. Neither the name of the nilearn developers nor the names of
        its contributors may be used to endorse or promote products
        derived from this software without specific prior written
        permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
    DAMAGE.
    """
    cols = ["Cluster ID", "X", "Y", "Z", "Peak Stat", "Cluster Size (mm3)"]
    # Replace None with 0
    cluster_threshold = 0 if cluster_threshold is None else cluster_threshold

    # check that stat_img is niimg-like object and 3D
    stat_img = check_niimg_3d(stat_img)
    affine = stat_img.affine
    shape = stat_img.shape

    # Apply threshold(s) to image
    stat_img = threshold_img(
        img=stat_img,
        threshold=stat_threshold,
        cluster_threshold=cluster_threshold,
        two_sided=two_sided,
        mask_img=None,
        copy=True,
    )

    # If cluster threshold is used, there is chance that stat_map will be
    # modified, therefore copy is needed
    stat_map = _safe_get_data(
        stat_img,
        ensure_finite=True,
        copy_data=(cluster_threshold is not None),
    )

    # Define array for 6-connectivity, aka NN1 or "faces"
    bin_struct = generate_binary_structure(rank=3, connectivity=1)

    voxel_size = np.prod(stat_img.header.get_zooms())

    signs = [1, -1] if two_sided else [1]
    no_clusters_found = True
    rows = []
    label_maps = []
    for sign in signs:
        # Flip map if necessary
        temp_stat_map = stat_map * sign

        # Binarize using cluster-defining threshold
        binarized = temp_stat_map > stat_threshold
        binarized = binarized.astype(int)

        # If the stat threshold is too high simply return an empty dataframe
        if np.sum(binarized) == 0:
            warnings.warn(
                "Attention: No clusters with stat {0} than {1}".format(
                    "higher" if sign == 1 else "lower",
                    stat_threshold * sign,
                )
            )
            continue

        # Now re-label and create table
        label_map = label(binarized, bin_struct)[0]
        clust_ids = sorted(list(np.unique(label_map)[1:]))
        peak_vals = np.array([np.max(temp_stat_map * (label_map == c)) for c in clust_ids])
        # Sort by descending max value
        clust_ids = [clust_ids[c] for c in (-peak_vals).argsort()]

        if return_label_maps:
            # Relabel label_map based on sorted ids
            relabel_idx = np.insert(clust_ids, 0, 0).argsort().astype(np.int32)
            relabel_map = relabel_idx[label_map.flatten()].reshape(shape)
            # Save label maps as nifti objects
            label_maps.append(new_img_like(stat_img, relabel_map, affine=affine))

        for c_id, c_val in enumerate(clust_ids):
            cluster_mask = label_map == c_val
            masked_data = temp_stat_map * cluster_mask

            cluster_size_mm = int(np.sum(cluster_mask) * voxel_size)

            # Get peaks, subpeaks and associated statistics
            subpeak_ijk, subpeak_vals = _local_max(
                masked_data,
                stat_img.affine,
                min_distance=min_distance,
            )
            subpeak_vals *= sign  # flip signs if necessary
            subpeak_xyz = np.asarray(
                resampling.coord_transform(
                    subpeak_ijk[:, 0],
                    subpeak_ijk[:, 1],
                    subpeak_ijk[:, 2],
                    stat_img.affine,
                )
            ).tolist()
            subpeak_xyz = np.array(subpeak_xyz).T

            # Only report peak and, at most, top 3 subpeaks.
            n_subpeaks = np.min((len(subpeak_vals), 4))
            for subpeak in range(n_subpeaks):
                if subpeak == 0:
                    row = [
                        c_id + 1,
                        subpeak_xyz[subpeak, 0],
                        subpeak_xyz[subpeak, 1],
                        subpeak_xyz[subpeak, 2],
                        subpeak_vals[subpeak],
                        cluster_size_mm,
                    ]
                else:
                    # Subpeak naming convention is cluster num+letter:
                    # 1a, 1b, etc
                    sp_id = "{0}{1}".format(
                        c_id + 1,
                        ascii_lowercase[subpeak - 1],
                    )
                    row = [
                        sp_id,
                        subpeak_xyz[subpeak, 0],
                        subpeak_xyz[subpeak, 1],
                        subpeak_xyz[subpeak, 2],
                        subpeak_vals[subpeak],
                        "",
                    ]
                rows += [row]

        # If we reach this point, there are clusters in this sign
        no_clusters_found = False

    if no_clusters_found:
        df = pd.DataFrame(columns=cols)
    else:
        df = pd.DataFrame(columns=cols, data=rows)

    return (df, label_maps) if return_label_maps else df


def _create_name(resource):
    """Take study/analysis object and try to create dataframe friendly/readable name."""
    return "_".join(resource.name.split()) if resource.name else resource.id


def load_nimads(studyset, annotation=None):
    """Load a studyset object from a dictionary, json file, or studyset object."""
    from nimare.nimads import Studyset

    if isinstance(studyset, dict):
        studyset = Studyset(studyset)
    elif isinstance(studyset, str):
        with open(studyset, "r") as f:
            studyset = Studyset(json.load(f))
    elif isinstance(studyset, Studyset):
        pass
    else:
        raise ValueError(
            "studyset must be: a dictionary, a path to a json file, or studyset object"
        )

    if annotation:
        studyset.annotations = annotation
    return studyset
