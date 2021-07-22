"""Utility functions for NiMARE."""
import datetime
import inspect
import logging
import os
import os.path as op
import re
from functools import wraps
from tempfile import mkstemp

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.input_data import NiftiMasker

from . import references
from .due import due

LGR = logging.getLogger(__name__)


def dict_to_df(id_df, data, key="labels"):
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
            id_ = "{0}-{1}".format(pid, expid)

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


def dict_to_coordinates(data, masker, space):
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
            rep_id = np.array([["{0}-{1}".format(pid, expid), pid, expid]] * n_coords).T

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
    df = transform_coordinates_to_ijk(df, masker, space)
    return df


def transform_coordinates_to_ijk(df, masker, space):
    """Convert xyz coordinates in a DataFrame to ijk indices for a given target space.

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
    masker : :class:`nilearn.input_data.NiftiMasker` or similar
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
        raise ValueError("Unrecognized space: {0}".format(space))

    found_spaces = df["space"].unique()
    for found_space in found_spaces:
        if found_space not in transform.keys():
            LGR.warning(
                "Not applying transforms to coordinates in "
                'unrecognized space "{0}"'.format(found_space)
            )
        alg = transform.get(found_space, None)
        idx = df["space"] == found_space
        if alg:
            df.loc[idx, ["x", "y", "z"]] = alg(df.loc[idx, ["x", "y", "z"]].values)
        df.loc[idx, "space"] = space

    xyz = df[["x", "y", "z"]].values
    ijk = mm2vox(xyz, masker.mask_img.affine)
    df[["i", "j", "k"]] = ijk
    return df


def validate_df(df):
    """Check that an input is a DataFrame and has a column for 'id'."""
    assert isinstance(df, pd.DataFrame)
    assert "id" in df.columns


def validate_images_df(image_df):
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
                "Mix of absolute and relative paths detected "
                "for images in column '{}'".format(col)
            )

    # Set relative paths from absolute ones
    if len(abs_cols):
        all_files = list(np.ravel(image_df[abs_cols].values))
        all_files = [f for f in all_files if isinstance(f, str)]

        if len(all_files) == 1:
            # In the odd case where there's only one absolute path
            shared_path = op.dirname(all_files[0]) + op.sep
        else:
            shared_path = find_stem(all_files)

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


def get_template(space="mni152_2mm", mask=None):
    """Load template file.

    Parameters
    ----------
    space : {'mni152_1mm', 'mni152_2mm', 'ale_2mm'}, optional
        Template to load. Default is 'mni152_1mm'.
    mask : {None, 'brain', 'gm'}, optional
        Whether to return the raw template (None), a brain mask ('brain'), or
        a gray-matter mask ('gm'). Default is None.

    Returns
    -------
    img : :obj:`nibabel.nifti1.Nifti1Image`
        Template image object.
    """
    if space == "mni152_1mm":
        if mask is None:
            img = nib.load(datasets.fetch_icbm152_2009()["t1"])
        elif mask == "brain":
            img = nib.load(datasets.fetch_icbm152_2009()["mask"])
        elif mask == "gm":
            img = datasets.fetch_icbm152_brain_gm_mask(threshold=0.2)
        else:
            raise ValueError("Mask {0} not supported".format(mask))
    elif space == "mni152_2mm":
        if mask is None:
            img = datasets.load_mni152_template()
        elif mask == "brain":
            img = datasets.load_mni152_brain_mask()
        elif mask == "gm":
            # this approach seems to approximate the 0.2 thresholded
            # GM mask pretty well
            temp_img = datasets.load_mni152_template()
            data = temp_img.get_fdata()
            data = data * -1
            data[data != 0] += np.abs(np.min(data))
            data = (data > 1200).astype(int)
            img = nib.Nifti1Image(data, temp_img.affine)
        else:
            raise ValueError("Mask {0} not supported".format(mask))
    elif space == "ale_2mm":
        if mask is None:
            img = datasets.load_mni152_template()
        else:
            # Not the same as the nilearn brain mask, but should correspond to
            # the default "more conservative" MNI152 mask in GingerALE.
            img = nib.load(op.join(get_resource_path(), "templates/MNI152_2x2x2_brainmask.nii"))
    else:
        raise ValueError("Space {0} not supported".format(space))
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
        mask = NiftiMasker(mask)

    if not (hasattr(mask, "transform") and hasattr(mask, "inverse_transform")):
        raise ValueError(
            "mask argument must be a string, a nibabel image, or a Nilearn Masker instance."
        )

    # Fit the masker if needed
    if not hasattr(mask, "mask_img_"):
        mask.fit()

    return mask


def listify(obj):
    """Wrap all non-list or tuple objects in a list.

    This provides a simple way to accept flexible arguments.
    """
    return obj if isinstance(obj, (list, tuple, type(None), np.ndarray)) else [obj]


def round2(ndarray):
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


def get_resource_path():
    """Return the path to general resources, terminated with separator.

    Resources are kept outside package folder in "datasets".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return op.abspath(op.join(op.dirname(__file__), "resources") + op.sep)


def try_prepend(value, prefix):
    """Try to prepend a value to a string with a separator ('/').

    If not a string, will just return the original value.
    """
    if isinstance(value, str):
        return op.join(prefix, value)
    else:
        return value


def find_stem(arr):
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


def uk_to_us(text):
    """Convert UK spellings to US based on a converter.

    .. versionadded:: 0.0.2

    english_spellings.csv: From http://www.tysto.com/uk-us-spelling-list.html

    Parameters
    ----------
    text : :obj:`str`

    Returns
    -------
    text : :obj:`str`
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
    from .extract.utils import _get_dataset_dir

    def inner_function(function):
        @wraps(function)
        def memmap_context(self, *args, **kwargs):
            if hasattr(self, "memory_limit") and self.memory_limit:
                self.memmap_filenames, filenames = [], []
                for i_file in range(n_files):
                    start_time = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
                    dataset_dir = _get_dataset_dir("temporary_files", data_dir=None)
                    _, filename = mkstemp(
                        prefix=self.__class__.__name__, suffix=start_time, dir=dataset_dir
                    )
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
                if (
                    hasattr(self, "memory_limit")
                    and self.memory_limit
                    and os.path.isfile(filename)
                ):
                    for filename in filenames:
                        logger.debug(f"Removing temporary file: {filename}")
                        os.remove(filename)

        return memmap_context

    return inner_function


BYTE = 2
KILOBYTE = BYTE ** 10
BYTE_CONVERSION = {
    "kb": KILOBYTE,
    "mb": KILOBYTE ** 2,
    "gb": KILOBYTE ** 3,
    "tb": KILOBYTE ** 4,
}


def determine_chunk_size(limit, arr, multiplier=1):
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


def safe_transform(imgs, masker, memory_limit="1gb", dtype="auto", memfile=None):
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
    chunk_size = determine_chunk_size(memory_limit, first_img_data)
    map_chunks = [imgs[i : i + chunk_size] for i in range(0, len(imgs), chunk_size)]
    idx = 0
    for map_chunk in map_chunks:
        end_idx = idx + len(map_chunk)
        masked_data[idx:end_idx, :] = masker.transform(map_chunk)
        idx = end_idx

    return masked_data


def add_metadata_to_dataframe(
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
    dataset : :obj:`nimare.dataset.Dataset`
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


def check_type(obj, clss, **kwargs):
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


def vox2mm(ijk, affine):
    """
    Convert matrix subscripts to coordinates.

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
    """
    Convert coordinates to matrix subscripts.

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


@due.dcite(
    references.LANCASTER_TRANSFORM,
    description="Introduces the Lancaster MNI-to-Talairach transform, "
    "as well as its inverse, the Talairach-to-MNI "
    "transform.",
)
@due.dcite(
    references.LANCASTER_TRANSFORM_VALIDATION,
    description="Validates the Lancaster MNI-to-Talairach and Talairach-to-MNI transforms.",
)
def tal2mni(coords):
    """
    Convert coordinates from Talairach space to MNI space.

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


@due.dcite(
    references.LANCASTER_TRANSFORM,
    description="Introduces the Lancaster MNI-to-Talairach transform, "
    "as well as its inverse, the Talairach-to-MNI "
    "transform.",
)
@due.dcite(
    references.LANCASTER_TRANSFORM_VALIDATION,
    description="Validates the Lancaster MNI-to-Talairach and Talairach-to-MNI transforms.",
)
def mni2tal(coords):
    """
    Convert coordinates from MNI space Talairach space.

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


def boolean_unmask(data_array, bool_array):
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
