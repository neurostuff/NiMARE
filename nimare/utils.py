"""
Utilities
"""
import logging
import os.path as op
import re

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.input_data import NiftiMasker

from .transforms import mm2vox, mni2tal, tal2mni

LGR = logging.getLogger(__name__)


def dict_to_df(id_df, data, key="labels"):
    """
    Load a given data type in NIMADS-format dictionary into DataFrame.

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
    """
    Load coordinates in NIMADS-format dictionary into DataFrame.
    """
    # Required columns
    columns = ["id", "study_id", "contrast_id", "x", "y", "z", "space"]
    core_columns = columns[:]  # Used in contrast for loop

    all_dfs = []
    for pid in data.keys():
        for expid in data[pid]["contrasts"].keys():
            if "coords" not in data[pid]["contrasts"][expid].keys():
                continue

            exp_columns = core_columns[:]
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
            for k in list(set(exp["coords"].keys()) - set(columns)):
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
    ijk = pd.DataFrame(mm2vox(xyz, masker.mask_img.affine), columns=["i", "j", "k"])
    df = pd.concat([df, ijk], axis=1)
    return df


def validate_df(df):
    """Check that an input is a DataFrame and has a column for 'id'."""
    assert isinstance(df, pd.DataFrame)
    assert "id" in df.columns


def validate_images_df(image_df):
    """
    Check and update image paths in DataFrame.

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
    file_cols = []
    for col in image_df.columns:
        vals = [v for v in image_df[col].values if isinstance(v, str)]
        fc = any([any([vs in v for vs in valid_suffixes]) for v in vals])
        if fc:
            file_cols.append(col)

    # Clean up image_df
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
                'for images in column "{}"'.format(col)
            )

    # Set relative paths from absolute ones
    if len(abs_cols):
        all_files = list(np.ravel(image_df[abs_cols].values))
        all_files = [f for f in all_files if isinstance(f, str)]
        shared_path = find_stem(all_files)
        # Get parent *directory* if shared path includes common prefix.
        if not shared_path.endswith(op.sep):
            shared_path = op.dirname(shared_path) + op.sep
        LGR.info('Shared path detected: "{0}"'.format(shared_path))
        for abs_col in abs_cols:
            image_df[abs_col + "__relative"] = image_df[abs_col].apply(
                lambda x: x.split(shared_path)[1] if isinstance(x, str) else x
            )
    return image_df


def get_template(space="mni152_2mm", mask=None):
    """
    Load template file.

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
    """
    Get an initialized, fitted nilearn Masker instance from passed argument.

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
            "mask argument must be a string, a nibabel image," " or a Nilearn Masker instance."
        )

    # Fit the masker if needed
    if not hasattr(mask, "mask_img_"):
        mask.fit()

    return mask


def listify(obj):
    """
    Wraps all non-list or tuple objects in a list; provides a simple way
    to accept flexible arguments.
    """
    return obj if isinstance(obj, (list, tuple, type(None), np.ndarray)) else [obj]


def round2(ndarray):
    """
    Numpy rounds X.5 values to nearest even integer. We want to round to the
    nearest integer away from zero.
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
    """
    Returns the path to general resources, terminated with separator. Resources
    are kept outside package folder in "datasets".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return op.abspath(op.join(op.dirname(__file__), "resources") + op.sep)


def try_prepend(value, prefix):
    """
    Try to prepend a value to a string with a separator ('/'). If not a string,
    will just return the original value.
    """
    if isinstance(value, str):
        return op.join(prefix, value)
    else:
        return value


def find_stem(arr):
    """
    Find longest common substring in array of strings.

    From https://www.geeksforgeeks.org/longest-common-substring-array-strings/
    """
    # Determine size of the array
    n = len(arr)

    # Take first word from array
    # as reference
    s = arr[0]
    ll = len(s)

    res = ""
    for i in range(ll):
        for j in range(i + 1, ll + 1):
            # generating all possible substrings of our ref string arr[0] i.e s
            stem = s[i:j]
            k = 1
            for k in range(1, n):
                # Check if the generated stem is common to to all words
                if stem not in arr[k]:
                    break

            # If current substring is present in all strings and its length is
            # greater than current result
            if k + 1 == n and len(res) < len(stem):
                res = stem

    return res


def uk_to_us(text):
    """
    Convert UK spellings to US based on a converter.

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
