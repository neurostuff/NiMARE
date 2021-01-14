"""Utility functions for the extract module."""
from __future__ import division

import logging
import os
import os.path as op

import numpy as np
import pandas as pd
import requests
from fuzzywuzzy import fuzz

from ..utils import uk_to_us

LGR = logging.getLogger(__name__)


def get_data_dirs(data_dir=None):
    """Return the directories in which NiMARE looks for data.

    This is typically useful for the end-user to check where the data is
    downloaded and stored.

    Parameters
    ----------
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    Returns
    -------
    paths: list of strings
        Paths of the dataset directories.

    Notes
    -----
    Taken from Nilearn.
    This function retrieves the datasets directories using the following
    priority :
    1. defaults system paths
    2. the keyword argument data_dir
    3. the global environment variable NIMARE_SHARED_DATA
    4. the user environment variable NIMARE_DATA
    5. nimare_data in the user home folder
    """
    # We build an array of successive paths by priority
    # The boolean indicates if it is a pre_dir: in that case, we won't add the
    # dataset name to the path.
    paths = []

    # Check data_dir which force storage in a specific location
    if data_dir is not None:
        paths.extend(data_dir.split(os.pathsep))

    # If data_dir has not been specified, then we crawl default locations
    if data_dir is None:
        global_data = os.getenv("NIMARE_SHARED_DATA")
        if global_data is not None:
            paths.extend(global_data.split(os.pathsep))

        local_data = os.getenv("NIMARE_DATA")
        if local_data is not None:
            paths.extend(local_data.split(os.pathsep))

        paths.append(os.path.expanduser("~/.nimare"))
    return paths


def _get_dataset_dir(dataset_name, data_dir=None, default_paths=None, verbose=1):
    """Create if necessary and returns data directory of given dataset.

    Parameters
    ----------
    dataset_name: string
        The unique name of the dataset.
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None
    default_paths: list of string, optional
        Default system paths in which the dataset may already have been
        installed by a third party software. They will be checked first.
    verbose: int, optional
        verbosity level (0 means no message).

    Returns
    -------
    data_dir: string
        Path of the given dataset directory.

    Notes
    -----
    Taken from Nilearn.
    This function retrieves the datasets directory (or data directory) using
    the following priority :
    1. defaults system paths
    2. the keyword argument data_dir
    3. the global environment variable NIMARE_SHARED_DATA
    4. the user environment variable NIMARE_DATA
    5. nimare_data in the user home folder
    """
    paths = []
    # Search possible data-specific system paths
    if default_paths is not None:
        for default_path in default_paths:
            paths.extend([(d, True) for d in default_path.split(os.pathsep)])

    paths.extend([(d, False) for d in get_data_dirs(data_dir=data_dir)])

    if verbose > 2:
        print("Dataset search paths: %s" % paths)

    # Check if the dataset exists somewhere
    for path, is_pre_dir in paths:
        if not is_pre_dir:
            path = os.path.join(path, dataset_name)
        if os.path.islink(path):
            # Resolve path
            path = readlinkabs(path)
        if os.path.exists(path) and os.path.isdir(path):
            if verbose > 1:
                print("\nDataset found in %s\n" % path)
            return path

    # If not, create a folder in the first writeable directory
    errors = []
    for (path, is_pre_dir) in paths:
        if not is_pre_dir:
            path = os.path.join(path, dataset_name)
        if not os.path.exists(path):
            try:
                os.makedirs(path)
                if verbose > 0:
                    print("\nDataset created in %s\n" % path)
                return path
            except Exception as exc:
                short_error_message = getattr(exc, "strerror", str(exc))
                errors.append("\n -{0} ({1})".format(path, short_error_message))

    raise OSError(
        "NiMARE tried to store the dataset in the following directories, but: " + "".join(errors)
    )


def readlinkabs(link):
    """Return an absolute path for the destination of a symlink.

    From nilearn.
    """
    path = os.readlink(link)
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(link), path)


def _download_zipped_file(url, filename=None):
    """Download from a URL to a file."""
    if filename is None:
        data_dir = op.abspath(op.getcwd())
        filename = op.join(data_dir, url.split("/")[-1])
    # NOTE the stream=True parameter
    req = requests.get(url, stream=True)
    with open(filename, "wb") as f_obj:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f_obj.write(chunk)
    return filename


def _longify(df):
    """Expand comma-separated lists of aliases in DataFrame into separate rows."""
    reduced = df[["id", "name", "alias"]]
    rows = []
    for index, row in reduced.iterrows():
        if isinstance(row["alias"], str) and "," in row["alias"]:
            aliases = row["alias"].split(", ") + [row["name"]]
        else:
            aliases = [row["name"]]

        for alias in aliases:
            rows.append([row["id"], row["name"].lower(), alias.lower()])
    out_df = pd.DataFrame(columns=["id", "name", "alias"], data=rows)
    out_df = out_df.replace("", np.nan)
    return out_df


def _get_ratio(tup):
    """Get fuzzy ratio."""
    if all(isinstance(t, str) for t in tup):
        return fuzz.ratio(tup[0], tup[1])
    else:
        return 100


def _gen_alt_forms(term):
    """Generate a list of alternate forms for a given term."""
    if not isinstance(term, str) or len(term) == 0:
        return [None]

    alt_forms = []
    # For one alternate form, put contents of parentheses at beginning of term
    if "(" in term:
        prefix = term[term.find("(") + 1 : term.find(")")]
        temp_term = term.replace("({0})".format(prefix), "").replace("  ", " ")
        alt_forms.append(temp_term)
        alt_forms.append("{0} {1}".format(prefix, temp_term))
    else:
        prefix = ""

    # Remove extra spaces
    alt_forms = [s.strip() for s in alt_forms]

    # Allow plurals
    # temp = [s+'s' for s in alt_forms]
    # temp += [s+'es' for s in alt_forms]
    # alt_forms += temp

    # Remove words "task" and/or "paradigm"
    alt_forms += [term.replace(" task", "") for term in alt_forms]
    alt_forms += [term.replace(" paradigm", "") for term in alt_forms]

    # Remove duplicates
    alt_forms = list(set(alt_forms))
    return alt_forms


def _get_concept_reltype(relationship, direction):
    """Convert two-part relationship info to more parsimonious representation.

    The two part representation includes relationship type and direction.
    """
    new_rel = None
    if relationship == "PARTOF":
        if direction == "child":
            new_rel = "hasPart"
        elif direction == "parent":
            new_rel = "isPartOf"
    elif relationship == "KINDOF":
        if direction == "child":
            new_rel = "hasKind"
        elif direction == "parent":
            new_rel = "isKindOf"
    return new_rel


def _expand_df(df):
    """Add alternate forms to DataFrame, then sort DataFrame by alias length and similarity.

    Sorting by alias length is done for order of extraction from text. Sorting by similarity to
    original name is done in order to select most appropriate term to associate with alias.
    """
    df = df.copy()
    df["alias"] = df["alias"].apply(uk_to_us)
    new_rows = []
    for index, row in df.iterrows():
        alias = row["alias"]
        alt_forms = _gen_alt_forms(alias)
        for alt_form in alt_forms:
            temp_row = row.copy()
            temp_row["alias"] = alt_form
            new_rows.append(temp_row.tolist())
    alt_df = pd.DataFrame(columns=df.columns, data=new_rows)
    df = pd.concat((df, alt_df), axis=0)
    # Sort by name length and similarity of alternate form to preferred term
    # For example, "task switching" the concept should take priority over the
    # "task switching" version of the "task-switching" task.
    df["length"] = df["alias"].str.len()
    df["ratio"] = df[["alias", "name"]].apply(_get_ratio, axis=1)
    df = df.sort_values(by=["length", "ratio"], ascending=[False, False])
    return df
