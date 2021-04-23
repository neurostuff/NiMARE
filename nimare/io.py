"""Input/Output operations."""
import json
import logging
import re
from collections import Counter
from itertools import groupby
from operator import itemgetter
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .dataset import Dataset
from .extract.utils import _get_dataset_dir

LGR = logging.getLogger(__name__)

DEFAULT_MAP_TYPE_CONVERSION = {
    "T map": "t",
    "variance": "varcope",
    "univariate-beta map": "beta",
    "Z map": "z",
    "p map": "p",
}


def convert_neurosynth_to_dict(text_file, annotations_file=None):
    """Convert Neurosynth database files to a dictionary.

    Parameters
    ----------
    text_file : :obj:`str`
        Text file with Neurosynth's coordinates. Normally named "database.txt".
    annotations_file : :obj:`str` or None, optional
        Optional file with Neurosynth's annotations. Normally named
        "features.txt". Default is None.

    Returns
    -------
    dict_ : :obj:`dict`
        NiMARE-organized dictionary containing experiment information from text
        files.
    """
    dset_df = pd.read_csv(text_file, sep="\t")
    if annotations_file is not None:
        label_df = pd.read_csv(annotations_file, sep="\t", index_col="pmid")
        label_df.index = label_df.index.astype(str)
        labels = label_df.columns
        if not all("__" in label for label in labels):
            labels = {label: "Neurosynth_TFIDF__" + label for label in labels}
        label_df = label_df.rename(columns=labels)
    else:
        label_df = None

    dset_df["id"] = dset_df["id"].astype(str)

    ids = dset_df["id"].unique()
    dict_ = {}
    for sid in ids:
        study_df = dset_df.loc[dset_df["id"] == sid]
        study_dict = {}
        study_dict["metadata"] = {}
        study_dict["metadata"]["authors"] = study_df["authors"].tolist()[0]
        study_dict["metadata"]["journal"] = study_df["journal"].tolist()[0]
        study_dict["metadata"]["year"] = study_df["year"].tolist()[0]
        study_dict["metadata"]["title"] = study_df["title"].tolist()[0]
        study_dict["contrasts"] = {}
        study_dict["contrasts"]["1"] = {}
        study_dict["contrasts"]["1"]["metadata"] = {}
        study_dict["contrasts"]["1"]["metadata"]["authors"] = study_df["authors"].tolist()[0]
        study_dict["contrasts"]["1"]["metadata"]["journal"] = study_df["journal"].tolist()[0]
        study_dict["contrasts"]["1"]["metadata"]["year"] = study_df["year"].tolist()[0]
        study_dict["contrasts"]["1"]["metadata"]["title"] = study_df["title"].tolist()[0]
        study_dict["contrasts"]["1"]["coords"] = {}
        study_dict["contrasts"]["1"]["coords"]["space"] = study_df["space"].tolist()[0]
        study_dict["contrasts"]["1"]["coords"]["x"] = study_df["x"].tolist()
        study_dict["contrasts"]["1"]["coords"]["y"] = study_df["y"].tolist()
        study_dict["contrasts"]["1"]["coords"]["z"] = study_df["z"].tolist()
        if label_df is not None:
            study_dict["contrasts"]["1"]["labels"] = label_df.loc[sid].to_dict()
        dict_[sid] = study_dict

    return dict_


def convert_neurosynth_to_json(text_file, out_file, annotations_file=None):
    """Convert Neurosynth dataset text file to a NiMARE json file.

    Parameters
    ----------
    text_file : :obj:`str`
        Text file with Neurosynth's coordinates. Normally named "database.txt".
    out_file : :obj:`str`
        Output NiMARE-format json file.
    annotations_file : :obj:`str` or None, optional
        Optional file with Neurosynth's annotations. Normally named
        "features.txt". Default is None.
    """
    dict_ = convert_neurosynth_to_dict(text_file, annotations_file)
    with open(out_file, "w") as fo:
        json.dump(dict_, fo, indent=4, sort_keys=True)


def convert_neurosynth_to_dataset(text_file, annotations_file=None, target="mni152_2mm"):
    """Convert Neurosynth database files into NiMARE Dataset.

    Parameters
    ----------
    text_file : :obj:`str`
        Text file with Neurosynth's coordinates. Normally named "database.txt".
    target : {'mni152_2mm', 'ale_2mm'}, optional
        Target template space for coordinates. Default is 'mni152_2mm'.
    annotations_file : :obj:`str` or None, optional
        Optional file with Neurosynth's annotations. Normally named
        "features.txt". Default is None.

    Returns
    -------
    :obj:`nimare.dataset.Dataset`
        Dataset object containing experiment information from text_file.
    """
    dict_ = convert_neurosynth_to_dict(text_file, annotations_file)
    return Dataset(dict_, target=target)


def convert_sleuth_to_dict(text_file):
    """Convert Sleuth text file to a dictionary.

    Parameters
    ----------
    text_file : :obj:`str` or :obj:`list` of :obj:`str`
        Path to Sleuth-format text file.
        More than one text file may be provided.

    Returns
    -------
    :obj:`dict`
        NiMARE-organized dictionary containing experiment information from text
        file.
    """
    if isinstance(text_file, list):
        dict_ = {}
        for tf in text_file:
            temp_dict = convert_sleuth_to_dict(tf)
            for sid in temp_dict.keys():
                if sid in dict_.keys():
                    dict_[sid]["contrasts"] = {
                        **dict_[sid]["contrasts"],
                        **temp_dict[sid]["contrasts"],
                    }
                else:
                    dict_[sid] = temp_dict[sid]
        return dict_

    with open(text_file, "r") as file_object:
        data = file_object.read()

    data = [line.rstrip() for line in re.split("\n\r|\r\n|\n|\r", data)]
    data = [line for line in data if line]
    # First line indicates space. The rest are studies, ns, and coords
    space = data[0].replace(" ", "").replace("//Reference=", "")

    SPACE_OPTS = ["MNI", "TAL", "Talairach"]
    if space not in SPACE_OPTS:
        raise ValueError(
            "Space {0} unknown. Options supported: {1}.".format(space, ", ".join(SPACE_OPTS))
        )

    # Split into experiments
    data = data[1:]
    exp_idx = []
    header_lines = [i for i in range(len(data)) if data[i].startswith("//")]

    # Get contiguous header lines to define contrasts
    ranges = []
    for k, g in groupby(enumerate(header_lines), lambda x: x[0] - x[1]):
        group = list(map(itemgetter(1), g))
        ranges.append((group[0], group[-1]))
        if "Subjects" not in data[group[-1]]:
            raise ValueError(
                "Sample size line missing for {}".format(data[group[0] : group[-1] + 1])
            )
    start_idx = [r[0] for r in ranges]
    end_idx = start_idx[1:] + [len(data) + 1]
    split_idx = zip(start_idx, end_idx)

    dict_ = {}
    for i_exp, exp_idx in enumerate(split_idx):
        exp_data = data[exp_idx[0] : exp_idx[1]]
        if exp_data:
            header_idx = [i for i in range(len(exp_data)) if exp_data[i].startswith("//")]
            study_info_idx = header_idx[:-1]
            n_idx = header_idx[-1]
            study_info = [exp_data[i].replace("//", "").strip() for i in study_info_idx]
            study_info = " ".join(study_info)
            study_name = study_info.split(":")[0]
            contrast_name = ":".join(study_info.split(":")[1:]).strip()
            sample_size = int(exp_data[n_idx].replace(" ", "").replace("//Subjects=", ""))
            xyz = exp_data[n_idx + 1 :]  # Coords are everything after study info and n
            xyz = [row.split() for row in xyz]
            correct_shape = np.all([len(coord) == 3 for coord in xyz])
            if not correct_shape:
                all_shapes = np.unique([len(coord) for coord in xyz]).astype(str)
                raise ValueError(
                    'Coordinates for study "{0}" are not all '
                    "correct length. Lengths detected: "
                    "{1}.".format(study_info, ", ".join(all_shapes))
                )

            try:
                xyz = np.array(xyz, dtype=float)
            except:
                # Prettify xyz before reporting error
                strs = [[str(e) for e in row] for row in xyz]
                lens = [max(map(len, col)) for col in zip(*strs)]
                fmt = "\t".join("{{:{}}}".format(x) for x in lens)
                table = "\n".join([fmt.format(*row) for row in strs])
                raise ValueError(
                    "Conversion to numpy array failed for study "
                    '"{0}". Coords:\n{1}'.format(study_info, table)
                )

            x, y, z = list(xyz[:, 0]), list(xyz[:, 1]), list(xyz[:, 2])

            if study_name not in dict_.keys():
                dict_[study_name] = {"contrasts": {}}
            dict_[study_name]["contrasts"][contrast_name] = {"coords": {}, "metadata": {}}
            dict_[study_name]["contrasts"][contrast_name]["coords"]["space"] = space
            dict_[study_name]["contrasts"][contrast_name]["coords"]["x"] = x
            dict_[study_name]["contrasts"][contrast_name]["coords"]["y"] = y
            dict_[study_name]["contrasts"][contrast_name]["coords"]["z"] = z
            dict_[study_name]["contrasts"][contrast_name]["metadata"]["sample_sizes"] = [
                sample_size
            ]
    return dict_


def convert_sleuth_to_json(text_file, out_file):
    """Convert Sleuth output text file into json.

    Parameters
    ----------
    text_file : :obj:`str` or :obj:`list` of :obj:`str`
        Path to Sleuth-format text file.
        More than one text file may be provided.
    out_file : :obj:`str`
        Path to output json file.
    """
    if not isinstance(text_file, str) and not isinstance(text_file, list):
        raise ValueError(
            'Unsupported type for parameter "text_file": ' "{0}".format(type(text_file))
        )
    dict_ = convert_sleuth_to_dict(text_file)

    with open(out_file, "w") as fo:
        json.dump(dict_, fo, indent=4, sort_keys=True)


def convert_sleuth_to_dataset(text_file, target="ale_2mm"):
    """Convert Sleuth output text file into NiMARE Dataset.

    Parameters
    ----------
    text_file : :obj:`str` or :obj:`list` of :obj:`str`
        Path to Sleuth-format text file.
        More than one text file may be provided.
    target : {'ale_2mm', 'mni152_2mm'}, optional
        Target template space for coordinates. Default is 'ale_2mm'
        (ALE-specific brainmask in MNI152 2mm space).

    Returns
    -------
    :obj:`nimare.dataset.Dataset`
        Dataset object containing experiment information from text_file.
    """
    if not isinstance(text_file, str) and not isinstance(text_file, list):
        raise ValueError(
            'Unsupported type for parameter "text_file": ' "{0}".format(type(text_file))
        )
    dict_ = convert_sleuth_to_dict(text_file)
    return Dataset(dict_, target=target)


def convert_neurovault_to_dataset(
    collection_ids, contrasts, img_dir=None, map_type_conversion=None, **dset_kwargs
):
    """
    Convert a group of NeuroVault collections into a NiMARE Dataset.

    Parameters
    ----------
    collection_ids : :obj:`list` of :obj:`int` or :obj:`dict`
        A list of collections on neurovault specified by their id.
        The collection ids can accessed through the neurovault API
        (i.e., https://neurovault.org/api/collections) or
        their main website (i.e., https://neurovault.org/collections).
        For example, in this URL https://neurovault.org/collections/8836/,
        `8836` is the collection id.
        collection_ids can also be a dictionary whose keys are the informative
        study name and the values are collection ids to give the collections
        more informative names in the dataset.
    contrasts : :obj:`dict`
        Dictionary whose keys represent the name of the contrast in
        the dataset and whose values represent a regular expression that would
        match the names represented in NeuroVault.
        For example, under the ``Name`` column in this URL
        https://neurovault.org/collections/8836/,
        a valid contrast could be "as-Animal", which will be called "animal" in the created
        dataset if the contrasts argument is ``{'animal': "as-Animal"}``.
    img_dir : :obj:`str` or None, optional
        Base path to save all the downloaded images, by default the images
        will be saved to a temporary directory with the prefix "neurovault".
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
    img_dir = Path(_get_dataset_dir("_".join(contrasts.keys()), data_dir=img_dir))

    if map_type_conversion is None:
        map_type_conversion = DEFAULT_MAP_TYPE_CONVERSION

    if not isinstance(collection_ids, dict):
        collection_ids = {nv_coll: nv_coll for nv_coll in collection_ids}

    dataset_dict = {}
    for coll_name, nv_coll in collection_ids.items():

        nv_url = f"https://neurovault.org/api/collections/{nv_coll}/images/?format=json"
        images = requests.get(nv_url).json()
        if "Not found" in images.get("detail", ""):
            raise ValueError(
                f"Collection {nv_coll} not found. "
                "Three likely causes are (1) the collection doesn't exist, "
                "(2) the collection is private, or "
                "(3) the provided ID corresponds to an image instead of a collection."
            )

        dataset_dict[f"study-{coll_name}"] = {"contrasts": {}}
        for contrast_name, contrast_regex in contrasts.items():
            dataset_dict[f"study-{coll_name}"]["contrasts"][contrast_name] = {
                "images": {
                    "beta": None,
                    "t": None,
                    "varcope": None,
                },
                "metadata": {"sample_sizes": None},
            }

            sample_sizes = []
            for img_dict in images["results"]:
                if not (
                    re.match(contrast_regex, img_dict["name"])
                    and img_dict["map_type"] in map_type_conversion
                ):
                    continue

                filename = img_dir / (
                    f"collection-{nv_coll}_id-{img_dict['id']}_" + Path(img_dict["file"]).name
                )

                if not filename.exists():
                    r = requests.get(img_dict["file"])
                    with open(filename, "wb") as f:
                        f.write(r.content)

                (
                    dataset_dict[f"study-{coll_name}"]["contrasts"][contrast_name]["images"][
                        map_type_conversion[img_dict["map_type"]]
                    ]
                ) = filename.as_posix()

                # aggregate sample sizes (should all be the same)
                sample_sizes.append(img_dict["number_of_subjects"])

            # take modal sample size (raise warning if there are multiple values)
            if len(set(sample_sizes)) > 1:
                sample_size = _resolve_sample_size(sample_sizes)
                LGR.warning(
                    (
                        f"Multiple sample sizes were found for neurovault collection: {nv_coll}"
                        f"for contrast: {contrast_name}, sample sizes: {set(sample_sizes)}"
                        f", selecting modal sample size: {sample_size}"
                    )
                )
            else:
                sample_size = sample_sizes[0]
            (
                dataset_dict[f"study-{coll_name}"]["contrasts"][contrast_name]["metadata"][
                    "sample_sizes"
                ]
            ) = [sample_size]

    dataset = Dataset(dataset_dict, **dset_kwargs)

    return dataset


def _resolve_sample_size(sample_sizes):
    """Choose modal sample_size if there are multiple sample_sizes to choose from."""
    sample_size_counts = Counter(sample_sizes)
    if None in sample_size_counts:
        sample_size_counts.pop(None)

    return sample_size_counts.most_common()[0][0]
