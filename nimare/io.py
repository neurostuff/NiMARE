"""
Input/Output operations.
"""
import json
import re
from itertools import groupby
from operator import itemgetter

import numpy as np
import pandas as pd

from .dataset import Dataset


def convert_neurosynth_to_dict(text_file, annotations_file=None):
    """
    Convert Neurosynth database files to a dictionary.

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
    """
    Convert Neurosynth dataset text file to a NiMARE json file.

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
    """
    Convert Neurosynth database files into dictionary and create NiMARE Dataset
    with dictionary.

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
    """
    Convert Sleuth text file to a dictionary.

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
    """
    Convert Sleuth output text file into json.

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
    """
    Convert Sleuth output text file into dictionary and create NiMARE Dataset
    with dictionary.

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
