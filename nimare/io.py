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
from scipy import sparse

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


def convert_neurosynth_to_dict(
    coordinates_file,
    metadata_file,
    annotations_files=None,
    feature_groups=None,
):
    """Convert Neurosynth/NeuroQuery database files to a dictionary.

    .. versionchanged:: 0.0.10

        * Use new format for Neurosynth and NeuroQuery files.

    .. versionchanged:: 0.0.9

        * Support annotations files organized in a dictionary.

    Parameters
    ----------
    coordinates_file : :obj:`str`
        TSV.GZ file with Neurosynth/NeuroQuery's coordinates.
    metadata_file : :obj:`str`
        TSV.GZ file with Neurosynth/NeuroQuery's metadata.
    annotations_files : :obj:`dict`, :obj:`list` of :obj:`dict`, or None, optional
        Optional file(s) with Neurosynth/NeuroQuery's annotations.
        This should consist of a dictionary with two keys: "features" and "vocabulary".
        "features" should have an NPZ file containing a sparse matrix of feature values.
        "vocabulary" should have a TXT file containing labels.
        The vocabulary corresponds to the columns of the feature matrix, while study IDs are
        inferred from the metadata file, which MUST be in the same order as the features matrix.
        Multiple sets of annotations may be provided, in which case "annotations_files" should be
        a list of dictionaries. The appropriate name of each annotation set will be inferred from
        the "features" filename, but this can be overwritten by using the "feature_groups"
        parameter.
        Default is None.
    feature_groups : :obj:`list` of :obj:`str`, or None, optional
        An optional list of names of annotation sets defined in "annotations_files".
        This should only be used if "annotations_files" is used and the users wants to override
        the automatically-extracted annotation set names.
        Default is None.

    Returns
    -------
    dset_dict : :obj:`dict`
        NiMARE-organized dictionary containing experiment information from text files.

    Warning
    -------
    Starting in version 0.0.10, this function operates on the new Neurosynth/NeuroQuery file
    format. Old code using this function **will not work** with the new version.
    """
    coords_df = pd.read_table(coordinates_file)
    metadata_df = pd.read_table(metadata_file)
    assert metadata_df["id"].is_unique, "Metadata file must have one row per ID."

    coords_df["id"] = coords_df["id"].astype(str)
    metadata_df["id"] = metadata_df["id"].astype(str)
    metadata_df = metadata_df.set_index("id", drop=False)
    ids = metadata_df["id"].tolist()

    if "space" not in metadata_df.columns:
        LGR.warning("No 'space' column detected. Defaulting to 'UNKNOWN'.")
        metadata_df["space"] = "UNKNOWN"

    if isinstance(annotations_files, dict):
        annotations_files = [annotations_files]

    if isinstance(feature_groups, str):
        feature_groups = [feature_groups]

    # Load labels into a single DataFrame
    if annotations_files is not None:
        label_dfs = []
        if feature_groups is not None:
            assert len(feature_groups) == len(annotations_files)

        for i_feature_group, annotations_dict in enumerate(annotations_files):
            features_file = annotations_dict["features"]
            vocabulary_file = annotations_dict["vocabulary"]

            vocab = re.findall("vocab-([a-zA-Z0-9]+)_", features_file)[0]
            source = re.findall("source-([a-zA-Z0-9]+)_", features_file)[0]
            value_type = re.findall("type-([a-zA-Z0-9]+)_", features_file)[0]

            if feature_groups is not None:
                feature_group = feature_groups[i_feature_group]
                feature_group = feature_group.rstrip("_") + "__"
            else:
                feature_group = f"{vocab}_{source}_{value_type}__"

            features = sparse.load_npz(features_file).todense()
            vocab = np.loadtxt(vocabulary_file, dtype=str, delimiter="\t")

            labels = [feature_group + label for label in vocab]

            temp_label_df = pd.DataFrame(features, index=ids, columns=labels)
            temp_label_df.index.name = "study_id"

            label_dfs.append(temp_label_df)

        label_df = pd.concat(label_dfs, axis=1)
    else:
        label_df = None

    # Compile (pseudo-)NIMADS-format dictionary
    dset_dict = {}
    for sid, study_metadata in metadata_df.iterrows():
        study_coords_df = coords_df.loc[coords_df["id"] == sid]
        study_dict = {}
        study_dict["metadata"] = {}
        study_dict["metadata"]["authors"] = study_metadata.get("authors", "n/a")
        study_dict["metadata"]["journal"] = study_metadata.get("journal", "n/a")
        study_dict["metadata"]["year"] = study_metadata.get("year", "n/a")
        study_dict["metadata"]["title"] = study_metadata.get("title", "n/a")
        study_dict["contrasts"] = {}
        study_dict["contrasts"]["1"] = {}
        # Duplicate metadata across study and contrast levels
        study_dict["contrasts"]["1"]["metadata"] = {}
        study_dict["contrasts"]["1"]["metadata"]["authors"] = study_metadata.get("authors", "n/a")
        study_dict["contrasts"]["1"]["metadata"]["journal"] = study_metadata.get("journal", "n/a")
        study_dict["contrasts"]["1"]["metadata"]["year"] = study_metadata.get("year", "n/a")
        study_dict["contrasts"]["1"]["metadata"]["title"] = study_metadata.get("title", "n/a")
        study_dict["contrasts"]["1"]["coords"] = {}
        study_dict["contrasts"]["1"]["coords"]["space"] = study_metadata["space"]
        study_dict["contrasts"]["1"]["coords"]["x"] = study_coords_df["x"].tolist()
        study_dict["contrasts"]["1"]["coords"]["y"] = study_coords_df["y"].tolist()
        study_dict["contrasts"]["1"]["coords"]["z"] = study_coords_df["z"].tolist()

        if label_df is not None:
            study_dict["contrasts"]["1"]["labels"] = label_df.loc[sid].to_dict()

        dset_dict[sid] = study_dict

    return dset_dict


def convert_neurosynth_to_json(
    coordinates_file,
    metadata_file,
    out_file,
    annotations_files=None,
    feature_groups=None,
):
    """Convert Neurosynth/NeuroQuery dataset text file to a NiMARE json file.

    .. versionchanged:: 0.0.10

        * Use new format for Neurosynth and NeuroQuery files.

    .. versionchanged:: 0.0.9

        * Support annotations files organized in a dictionary.

    Parameters
    ----------
    coordinates_file : :obj:`str`
        TSV.GZ file with Neurosynth/NeuroQuery's coordinates and metadata.
    metadata_file : :obj:`str`
        TSV.GZ file with Neurosynth/NeuroQuery's metadata.
    out_file : :obj:`str`
        Output NiMARE-format json file.
    annotations_files : :obj:`dict`, :obj:`list` of :obj:`dict`, or None, optional
        Optional file(s) with Neurosynth/NeuroQuery's annotations.
        This should consist of a dictionary with two keys: "features" and "vocabulary".
        "features" should have an NPZ file containing a sparse matrix of feature values.
        "vocabulary" should have a TXT file containing labels.
        The vocabulary corresponds to the columns of the feature matrix, while study IDs are
        inferred from the metadata file, which MUST be in the same order as the features matrix.
        Multiple sets of annotations may be provided, in which case "annotations_files" should be
        a list of dictionaries. The appropriate name of each annotation set will be inferred from
        the "features" filename, but this can be overwritten by using the "feature_groups"
        parameter.
        Default is None.
    feature_groups : :obj:`list` of :obj:`str`, or None, optional
        An optional list of names of annotation sets defined in "annotations_files".
        This should only be used if "annotations_files" is used and the users wants to override
        the automatically-extracted annotation set names.
        Default is None.

    Warning
    -------
    Starting in version 0.0.10, this function operates on the new Neurosynth/NeuroQuery file
    format. Old code using this function **will not work** with the new version.
    """
    dset_dict = convert_neurosynth_to_dict(
        coordinates_file, metadata_file, annotations_files, feature_groups
    )
    with open(out_file, "w") as fo:
        json.dump(dset_dict, fo, indent=4, sort_keys=True)


def convert_neurosynth_to_dataset(
    coordinates_file,
    metadata_file,
    annotations_files=None,
    feature_groups=None,
    target="mni152_2mm",
):
    """Convert Neurosynth/NeuroQuery database files into NiMARE Dataset.

    .. versionchanged:: 0.0.10

        * Use new format for Neurosynth and NeuroQuery files.

    .. versionchanged:: 0.0.9

        * Support annotations files organized in a dictionary.

    Parameters
    ----------
    coordinates_file : :obj:`str`
        TSV.GZ file with Neurosynth/NeuroQuery's coordinates and metadata.
    metadata_file : :obj:`str`
        TSV.GZ file with Neurosynth/NeuroQuery's metadata.
    annotations_files : :obj:`dict`, :obj:`list` of :obj:`dict`, or None, optional
        Optional file(s) with Neurosynth/NeuroQuery's annotations.
        This should consist of a dictionary with two keys: "features" and "vocabulary".
        "features" should have an NPZ file containing a sparse matrix of feature values.
        "vocabulary" should have a TXT file containing labels.
        The vocabulary corresponds to the columns of the feature matrix, while study IDs are
        inferred from the metadata file, which MUST be in the same order as the features matrix.
        Multiple sets of annotations may be provided, in which case "annotations_files" should be
        a list of dictionaries. The appropriate name of each annotation set will be inferred from
        the "features" filename, but this can be overwritten by using the "feature_groups"
        parameter.
        Default is None.
    feature_groups : :obj:`list` of :obj:`str`, or None, optional
        An optional list of names of annotation sets defined in "annotations_files".
        This should only be used if "annotations_files" is used and the users wants to override
        the automatically-extracted annotation set names.
        Default is None.
    target : {'mni152_2mm', 'ale_2mm'}, optional
        Target template space for coordinates. Default is 'mni152_2mm'.

    Returns
    -------
    :obj:`~nimare.dataset.Dataset`
        Dataset object containing experiment information from text_file.

    Warning
    -------
    Starting in version 0.0.10, this function operates on the new Neurosynth/NeuroQuery file
    format. Old code using this function **will not work** with the new version.
    """
    dset_dict = convert_neurosynth_to_dict(
        coordinates_file,
        metadata_file,
        annotations_files,
        feature_groups,
    )
    return Dataset(dset_dict, target=target)


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
        NiMARE-organized dictionary containing experiment information from text file.
    """
    if isinstance(text_file, list):
        dset_dict = {}
        for tf in text_file:
            temp_dict = convert_sleuth_to_dict(tf)
            for sid in temp_dict.keys():
                if sid in dset_dict.keys():
                    dset_dict[sid]["contrasts"] = {
                        **dset_dict[sid]["contrasts"],
                        **temp_dict[sid]["contrasts"],
                    }
                else:
                    dset_dict[sid] = temp_dict[sid]
        return dset_dict

    with open(text_file, "r") as file_object:
        data = file_object.read()

    data = [line.rstrip() for line in re.split("\n\r|\r\n|\n|\r", data)]
    data = [line for line in data if line]
    # First line indicates space. The rest are studies, ns, and coords
    space = data[0].replace(" ", "").replace("//Reference=", "")

    SPACE_OPTS = ["MNI", "TAL", "Talairach"]
    if space not in SPACE_OPTS:
        raise ValueError(f"Space {space} unknown. Options supported: {', '.join(SPACE_OPTS)}.")

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
            raise ValueError(f"Sample size line missing for {data[group[0] : group[-1] + 1]}")
    start_idx = [r[0] for r in ranges]
    end_idx = start_idx[1:] + [len(data) + 1]
    split_idx = zip(start_idx, end_idx)

    dset_dict = {}
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
                    f"Coordinates for study '{study_info}' are not all "
                    f"correct length. Lengths detected: {', '.join(all_shapes)}."
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
                    f"Conversion to numpy array failed for study '{study_info}'. Coords:\n{table}"
                )

            x, y, z = list(xyz[:, 0]), list(xyz[:, 1]), list(xyz[:, 2])

            if study_name not in dset_dict.keys():
                dset_dict[study_name] = {"contrasts": {}}
            dset_dict[study_name]["contrasts"][contrast_name] = {"coords": {}, "metadata": {}}
            dset_dict[study_name]["contrasts"][contrast_name]["coords"]["space"] = space
            dset_dict[study_name]["contrasts"][contrast_name]["coords"]["x"] = x
            dset_dict[study_name]["contrasts"][contrast_name]["coords"]["y"] = y
            dset_dict[study_name]["contrasts"][contrast_name]["coords"]["z"] = z
            dset_dict[study_name]["contrasts"][contrast_name]["metadata"]["sample_sizes"] = [
                sample_size
            ]
    return dset_dict


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
        raise ValueError(f"Unsupported type for parameter 'text_file': {type(text_file)}")
    dset_dict = convert_sleuth_to_dict(text_file)

    with open(out_file, "w") as fo:
        json.dump(dset_dict, fo, indent=4, sort_keys=True)


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
    :obj:`~nimare.dataset.Dataset`
        Dataset object containing experiment information from text_file.
    """
    if not isinstance(text_file, str) and not isinstance(text_file, list):
        raise ValueError(f"Unsupported type for parameter 'text_file': {type(text_file)}")
    dset_dict = convert_sleuth_to_dict(text_file)
    return Dataset(dset_dict, target=target)


def convert_neurovault_to_dataset(
    collection_ids,
    contrasts,
    img_dir=None,
    map_type_conversion=None,
    **dset_kwargs,
):
    """Convert a group of NeuroVault collections into a NiMARE Dataset.

    .. versionadded:: 0.0.8

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
        see :obj:`~nimare.dataset.Dataset` for details.

    Returns
    -------
    :obj:`~nimare.dataset.Dataset`
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
            no_images = True
            for img_dict in images["results"]:
                if not (
                    re.match(contrast_regex, img_dict["name"])
                    and img_dict["map_type"] in map_type_conversion
                    and img_dict["analysis_level"] == "group"
                ):
                    continue

                no_images = False
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

            if no_images:
                raise ValueError(
                    f"No images were found for contrast {contrast_name}. "
                    f"Please check the contrast regular expression: {contrast_regex}"
                )
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
