"""Input/Output operations."""

import json
import logging
import re
from collections import Counter, defaultdict
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
import requests
from scipy import sparse

from nimare.dataset import Dataset
from nimare.exceptions import InvalidStudysetError
from nimare.extract.utils import _get_dataset_dir
from nimare.utils import load_nimads, mni2tal, tal2mni

LGR = logging.getLogger(__name__)

DEFAULT_MAP_TYPE_CONVERSION = {
    "T map": "t",
    "variance": "varcope",
    "univariate-beta map": "beta",
    "Z map": "z",
    "p map": "p",
}


def convert_nimads_to_dataset(studyset, annotation=None):
    """Convert nimads studyset to a dataset.

    .. versionadded:: 0.0.14

    Parameters
    ----------
    studyset : :obj:`str`, :obj:`dict`, :obj:`nimare.nimads.StudySet`
        Path to a JSON file containing a nimads studyset, a dictionary containing a nimads
        studyset, or a nimads studyset object.
    annotation : :obj:`str`, :obj:`dict`, :obj:`nimare.nimads.Annotation`, optional
        Optional path to a JSON file containing a nimads annotation, a dictionary containing a
        nimads annotation, or a nimads annotation object.

    Returns
    -------
    dset : :obj:`nimare.dataset.Dataset`
        NiMARE Dataset object containing experiment information from nimads studyset.
    """

    def _analysis_to_dict(study, analysis):
        study_name = study.name or study.id
        analysis_name = analysis.name or analysis.id

        result = {
            "metadata": {
                "authors": study.authors,
                "journal": study.publication,
                "study_name": study_name,
                "analysis_name": analysis_name,
                "name": f"{study_name}-{analysis_name}",
            },
            "coords": {
                "space": analysis.points[0].space if analysis.points else "UNKNOWN",
                "x": [p.x for p in analysis.points] or [None],
                "y": [p.y for p in analysis.points] or [None],
                "z": [p.z for p in analysis.points] or [None],
            },
        }

        sample_sizes = analysis.metadata.get("sample_sizes", None) or study.metadata.get(
            "sample_sizes", None
        )
        sample_size = analysis.metadata.get("sample_size", None) or study.metadata.get(
            "sample_size", None
        )

        # Validate sample sizes if present
        if sample_sizes is not None and not isinstance(sample_sizes, (list, tuple)):
            LGR.warning(
                f"Expected sample_sizes to be list or tuple, but got {type(sample_sizes)}."
            )
            sample_sizes = None
        elif sample_sizes is not None:
            # Validate each sample size in the list
            for i, ss in enumerate(sample_sizes):
                if not isinstance(ss, (int, float)):
                    LGR.warning(
                        f"Expected sample_sizes[{i}] to be numeric, but got {type(ss)}."
                        " Attempting to convert to numeric."
                    )
                try:
                    sample_sizes[i] = int(ss)
                except (ValueError, TypeError):
                    try:
                        sample_sizes[i] = float(ss)
                    except (ValueError, TypeError):
                        LGR.warning(f"Could not convert {ss} to numeric from type {type(ss)}.")
                        sample_sizes = None
                        break

        if not sample_sizes and sample_size:
            # Validate single sample size if present
            if not isinstance(sample_size, (int, float)):
                LGR.warning(
                    f"Expected sample_size to be numeric, but got {type(sample_size)}."
                    " Attempting to convert to numeric."
                )
            try:
                sample_sizes = [int(sample_size)]
            except (ValueError, TypeError):
                try:
                    sample_sizes = [float(sample_size)]
                except (ValueError, TypeError):
                    LGR.warning(
                        f"Could not convert {sample_size} to"
                        f" numeric from type {type(sample_size)}."
                    )
                    sample_sizes = None
        if sample_sizes:
            result["metadata"]["sample_sizes"] = sample_sizes

        # Handle annotations if present
        if analysis.annotations:
            result["labels"] = {}
            try:
                for annotation in analysis.annotations.values():
                    if not isinstance(annotation, dict):
                        raise TypeError(
                            f"Expected annotation to be dict, but got {type(annotation)}"
                        )
                    result["labels"].update(annotation)
            except (TypeError, AttributeError) as e:
                raise ValueError(f"Invalid annotation format: {str(e)}") from e

        return result

    def _study_to_dict(study):
        study_name = study.name or study.id
        return {
            "metadata": {
                "authors": study.authors,
                "journal": study.publication,
                "title": study_name,
                "study_name": study_name,
            },
            "contrasts": {a.id: _analysis_to_dict(study, a) for a in study.analyses},
        }

    # load nimads studyset
    studyset = load_nimads(studyset, annotation)
    return Dataset({s.id: _study_to_dict(s) for s in list(studyset.studies)})


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

    Warnings
    --------
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
    x = coords_df["x"].values
    y = coords_df["y"].values
    z = coords_df["z"].values

    dset_dict = {}

    for sid, study_metadata in metadata_df.iterrows():
        coord_inds = np.where(coords_df["id"].values == sid)[0]
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
        study_dict["contrasts"]["1"]["coords"]["x"] = list(x[coord_inds])
        study_dict["contrasts"]["1"]["coords"]["y"] = list(y[coord_inds])
        study_dict["contrasts"]["1"]["coords"]["z"] = list(z[coord_inds])

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

    Warnings
    --------
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

    Warnings
    --------
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


def convert_nimads_to_sleuth(
    studyset: Union[str, Dict[str, Any], Any],
    output_dir: Union[str, Path],
    target: Union[Literal["MNI", "TAL"], str] = "MNI",
    decimal_precision: Optional[int] = 2,
    annotation: Optional[Union[str, Dict[str, Any], Any]] = None,
    export_metadata: Optional[
        List[Literal["authors", "publication", "year", "doi", "pmid", "affiliations"]]
    ] = None,
    *,
    annotation_columns: Optional[Sequence[str]] = None,
    annotation_values: Optional[Dict[str, Sequence[Any]]] = None,
    split_booleans: Optional[bool] = True,
):
    """Convert a NIMADS Studyset to Sleuth text file(s).

    Parameters
    ----------
    studyset : :obj:`str` or :obj:`dict` or :obj:`nimare.nimads.Studyset`
        Path to a Studyset JSON, a Studyset-like dictionary, or a Studyset object.
    output_dir : :obj:`str` or :obj:`pathlib.Path`
        Directory for Sleuth output. Created if missing.
    target : str, optional
        Target space for coordinates. Accepts "MNI" or "TAL", but also commonly-used
        dataset target strings such as "ale_2mm" or "mni152_2mm". These are normalized
        to either "MNI" or "TAL" internally. Default is "MNI".
    decimal_precision : :obj:`int`, optional
        Decimal places for output coordinates. Default is 2.
    annotation : :obj:`str` or :obj:`dict` or :obj:`nimare.nimads.Annotation`, optional
        Optional annotation to split output files by annotation values.
    export_metadata : :obj:`list` of {"authors", "publication", "year", "doi",
        "pmid", "affiliations"}, optional
        Metadata fields to export as comments. Default is ["doi", "pmid"].
    annotation_columns : :obj:`list` of :obj:`str`, optional
        Restrict splitting to these annotation keys.
    annotation_values : :obj:`dict`, optional
        Filter annotation splits to these values per key.
    split_booleans : :obj:`bool`, optional
        When True, boolean columns produce ``_true``/``_false`` files. Default is True.

    Raises
    ------
    InvalidStudysetError
        On invalid inputs or schema/structure mismatch.
    """
    if export_metadata is None:
        export_metadata = ["doi", "pmid"]
    if decimal_precision < 0:
        raise InvalidStudysetError("decimal_precision must be non-negative")

    # Normalize target string to canonical Sleuth reference-space ("MNI" or "TAL")
    try:
        tgt_str = str(target).lower()
    except Exception:
        tgt_str = "mni"
    if tgt_str in ("mni", "mni152_2mm", "ale_2mm", "mni152"):
        target_norm = "MNI"
    elif "tal" in tgt_str:
        target_norm = "TAL"
    else:
        # Fallback to MNI for unknown values
        target_norm = "MNI"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load studyset object and dict for validation
    try:
        # Local import to avoid circular import at module import time.
        from nimare.nimads import Studyset as _Studyset

        studyset_obj = studyset if isinstance(studyset, _Studyset) else _Studyset(studyset)
    except Exception as e:
        raise InvalidStudysetError(f"Failed to load studyset: {e}") from e

    if annotation is not None:
        try:
            studyset_obj.annotations = annotation
        except Exception as e:
            raise InvalidStudysetError(f"Failed to load annotation: {e}") from e

    # Handle annotations
    if studyset_obj.annotations:
        # Process with annotations - create separate files
        _process_with_annotations(
            studyset_obj,
            output_dir,
            target_norm,
            decimal_precision,
            export_metadata,
            annotation_columns,
            annotation_values,
            split_booleans,
        )
    else:
        # Process without annotations - single file
        _process_without_annotations(
            studyset_obj,
            output_dir,
            target_norm,
            decimal_precision,
            export_metadata,
        )


def _write_analysis(
    f,
    study,
    analysis,
    target: str,
    decimal_precision: int,
    export_metadata,
):
    """Write a single analysis block (header, metadata, coordinates) to an open file handle.

    This centralizes the previously duplicated code for writing study/analysis headers,
    sample sizes, metadata comment lines, and coordinates (including named-space transforms).
    """
    # Validate points
    if not hasattr(analysis, "points"):
        raise InvalidStudysetError(f"Analysis {analysis.id} is missing points")
    if not analysis.points:
        return

    for point in analysis.points:
        if not (hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z")):
            raise InvalidStudysetError(f"Point in analysis {analysis.id} is missing coordinates")

    # Generate hierarchical label with metadata
    study_name = study.name or f"Study_{study.id}"
    analysis_name = analysis.name or f"Analysis_{analysis.id}"

    if getattr(study, "authors", ""):
        label = f"{study.authors} ({study_name}): {analysis_name}"
    else:
        label = f"{study_name}: {analysis_name}"

    # Write study header
    f.write(f"//{label}\n")

    # Write sample size if available
    sample_size = _get_sample_size(analysis, study)
    if sample_size:
        f.write(f"//Subjects={sample_size}\n")

    # Write additional metadata as comments
    metadata_lines = _generate_metadata_comments(study, export_metadata)
    for line in metadata_lines:
        f.write(f"//{line}\n")

    # Transform and write coordinates
    coords = []
    source_space = None
    for point in analysis.points:
        if source_space is None:
            source_space = getattr(point, "space", None) or "UNKNOWN"
        coords.append([point.x, point.y, point.z])

    if coords:
        coords = np.array(coords)
        # Validate coordinates are numeric
        if not np.issubdtype(coords.dtype, np.number):
            raise InvalidStudysetError("Coordinates must be numeric")
        # Apply named-space normalization if requested
        if target == "MNI" and source_space == "TAL":
            coords = tal2mni(coords)
        elif target == "TAL" and source_space == "MNI":
            coords = mni2tal(coords)

        # Write coordinates with specified precision
        for coord in coords:
            f.write(
                f"{coord[0]:.{decimal_precision}f}\t"
                f"{coord[1]:.{decimal_precision}f}\t"
                f"{coord[2]:.{decimal_precision}f}\n"
            )

    f.write("\n")  # Empty line between analyses


def _process_without_annotations(
    studyset,
    output_dir,
    target,
    decimal_precision,
    export_metadata,
):
    """Process studyset without annotations (single Sleuth file)."""
    output_file = output_dir / "nimads_sleuth_file.txt"

    with open(output_file, "w") as f:
        f.write(f"//Reference={target}\n")

        # If no studies, leave just the header
        if not studyset.studies:
            return

        # Iterate deterministically and use helper to write each analysis
        for study in sorted(studyset.studies, key=lambda s: s.id):
            if not hasattr(study, "analyses"):
                raise InvalidStudysetError(f"Study {study.id} is missing analyses")

            for analysis in sorted(study.analyses, key=lambda a: a.id):
                if not hasattr(analysis, "points"):
                    raise InvalidStudysetError(f"Analysis {analysis.id} is missing points")
                if not analysis.points:
                    continue

                _write_analysis(
                    f,
                    study,
                    analysis,
                    target=target,
                    decimal_precision=decimal_precision,
                    export_metadata=export_metadata,
                )


def _process_with_annotations(
    studyset,
    output_dir,
    target,
    decimal_precision,
    export_metadata,
    annotation_columns,
    annotation_values,
    split_booleans,
):
    """Process studyset with annotations, creating multiple files."""
    annotation = studyset.annotations[0]  # Use first annotation

    # Validate annotation
    if not hasattr(annotation, "notes"):
        raise InvalidStudysetError("Annotation is missing notes")

    # Group analyses by annotation values
    analysis_groups = defaultdict(list)

    # Process each analysis and group by annotation values
    for study in sorted(studyset.studies, key=lambda s: s.id):
        # Validate study
        if not hasattr(study, "analyses"):
            raise InvalidStudysetError(f"Study {study.id} is missing analyses")

        for analysis in sorted(study.analyses, key=lambda a: a.id):
            # Validate analysis
            if not hasattr(analysis, "points"):
                raise InvalidStudysetError(f"Analysis {analysis.id} is missing points")

            # Lookup this annotation's note (annotation.id -> note dict) on the analysis
            note = None
            if getattr(analysis, "annotations", None):
                note = analysis.annotations.get(annotation.id)

            if note:
                # Group by annotation values (restricted and filtered if requested)
                keyvals = list(note.items())
                if annotation_columns is not None:
                    keyvals = [(k, v) for k, v in keyvals if k in set(annotation_columns)]
                for key, value in sorted(keyvals, key=lambda kv: (kv[0], str(kv[1]))):
                    if (
                        annotation_values
                        and key in annotation_values
                        and value not in annotation_values[key]
                    ):
                        continue
                    if isinstance(value, bool) and split_booleans:
                        group_name = f"{key}_{str(value).lower()}"
                    else:
                        group_name = f"{key}_{value}"
                    analysis_groups[group_name].append((study, analysis))
            else:
                # Add to default group if no annotation for this analysis
                analysis_groups["default"].append((study, analysis))

    # Create separate files for each group (sorted deterministically)
    for group_name in sorted(analysis_groups.keys()):
        analyses = analysis_groups[group_name]
        # Sanitize filename
        safe_name = "".join(c for c in group_name if c.isalnum() or c in "._-").rstrip()
        if not safe_name:
            safe_name = "unnamed_group"

        output_file = output_dir / f"{safe_name}.txt"

        with open(output_file, "w") as f:
            f.write(f"//Reference={target}\n")

            # Process each analysis in this group
            for study, analysis in analyses:
                # Validate analysis
                if not hasattr(analysis, "points"):
                    raise InvalidStudysetError(f"Analysis {analysis.id} is missing points")

                if not analysis.points:
                    continue

                # Validate points
                for point in analysis.points:
                    if (
                        not hasattr(point, "x")
                        or not hasattr(point, "y")
                        or not hasattr(point, "z")
                    ):
                        raise InvalidStudysetError(
                            f"Point in analysis {analysis.id} is missing coordinates"
                        )

                _write_analysis(
                    f,
                    study,
                    analysis,
                    target=target,
                    decimal_precision=decimal_precision,
                    export_metadata=export_metadata,
                )


def _generate_metadata_comments(study, export_metadata):
    """Generate metadata comment lines for Sleuth file."""
    comments = []

    # Authors and publication info (prefer top-level attributes)
    if getattr(study, "authors", "") and "authors" in export_metadata:
        comments.append(f"Authors={study.authors}")
    if getattr(study, "publication", "") and "publication" in export_metadata:
        comments.append(f"Publication={study.publication}")

    # Collect metadata from Study object and any top-level attributes that may
    # have been provided in the original JSON but not stored in study.metadata.
    md = getattr(study, "metadata", {}) or {}

    # Year may exist as a top-level attribute or inside metadata
    year = getattr(study, "year", None) or md.get("year")
    if year and "year" in export_metadata:
        comments.append(f"Year={year}")

    # DOI and PMID may also be top-level, in __dict__, or in metadata
    doi = getattr(study, "doi", None) or md.get("doi")
    if doi and "doi" in export_metadata:
        comments.append(f"DOI={doi}")

    pmid = getattr(study, "pmid", None) or md.get("pmid")
    if pmid and "pmid" in export_metadata:
        comments.append(f"PubMedId={pmid}")

    # Affiliations/institutions may be in metadata or as a top-level attribute
    aff = (
        md.get("affiliations")
        or md.get("institutions")
        or md.get("institution")
        or getattr(study, "affiliations", None)
    )
    if aff and "affiliations" in export_metadata:
        comments.append(f"Affiliations={aff}")

    return comments


def _get_sample_size(analysis, study):
    """Extract sample size from analysis or study metadata."""
    # Check analysis metadata first
    sample_sizes = analysis.metadata.get("sample_sizes")
    sample_size = analysis.metadata.get("sample_size")

    # Fall back to study metadata
    if not sample_sizes and not sample_size:
        sample_sizes = study.metadata.get("sample_sizes")
        sample_size = study.metadata.get("sample_size")

    # Return appropriate sample size value
    if sample_sizes:
        if isinstance(sample_sizes, (list, tuple)):
            return min(sample_sizes) if sample_sizes else None
        else:
            return sample_sizes
    elif sample_size:
        return sample_size

    return None


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


def convert_dataset_to_nimads_dict(
    dataset: Dataset,
    *,
    studyset_id: str = "nimads_from_dataset",
    studyset_name: str = "",
    out_file: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Convert a NiMARE Dataset to a NIMADS Studyset dictionary.

    Parameters
    ----------
    dataset : :obj:`~nimare.dataset.Dataset`
        Dataset instance derived from Sleuth or other sources.
    studyset_id : :obj:`str`, optional
        Identifier for the resulting Studyset. Default is 'nimads_from_dataset'.
    studyset_name : :obj:`str`, optional
        Human-readable name for the Studyset. Default is ''.
    out_file : :obj:`str` or :obj:`pathlib.Path`, optional
        Optional path to write the Studyset JSON. Default is None.

    Returns
    -------
    dict
        NIMADS Studyset dictionary.
    """
    # Build studies in the deterministic order of dataset.ids (sorted in Dataset)
    studies: Dict[str, Dict[str, Any]] = {}

    # Prepare lookups
    md = dataset.metadata
    coords = dataset.coordinates

    # Ensure minimal required columns exist even if empty
    if md is None or md.empty:
        md = md if md is not None else pd.DataFrame(columns=["id", "study_id", "contrast_id"])
    if coords is None or coords.empty:
        coords = (
            coords
            if coords is not None
            else pd.DataFrame(columns=["id", "study_id", "contrast_id", "x", "y", "z", "space"])
        )

    for id_ in dataset.ids:
        # Pull identifiers
        row = md.loc[md["id"] == id_]
        if row.empty:
            # Fall back to coordinates to get identifiers
            row = coords.loc[coords["id"] == id_]
        if row.empty:
            # Skip if completely missing
            continue
        study_id = str(row["study_id"].iloc[0])
        contrast_id = str(row["contrast_id"].iloc[0])

        # Study object (create if needed)
        if study_id not in studies:
            studies[study_id] = {
                "id": study_id,
                "name": study_id,
                "authors": "",
                "publication": "",
                "metadata": {},
                "analyses": [],
            }

        # Analysis object
        analysis: Dict[str, Any] = {
            "id": contrast_id,
            "name": contrast_id,
            "conditions": [{"name": "default", "description": ""}],
            "weights": [1.0],
            "images": [],
            "points": [],
            "metadata": {},
        }

        # Sample size metadata if available
        if "sample_sizes" in row.columns and pd.notnull(row["sample_sizes"].iloc[0]):
            analysis["metadata"]["sample_sizes"] = row["sample_sizes"].iloc[0]
        elif "sample_size" in row.columns and pd.notnull(row["sample_size"].iloc[0]):
            analysis["metadata"]["sample_size"] = row["sample_size"].iloc[0]

        # Collect points for this analysis in order of appearance
        crows = coords.loc[coords["id"] == id_]
        for _, crow in crows.iterrows():
            sp = crow.get("space", None)
            if isinstance(sp, str):
                spl = sp.lower()
                if "mni" in spl or "ale" in spl:
                    sp = "MNI"
                elif "tal" in spl:
                    sp = "TAL"
            analysis["points"].append(
                {
                    "space": sp,
                    "coordinates": [float(crow["x"]), float(crow["y"]), float(crow["z"])],
                }
            )

        studies[study_id]["analyses"].append(analysis)

    studyset: Dict[str, Any] = {
        "id": studyset_id,
        "name": studyset_name,
        "studies": list(studies.values()),
    }

    if out_file is not None:
        out_path = Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(studyset, f, indent=2, sort_keys=False)

    return studyset


def convert_dataset_to_studyset(
    dataset: Dataset,
    *,
    studyset_id: str = "nimads_from_dataset",
    studyset_name: str = "",
    out_file: Optional[Union[str, Path]] = None,
):
    """Convert a NiMARE Dataset into a nimads.Studyset object.

    This is a convenience wrapper around :func:`convert_dataset_to_nimads_dict` that
    returns a fully-constructed :class:`nimare.nimads.Studyset` instance instead of a
    plain dictionary. If ``out_file`` is provided, the underlying NIMADS dictionary will
    also be written to disk (same behavior as :func:`convert_dataset_to_nimads_dict`).

    Parameters
    ----------
    dataset
        NiMARE Dataset to convert.
    studyset_id
        Identifier for the resulting Studyset.
    studyset_name
        Human-readable name for the Studyset.
    out_file
        Optional path to write the intermediate Studyset JSON.

    Returns
    -------
    nimads.Studyset
        Constructed Studyset object.
    """
    nimads_dict = convert_dataset_to_nimads_dict(
        dataset, studyset_id=studyset_id, studyset_name=studyset_name, out_file=out_file
    )
    # Local import to avoid circular import at module import time.
    from nimare.nimads import Studyset as _Studyset

    return _Studyset(nimads_dict)


def convert_sleuth_to_nimads_dict(
    text_file: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    target: str = "MNI",
    studyset_id: str = None,
    studyset_name: str = "",
) -> Dict[str, Any]:
    """Convert Sleuth text file(s) to a NIMADS Studyset dictionary.

    Parameters
    ----------
    text_file : :obj:`str`, :obj:`pathlib.Path`, or sequence of such
        Path(s) to Sleuth text file(s).
    target : :obj:`str`, optional
        Target space for Dataset loader. Accepts common dataset targets
        (e.g., "ale_2mm", "mni152_2mm") but also user-friendly strings like
        "MNI", "TAL", or variants such as "Talaraich". These are
        normalized to a Dataset-supported template string internally.
    studyset_id : :obj:`str`, optional
        Identifier for the resulting Studyset. Default is 'nimads_from_sleuth'.
    studyset_name : :obj:`str`, optional
        Human-readable name for the Studyset. Default is ''.

    Returns
    -------
    dict
        NIMADS Studyset dictionary.
    """
    # Normalize incoming target string to dataset template keys accepted by
    # Dataset (mni152_2mm or ale_2mm). Accept common variants including misspellings.
    try:
        tgt_str = str(target).lower()
    except Exception:
        tgt_str = "mni"

    # Map common/variant user inputs to Dataset-supported template keys.
    if "tal" in tgt_str:
        # Represent Talairach-like Sleuth targets using the MNI template
        # when converting Sleuth -> Dataset (Dataset expects 'mni152_2mm' or 'ale_2mm').
        ds_target = "mni152_2mm"
    elif "ale" in tgt_str:
        ds_target = "ale_2mm"
    elif "mni" in tgt_str:
        ds_target = "mni152_2mm"
    else:
        # Fallback to ALE 2mm when unknown
        ds_target = "ale_2mm"

    dset = convert_sleuth_to_dataset(text_file, target=ds_target)
    return convert_dataset_to_nimads_dict(
        dset, studyset_id=studyset_id, studyset_name=studyset_name
    )


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
