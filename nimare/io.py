"""Input/Output operations."""

import json
import logging
import re
from collections import Counter
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import requests
from scipy import sparse

from nimare.dataset import Dataset
from nimare.extract.utils import _get_dataset_dir
from nimare.utils import _create_name, load_nimads

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
        result = {
            "metadata": {
                "authors": study.name,
                "journal": study.publication,
                "title": study.name,
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
        return {
            "metadata": {
                "authors": study.authors,
                "journal": study.publication,
                "title": study.name,
            },
            "contrasts": {_create_name(a): _analysis_to_dict(study, a) for a in study.analyses},
        }

    # load nimads studyset
    studyset = load_nimads(studyset, annotation)
    return Dataset({_create_name(s): _study_to_dict(s) for s in list(studyset.studies)})


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
    annotation: Optional[Union[str, Dict[str, Any], Any]] = None,
    decimal_precision: int = 2,
    target_space: Optional[str] = None,
    *,
    normalize: bool = True,
    transform_affine: Optional[np.ndarray] = None,
    annotation_columns: Optional[Sequence[str]] = None,
    annotation_values: Optional[Dict[str, Sequence[Any]]] = None,
    split_booleans: bool = True,
    expected_version: Optional[str] = None,
):
    """Convert a NIMADS Studyset to Sleuth text file(s).

    Parameters
    ----------
    studyset : :obj:`str` or :obj:`dict` or :obj:`nimare.nimads.Studyset`
        Path to a Studyset JSON, a Studyset-like dictionary, or a Studyset object.
    output_dir : :obj:`str` or :obj:`pathlib.Path`
        Directory for Sleuth output. Created if missing.
    annotation : :obj:`str` or :obj:`dict` or :obj:`nimare.nimads.Annotation`, optional
        Optional annotation to split output files by annotation values.
    decimal_precision : :obj:`int`, optional
        Decimal places for output coordinates. Default is 2.
    target_space : :obj:`str` or None, optional
        Target space ("MNI", "TAL"/"Talairach"). If None, inferred from first analysis.
    normalize : :obj:`bool`, optional
        Apply named-space normalization when spaces differ. Default is True.
    transform_affine : (4, 4) :obj:`numpy.ndarray` or None, optional
        Optional affine applied to coordinates prior to normalization.
    annotation_columns : :obj:`list` of :obj:`str`, optional
        Restrict splitting to these annotation keys.
    annotation_values : :obj:`dict`, optional
        Filter annotation splits to these values per key.
    split_booleans : :obj:`bool`, optional
        When True, boolean columns produce ``_true``/``_false`` files. Default is True.
    expected_version : :obj:`str` or None, optional
        If provided, validate that ``studyset['version']`` matches this value when present.

    Raises
    ------
    InvalidStudysetError
        On invalid inputs or schema/structure mismatch.

    Notes
    -----
    Sleuth can only represent one reference space. This function normalizes
    coordinates into ``target_space`` when requested and available. Subjects
    lines are included when available but never synthesized.
    """
    import warnings
    try:
        from pydantic import validate_call
    except Exception:  # pragma: no cover - fallback when pydantic isn't installed
        def validate_call(func=None, **kwargs):  # type: ignore
            if func is None:
                def _wrap(f):
                    return f
                return _wrap
            return func
    from nimare import nimads
    from nimare.exceptions import ConversionWarning, InvalidStudysetError

    # Runtime validation of arguments (pydantic v2)
    @validate_call
    def _validate_args(
        studyset: Union[str, Dict[str, Any], Any],
        output_dir: Union[str, Path],
        annotation: Optional[Union[str, Dict[str, Any], Any]] = None,
        decimal_precision: int = 2,
        target_space: Optional[str] = None,
        normalize: bool = True,
        transform_affine: Optional[np.ndarray] = None,
        annotation_columns: Optional[Sequence[str]] = None,
        annotation_values: Optional[Dict[str, Sequence[Any]]] = None,
        split_booleans: bool = True,
        expected_version: Optional[str] = None,
    ):
        return (
            studyset,
            output_dir,
            annotation,
            decimal_precision,
            target_space,
            normalize,
            transform_affine,
            annotation_columns,
            annotation_values,
            split_booleans,
            expected_version,
        )

    (
        studyset,
        output_dir,
        annotation,
        decimal_precision,
        target_space,
        normalize,
        transform_affine,
        annotation_columns,
        annotation_values,
        split_booleans,
        expected_version,
    ) = _validate_args(
        studyset,
        output_dir,
        annotation,
        decimal_precision,
        target_space,
        normalize,
        transform_affine,
        annotation_columns,
        annotation_values,
        split_booleans,
        expected_version,
    )

    if decimal_precision < 0:
        raise InvalidStudysetError("decimal_precision must be non-negative")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load studyset object and dict for validation
    try:
        studyset_obj = studyset if isinstance(studyset, nimads.Studyset) else nimads.Studyset(studyset)
    except Exception as e:
        raise InvalidStudysetError(f"Failed to load studyset: {e}") from e

    if annotation is not None:
        try:
            studyset_obj.annotations = annotation
        except Exception as e:
            raise InvalidStudysetError(f"Failed to load annotation: {e}") from e

    # Schema validation
    studyset_dict = studyset_obj.to_dict()
    if expected_version is not None:
        version = studyset_dict.get("version")
        if version is not None and version != expected_version:
            raise InvalidStudysetError(
                f"Studyset version mismatch: expected {expected_version}, got {version}"
            )
    try:
        _validate_nimads_schema(studyset_dict)
    except Exception as e:
        # Wrap any schema/structure error
        raise InvalidStudysetError(f"Studyset failed schema validation: {e}") from e
    
    # Determine target space
    if target_space is None and studyset_obj.studies:
        # Use space from first analysis with points
        for study in studyset_obj.studies:
            for analysis in study.analyses:
                if analysis.points:
                    target_space = analysis.points[0].space
                    break
            if target_space:
                break
    
    # Validate target space
    if target_space and target_space not in ["MNI", "TAL", "Talairach"]:
        warnings.warn(
            f"Unknown space '{target_space}'. Defaulting to 'MNI'.",
            ConversionWarning,
        )
        target_space = "MNI"
    
    # Handle annotations
    if studyset_obj.annotations:
        # Process with annotations - create separate files
        _process_with_annotations(
            studyset_obj,
            output_dir,
            target_space,
            decimal_precision,
            normalize,
            transform_affine,
            annotation_columns,
            annotation_values,
            split_booleans,
        )
    else:
        # Process without annotations - single file
        _process_without_annotations(
            studyset_obj,
            output_dir,
            target_space,
            decimal_precision,
            normalize,
            transform_affine,
        )


def _process_without_annotations(
    studyset,
    output_dir,
    target_space,
    decimal_precision,
    normalize,
    transform_affine,
):
    """Process studyset without annotations (single Sleuth file)."""
    import numpy as np
    from nimare.exceptions import InvalidStudysetError
    from nibabel.affines import apply_affine

    # Import coordinate transformation functions only if needed
    if target_space:
        from nimare.utils import tal2mni, mni2tal

    # Create single output file
    output_file = output_dir / "nimads_sleuth_file.txt"

    with open(output_file, "w") as f:
        # Write space header
        space_header = target_space if target_space else "MNI"
        f.write(f"// Reference = {space_header}\n")

        # If no studies, leave just the header
        if not studyset.studies:
            return

        # Process each study in deterministic order
        for study in sorted(studyset.studies, key=lambda s: s.id):
            # Validate study
            if not hasattr(study, 'analyses'):
                raise InvalidStudysetError(f"Study {study.id} is missing analyses")
            
            # Process each analysis in deterministic order
            for analysis in sorted(study.analyses, key=lambda a: a.id):
                # Validate analysis
                if not hasattr(analysis, 'points'):
                    raise InvalidStudysetError(f"Analysis {analysis.id} is missing points")
                
                if not analysis.points:
                    continue
                
                # Validate points
                for point in analysis.points:
                    if (not hasattr(point, 'x') or not hasattr(point, 'y')
                        or not hasattr(point, 'z')):
                        raise InvalidStudysetError(
                            f"Point in analysis {analysis.id} is missing coordinates"
                        )
                
                # Generate hierarchical label with metadata
                study_name = study.name or f"Study_{study.id}"
                analysis_name = analysis.name or f"Analysis_{analysis.id}"
                
                # Add author information to label if available
                if study.authors:
                    label = f"{study.authors} ({study_name}): {analysis_name}"
                else:
                    label = f"{study_name}: {analysis_name}"
                
                # Write study header
                f.write(f"// {label}\n")
                
                # Write sample size if available
                sample_size = _get_sample_size(analysis, study)
                if sample_size:
                    f.write(f"// Subjects = {sample_size}\n")
                
                # Write additional metadata as comments
                metadata_lines = _generate_metadata_comments(study, analysis)
                for line in metadata_lines:
                    f.write(f"// {line}\n")
                
                # Transform and write coordinates
                coords = []
                source_space = None
                for point in analysis.points:
                    if source_space is None:
                        source_space = point.space or "UNKNOWN"
                    coords.append([point.x, point.y, point.z])
                
                if coords:
                    coords = np.array(coords)
                    # Validate coordinates are numeric
                    if not np.issubdtype(coords.dtype, np.number):
                        raise InvalidStudysetError("Coordinates must be numeric")
                    # Apply affine first, then named-space normalization if requested
                    if transform_affine is not None:
                        coords = apply_affine(transform_affine, coords)
                    if normalize and target_space and source_space and target_space != source_space:
                        if target_space in ["MNI"] and source_space in ["TAL", "Talairach"]:
                            coords = tal2mni(coords)
                        elif target_space in ["TAL", "Talairach"] and source_space in ["MNI"]:
                            coords = mni2tal(coords)
                    
                    # Write coordinates with specified precision
                    for coord in coords:
                        f.write(
                            f"{coord[0]:.{decimal_precision}f}\t"
                            f"{coord[1]:.{decimal_precision}f}\t"
                            f"{coord[2]:.{decimal_precision}f}\n"
                        )
                
                f.write("\n")  # Empty line between analyses


def _generate_metadata_comments(study, analysis):
    """Generate metadata comment lines for Sleuth file."""
    comments = []

    # Authors and publication info
    if getattr(study, "authors", ""):
        comments.append(f"Authors: {study.authors}")
    if getattr(study, "publication", ""):
        comments.append(f"Publication: {study.publication}")

    md = getattr(study, "metadata", {}) or {}
    if md.get("year"):
        comments.append(f"Year: {md.get('year')}")
    if md.get("doi"):
        comments.append(f"DOI: {md.get('doi')}")
    if md.get("pmid"):
        comments.append(f"PMID: {md.get('pmid')}")
    aff = md.get("affiliations") or md.get("institutions") or md.get("institution")
    if aff:
        comments.append(f"Affiliations: {aff}")

    return comments


def _process_with_annotations(
    studyset,
    output_dir,
    target_space,
    decimal_precision,
    normalize,
    transform_affine,
    annotation_columns,
    annotation_values,
    split_booleans,
):
    """Process studyset with annotations, creating multiple files."""
    import numpy as np
    from collections import defaultdict
    from nibabel.affines import apply_affine
    from nimare.exceptions import InvalidStudysetError

    # Get the first annotation
    if not studyset.annotations:
        _process_without_annotations(
            studyset, output_dir, target_space, decimal_precision, normalize, transform_affine
        )
        return

    annotation = studyset.annotations[0]  # Use first annotation

    # Validate annotation
    if not hasattr(annotation, 'notes'):
        raise InvalidStudysetError("Annotation is missing notes")

    # Group analyses by annotation values
    analysis_groups = defaultdict(list)

    # Process each analysis and group by annotation values
    for study in sorted(studyset.studies, key=lambda s: s.id):
        # Validate study
        if not hasattr(study, 'analyses'):
            raise InvalidStudysetError(f"Study {study.id} is missing analyses")

        for analysis in sorted(study.analyses, key=lambda a: a.id):
            # Validate analysis
            if not hasattr(analysis, 'points'):
                raise InvalidStudysetError(f"Analysis {analysis.id} is missing points")
            
            # Find annotation note for this analysis
            note = None
            for n in annotation.notes:
                if n.analysis.id == analysis.id:
                    note = n
                    break
            
            if note:
                # Validate note
                if not hasattr(note, 'note'):
                    raise InvalidStudysetError(f"Note for analysis {analysis.id} is missing note data")
                # Group by annotation values (restricted and filtered if requested)
                keyvals = note.note.items()
                if annotation_columns is not None:
                    keyvals = [(k, v) for k, v in keyvals if k in set(annotation_columns)]
                for key, value in sorted(keyvals, key=lambda kv: (kv[0], str(kv[1]))):
                    if annotation_values and key in annotation_values and value not in annotation_values[key]:
                        continue
                    if isinstance(value, bool) and split_booleans:
                        group_name = f"{key}_{str(value).lower()}"
                    else:
                        group_name = f"{key}_{value}"
                    analysis_groups[group_name].append((study, analysis))
            else:
                # Add to default group if no annotation
                analysis_groups["default"].append((study, analysis))

    # Import coordinate transformation functions only if needed
    if target_space:
        from nimare.utils import tal2mni, mni2tal

    # Create separate files for each group (sorted deterministically)
    for group_name in sorted(analysis_groups.keys()):
        analyses = analysis_groups[group_name]
        # Sanitize filename
        safe_name = "".join(c for c in group_name if c.isalnum() or c in "._-").rstrip()
        if not safe_name:
            safe_name = "unnamed_group"
        
        output_file = output_dir / f"{safe_name}.txt"
        
        with open(output_file, "w") as f:
            # Write space header
            space_header = target_space if target_space else "MNI"
            f.write(f"// Reference = {space_header}\n")
            
            # Process each analysis in this group
            for study, analysis in analyses:
                # Validate analysis
                if not hasattr(analysis, 'points'):
                    raise InvalidStudysetError(f"Analysis {analysis.id} is missing points")
                
                if not analysis.points:
                    continue
                
                # Validate points
                for point in analysis.points:
                    if not hasattr(point, 'x') or not hasattr(point, 'y') or not hasattr(point, 'z'):
                        raise InvalidStudysetError(f"Point in analysis {analysis.id} is missing coordinates")
                
                # Generate hierarchical label with metadata
                study_name = study.name or f"Study_{study.id}"
                analysis_name = analysis.name or f"Analysis_{analysis.id}"
                
                # Add author information to label if available
                if study.authors:
                    label = f"{study.authors} ({study_name}): {analysis_name}"
                else:
                    label = f"{study_name}: {analysis_name}"
                
                # Write study header
                f.write(f"// {label}\n")
                
                # Write sample size if available
                sample_size = _get_sample_size(analysis, study)
                if sample_size:
                    f.write(f"// Subjects = {sample_size}\n")
                
                # Write additional metadata as comments
                metadata_lines = _generate_metadata_comments(study, analysis)
                for line in metadata_lines:
                    f.write(f"// {line}\n")
                
                # Transform and write coordinates
                coords = []
                source_space = None
                for point in analysis.points:
                    if source_space is None:
                        source_space = point.space or "UNKNOWN"
                    coords.append([point.x, point.y, point.z])
                
                if coords:
                    coords = np.array(coords)
                    # Validate coordinates are numeric
                    if not np.issubdtype(coords.dtype, np.number):
                        raise InvalidStudysetError("Coordinates must be numeric")
                    # Apply affine then optional normalization
                    if transform_affine is not None:
                        coords = apply_affine(transform_affine, coords)
                    if normalize and target_space and source_space and target_space != source_space:
                        if target_space in ["MNI"] and source_space in ["TAL", "Talairach"]:
                            coords = tal2mni(coords)
                        elif target_space in ["TAL", "Talairach"] and source_space in ["MNI"]:
                            coords = mni2tal(coords)
                    
                    # Write coordinates with specified precision
                    for coord in coords:
                        f.write(
                            f"{coord[0]:.{decimal_precision}f}\t"
                            f"{coord[1]:.{decimal_precision}f}\t"
                            f"{coord[2]:.{decimal_precision}f}\n"
                        )
                
                f.write("\n")  # Empty line between analyses


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
            return sample_sizes[0] if sample_sizes else None
        else:
            return sample_sizes
    elif sample_size:
        return sample_size
    
    return None


# -----------------
# JSON Schema (v7)
# -----------------
_POINT_SCHEMA = {
    "type": "object",
    "properties": {
        "space": {"type": ["string", "null"]},
        "coordinates": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {"type": ["number", "integer"]},
        },
    },
    "required": ["coordinates"],
}

_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "name": {"type": ["string", "null"]},
        "conditions": {"type": "array"},
        "weights": {"type": "array"},
        "images": {"type": "array"},
        "points": {"type": "array", "items": _POINT_SCHEMA},
        "metadata": {"type": ["object", "null"]},
    },
    "required": ["id", "name", "conditions", "weights", "images", "points"],
}

_STUDY_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "name": {"type": ["string", "null"]},
        "authors": {"type": ["string", "null"]},
        "publication": {"type": ["string", "null"]},
        "metadata": {"type": ["object", "null"]},
        "analyses": {"type": "array", "items": _ANALYSIS_SCHEMA},
    },
    "required": ["id", "name", "analyses"],
}

_STUDYSET_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "name": {"type": ["string", "null"]},
        "version": {"type": ["string", "number", "null"]},
        "studies": {"type": "array", "items": _STUDY_SCHEMA},
    },
    "required": ["id", "name", "studies"],
}


def _validate_nimads_schema(studyset_dict: Dict[str, Any]) -> None:
    """Validate a NIMADS-like Studyset dictionary against Draft-07 schema."""
    try:
        import jsonschema  # type: ignore
    except Exception:
        # Minimal structural checks when jsonschema isn't available
        if not isinstance(studyset_dict, dict):
            raise ValueError("Studyset must be a dictionary-like object")
        if "studies" not in studyset_dict or not isinstance(studyset_dict["studies"], list):
            raise ValueError("Studyset must include a list field 'studies'")
        for study in studyset_dict["studies"]:
            if not isinstance(study, dict) or "analyses" not in study or not isinstance(
                study["analyses"], list
            ):
                raise ValueError("Each study must include an 'analyses' list")
        return

    jsonschema.validate(instance=studyset_dict, schema=_STUDYSET_SCHEMA)


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


def convert_dataset_to_nimads(
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
        md = md if md is not None else pd.DataFrame(columns=["id", "study_id", "contrast_id"])  # type: ignore
    if coords is None or coords.empty:
        coords = coords if coords is not None else pd.DataFrame(
            columns=["id", "study_id", "contrast_id", "x", "y", "z", "space"]
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


def convert_sleuth_to_nimads(
    text_file: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    target: str = "ale_2mm",
    studyset_id: str = "nimads_from_sleuth",
    studyset_name: str = "",
    out_file: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Convert Sleuth text file(s) to a NIMADS Studyset dictionary.

    Parameters
    ----------
    text_file : :obj:`str`, :obj:`pathlib.Path`, or sequence of such
        Path(s) to Sleuth text file(s).
    target : :obj:`str`, optional
        Target space for Dataset loader. Default is 'ale_2mm'.
    studyset_id : :obj:`str`, optional
        Identifier for the resulting Studyset. Default is 'nimads_from_sleuth'.
    studyset_name : :obj:`str`, optional
        Human-readable name for the Studyset. Default is ''.
    out_file : :obj:`str` or :obj:`pathlib.Path`, optional
        Optional path to write the Studyset JSON. Default is None.

    Returns
    -------
    dict
        NIMADS Studyset dictionary.
    """
    dset = convert_sleuth_to_dataset(text_file, target=target)
    return convert_dataset_to_nimads(
        dset, studyset_id=studyset_id, studyset_name=studyset_name, out_file=out_file
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
