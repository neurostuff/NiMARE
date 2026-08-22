"""Miscellaneous spatial and statistical transforms."""

import logging
import os
import os.path as op
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.reporting import get_clusters_table
from pymare.stats import log_chi2_sf
from scipy import special, stats

from nimare.base import NiMAREBase
from nimare.studyset import normalize_collection
from nimare.utils import (
    DEFAULT_FLOAT_DTYPE,
    _clip_p_values,
    _dict_to_coordinates,
    _dict_to_df,
    _listify,
    get_masker,
)

LGR = logging.getLogger(__name__)


def _coerce_image_like(image):
    """Return image-like input as either a numpy array or a loaded niimg."""
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, (str, os.PathLike)):
        return nib.load(str(image))
    if isinstance(image, nib.spatialimages.SpatialImage):
        return image
    raise ValueError(
        "image inputs must be numpy arrays, Niimg-like objects, or paths to image files; "
        f"got {type(image)}"
    )


def threshold_image(image, threshold, thresholding_values=None, tail="upper", fill_value=0):
    """Threshold an array or Niimg-like object and zero voxels that fail the criterion.

    Parameters
    ----------
    image : array-like or Niimg-like
        Values to retain where the threshold criterion passes.
    threshold : :obj:`float`
        Threshold applied to ``thresholding_values`` or, when that argument is omitted,
        to ``image`` itself.
    thresholding_values : array-like or Niimg-like, optional
        Values used to determine which voxels survive thresholding. Must match the shape of
        ``image``. If omitted, thresholding is applied directly to ``image``.
    tail : {"upper", "lower", "two-sided"}, optional
        Threshold direction. ``"upper"`` keeps values >= threshold, ``"lower"`` keeps values
        <= threshold, and ``"two-sided"`` keeps values whose absolute value >= threshold.
        Default is ``"upper"``.
    fill_value : scalar, optional
        Replacement value for voxels that do not survive thresholding. Default is 0.

    Returns
    -------
    array-like or Niimg-like
        Thresholded output with the same container type as ``image``.
    """
    image = _coerce_image_like(image)
    thresholding_values = (
        image if thresholding_values is None else _coerce_image_like(thresholding_values)
    )

    image_data = (
        np.asarray(image) if isinstance(image, np.ndarray) else np.asanyarray(image.dataobj)
    )
    threshold_data = (
        np.asarray(thresholding_values)
        if isinstance(thresholding_values, np.ndarray)
        else np.asanyarray(thresholding_values.dataobj)
    )

    if image_data.shape != threshold_data.shape:
        raise ValueError(
            "image and thresholding_values must have matching shapes; "
            f"got {image_data.shape} and {threshold_data.shape}."
        )

    if tail == "upper":
        keep_mask = threshold_data >= threshold
    elif tail == "lower":
        keep_mask = threshold_data <= threshold
    elif tail == "two-sided":
        keep_mask = np.abs(threshold_data) >= threshold
    else:
        raise ValueError(f"Unsupported tail '{tail}'.")

    thresholded = np.where(keep_mask, image_data, fill_value)
    if np.issubdtype(image_data.dtype, np.floating) and thresholded.dtype != image_data.dtype:
        thresholded = thresholded.astype(image_data.dtype, copy=False)

    if isinstance(image, np.ndarray):
        return thresholded

    header = image.header.copy()
    header.set_data_dtype(thresholded.dtype)
    return image.__class__(thresholded, image.affine, header)


class ImageTransformer(NiMAREBase):
    """A class to create new images from existing ones within a collection.

    This class is a light wrapper around :func:`~nimare.transforms.transform_images`.

    .. warning::
        Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed in
        a future release. Prefer :class:`~nimare.nimads.Studyset`.

    .. versionadded:: 0.0.9

    Parameters
    ----------
    target : {'z', 'p', 't', 'beta', 'varcope', 'd', 'g', 'g_var'} or list
        Target image type. Multiple target types may be specified as a list.
        ``'g'`` and ``'g_var'`` give a standardized effect size and its variance, which
        unlike :term:`beta` and :term:`varcope` are comparable across studies whose
        pipelines used different units.
    overwrite : :obj:`bool`, optional
        Whether to overwrite existing files or not. Default is False.

    See Also
    --------
    nimare.transforms.transform_images : The function called by this class.
    """

    def __init__(self, target, overwrite=False):
        self.target = _listify(target)
        self.overwrite = overwrite

    def transform(self, dataset):
        """Generate images of the target type from other image types in a collection.

        Parameters
        ----------
        dataset : :obj:`~nimare.nimads.Studyset` or :obj:`~nimare.dataset.Dataset`
            A collection containing images and relevant metadata.

        Returns
        -------
        new_dataset : same type as input when possible
            A copy of the input collection, with new images added to its images attribute.

        .. warning::
            Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed
            in a future release. Prefer :class:`~nimare.nimads.Studyset`.
        """
        dataset = normalize_collection(dataset)

        temp_images = dataset.images
        for target_type in self.target:
            temp_images = transform_images(
                temp_images,
                target=target_type,
                masker=dataset.masker,
                metadata_df=dataset.metadata,
                out_dir=dataset.basepath,
                overwrite=self.overwrite,
            )

        # Append the derived maps rather than replacing an images table. A
        # studyset holds every image an analysis has, so a generated map sits
        # alongside the one it came from instead of overwriting a type slot.
        existing = dataset.images
        # Grouped by type as it is gathered. Three parallel lists that had to
        # stay in step, then be regrouped by scanning them once per type, is
        # precisely the bookkeeping a block-shaped input is supposed to remove.
        added = {}
        row_of = dataset.row_of_id()
        for imtype in self.target:
            if imtype not in temp_images.columns:
                continue
            before = existing[imtype] if imtype in existing.columns else None
            for i, (analysis_id, path) in enumerate(zip(temp_images["id"], temp_images[imtype])):
                if not isinstance(path, str) or not path:
                    continue
                if before is not None and before.iloc[i] == path:
                    continue  # already present
                row = row_of.get(str(analysis_id))
                if row is None:
                    continue
                positions, refs = added.setdefault(imtype, ([], []))
                positions.append(row)
                refs.append(path)
        new_dataset = dataset
        for imtype in sorted(added):
            positions, refs = added[imtype]
            new_dataset = new_dataset.with_images(positions, refs, imtype)
        return new_dataset


def transform_images(images_df, target, masker, metadata_df=None, out_dir=None, overwrite=False):
    """Generate images of a given type from other image types and write out to files.

    .. versionchanged:: 0.0.9

        * [ENH] Add overwrite option to transform_images

    .. versionadded:: 0.0.4

    Parameters
    ----------
    images_df : :class:`pandas.DataFrame`
        DataFrame with paths to images for studies in Dataset.
    target : {'z', 'p', 't', 'beta', 'varcope', 'd', 'g', 'g_var'}
        Target data type.
    masker : :class:`~nilearn.maskers.NiftiMasker` or similar
        Masker used to define orientation and resolution of images.
        Specific voxels defined in mask will not be used, and a new masker
        with _all_ voxels in acquisition matrix selected will be created.
    metadata_df : :class:`pandas.DataFrame` or :obj:`None`, optional
        DataFrame with metadata. Rows in this DataFrame must match those in
        ``images_df``, including the ``'id'`` column.
    out_dir : :obj:`str` or :obj:`None`, optional
        Path to output directory. If None, use folder containing first image
        for each study in ``images_df``.
    overwrite : :obj:`bool`, optional
        Whether to overwrite existing files or not. Default is False.

    Returns
    -------
    images_df : :class:`pandas.DataFrame`
        DataFrame with paths to new images added.
    """
    new_images_df = images_df.copy()  # Work on a copy of the images_df

    valid_targets = {"t", "z", "p", "beta", "varcope", "d", "g", "g_var"}
    if target not in valid_targets:
        raise ValueError(
            f"Target type {target} not supported. Must be one of: {', '.join(valid_targets)}"
        )

    mask_img = masker.mask_img
    new_mask = np.ones(mask_img.shape, int)
    new_mask = nib.Nifti1Image(new_mask, mask_img.affine, header=mask_img.header)
    new_masker = get_masker(new_mask)
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
            if not options:
                LGR.warning(f"No existing image files for {id_}, skipping {target} transform.")
                continue
            id_out_dir = op.dirname(options[0])
        else:
            id_out_dir = out_dir
        new_file = op.join(id_out_dir, f"{id_}_{res}_{target}.nii.gz")

        # Grab columns with actual values
        available_data = row[~row.isnull()].to_dict()
        if metadata_df is not None:
            metadata_row = metadata_df.loc[metadata_df["id"] == id_].iloc[0]
            metadata = metadata_row[~metadata_row.isnull()].to_dict()
            for k, v in metadata.items():
                if k not in available_data.keys():
                    available_data[k] = v

        # Get converted data
        img = resolve_transforms(target, available_data, new_masker, id_=id_)
        if img is not None:
            if overwrite or not op.isfile(new_file):
                img.to_filename(new_file)
            else:
                LGR.debug("Image already exists. Not overwriting.")

            new_images_df.loc[new_images_df["id"] == id_, target] = new_file
        else:
            new_images_df.loc[new_images_df["id"] == id_, target] = None

    # Ensure the target column exists even when every study was skipped.
    if target not in new_images_df.columns:
        new_images_df[target] = None

    return new_images_df


def resolve_transforms(target, available_data, masker, id_=None):
    """Determine and apply the appropriate transforms to a target image type from available data.

    .. versionchanged:: 0.21.0

        * [FIX] Take the sign from a t map when converting a p map without a sample size.
        * [ENH] Warn when ``z`` is derived from a p map with no sign available anywhere.
        * [ENH] Accept ``id_``, so that warning can name the analysis it came from.

    .. versionchanged:: 0.0.8

        * [FIX] Remove unnecessary dimensions from output image object *img_like*. \
                Now, the image object only has 3 dimensions.

    .. versionadded:: 0.0.4

    Parameters
    ----------
    target : {'z', 'p', 't', 'beta', 'varcope', 'd', 'g', 'g_var'}
        Target image type. ``'d'`` is Cohen's d, ``'g'`` is Hedges' g and ``'g_var'`` is the
        sampling variance of g; the latter two are the effect estimate and variance a
        meta-regression takes.
    available_data : dict
        Dictionary mapping data types to their values. Images in the dictionary
        are paths to files.
    masker : nilearn Masker
        Masker used to convert images to arrays and back. Preferably, this mask
        should cover the full acquisition matrix (rather than an ROI), given
        that the calculated images will be saved and used for the full Dataset.
    id_ : :obj:`str` or None, optional
        Identifier of the analysis the data belong to. Used only to name the analysis in
        warnings. Default is None.

    Returns
    -------
    img_like or None
        Image object with the desired data type, if it can be generated.
        Otherwise, None.

    Notes
    -----
    A p-value does not record the direction of its effect, so a ``z`` derived from a p map
    and nothing else is unsigned, as is anything derived from it in turn (``t``, ``beta``,
    ``d``, ``g``). That conversion is still performed, but it warns. A t map in the same
    analysis supplies the direction even when it is missing the sample size the magnitude
    would need, and the sign is taken from it instead, silently.
    """
    if target in available_data.keys():
        LGR.warning(f"Target '{target}' already available.")
        return available_data[target]

    if target == "z":
        if ("t" in available_data.keys()) and ("sample_sizes" in available_data.keys()):
            dof = sample_sizes_to_dof(available_data["sample_sizes"])
            t = masker.transform(available_data["t"])
            z = t_to_z(t, dof)
        elif "p" in available_data.keys():
            p = masker.transform(available_data["p"])
            z = p_to_z(p)
            if "t" in available_data.keys():
                # Reached only without a sample size, so the t map cannot set the
                # magnitude, but it still carries the direction, which p does not, so take
                # the sign from it rather than return an unsigned map. Where t is
                # exactly 0 the voxel goes to 0: sign(0) is 0, and a t of 0 is a p of 1,
                # whose z is 0 anyway, so the two agree. Non-finite t propagates.
                t = masker.transform(available_data["t"])
                z = np.sign(t) * z
            else:
                prefix = f"{id_}: " if id_ is not None else ""
                LGR.warning(
                    f"{prefix}Deriving 'z' from a p map, with nothing in the analysis to "
                    "give the direction. A p-value carries no sign, so the result is "
                    "unsigned: every voxel is positive. In a signed test (Stouffers, "
                    "Fishers) this analysis then contributes positive evidence only, "
                    "whichever way its contrast ran. The p-values are read as two-tailed "
                    "as well, so a one-tailed p map comes out with the wrong magnitude. "
                    "Supply a z map, or a t map: with a sample size it sets the "
                    "magnitude, without one it still sets the sign."
                )
        else:
            return None
        z = masker.inverse_transform(z.squeeze())
        return z
    elif target == "t":
        # will return none given no transform/target exists
        temp = resolve_transforms("z", available_data, masker, id_=id_)
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
    elif target in ("d", "g", "g_var"):
        # All three start from t and the sample size. t itself resolves from z, so a
        # studyset holding only z maps can still reach a standardized effect size.
        if "t" not in available_data.keys():
            temp = resolve_transforms("t", available_data, masker, id_=id_)
            if temp is not None:
                available_data["t"] = temp

        if ("t" not in available_data.keys()) or ("sample_sizes" not in available_data.keys()):
            return None

        sample_sizes = available_data["sample_sizes"]
        if np.size(sample_sizes) > 1:
            LGR.warning(
                "Converting to '%s' from %d group sample sizes. The conversion is the "
                "one-sample d = t / sqrt(N), so a between-group contrast will come out "
                "wrong; that needs both group sizes separately.",
                target,
                np.size(sample_sizes),
            )
        sample_size = sample_sizes_to_sample_size(sample_sizes)

        d = t_to_d(masker.transform(available_data["t"]), sample_size)
        if target == "d":
            values = d
        else:
            g, g_var = d_to_g(d, sample_size, return_variance=True)
            values = g if target == "g" else g_var

        return masker.inverse_transform(values.squeeze())
    elif target == "beta":
        if "t" not in available_data.keys():
            # will return none given no transform/target exists
            temp = resolve_transforms("t", available_data, masker, id_=id_)
            if temp is not None:
                available_data["t"] = temp

        if "varcope" not in available_data.keys():
            temp = resolve_transforms("varcope", available_data, masker, id_=id_)
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
    elif target == "p":
        if ("t" in available_data.keys()) and ("sample_sizes" in available_data.keys()):
            dof = sample_sizes_to_dof(available_data["sample_sizes"])
            t = masker.transform(available_data["t"])
            z = t_to_z(t, dof)
            p = z_to_p(z)
        elif "z" in available_data.keys():
            z = masker.transform(available_data["z"])
            p = z_to_p(z)
        else:
            return None
        p = masker.inverse_transform(p.squeeze())
        return p
    else:
        return None


class ImagesToCoordinates(NiMAREBase):
    """Transformer from images to coordinates.

    .. versionadded:: 0.0.8

    Parameters
    ----------
    merge_strategy : {"fill", "replace", "demolish"}, optional
        Strategy for how to incorporate the generated coordinates with possible pre-existing
        coordinates. The available options are

        ================ =========================================================================
        "fill" (default) Only add coordinates to study contrasts that do not have coordinates.
                         If a study contrast has both image and coordinate data, the original
                         coordinate data will be kept.
        "replace"        Replace existing coordinates with coordinates generated by this function.
                         If a study contrast only has coordinate data and no images or if the
                         statistical threshold is too high for nimare to detect any peaks the
                         original coordinates will be kept.
        "demolish"       Only keep generated coordinates and discard any study contrasts with
                         coordinate data, but no images.
        ================ =========================================================================

    cluster_threshold : :obj:`int` or `None`, optional
        Cluster size threshold, in voxels. Default=None.
    remove_subpeaks : :obj:`bool`, optional
        If True, removes subpeaks from the cluster results. Default=False.
    two_sided : :obj:`bool`, optional
        Whether to employ two-sided thresholding or to evaluate positive values only.
        Default=False.
    min_distance : :obj:`float`, optional
        Minimum distance between subpeaks in mm. Default=8mm.
    z_threshold : :obj:`float`
        Cluster forming z-scale threshold. Default=3.1.

    Notes
    -----
    The raw Z and/or P maps are not corrected for multiple comparisons. Uncorrected z-values and/or
    p-values are used for thresholding.
    """

    def __init__(
        self,
        merge_strategy="fill",
        cluster_threshold=None,
        remove_subpeaks=False,
        two_sided=False,
        min_distance=8.0,
        z_threshold=3.1,
    ):
        self.merge_strategy = merge_strategy
        self.cluster_threshold = cluster_threshold
        self.remove_subpeaks = remove_subpeaks
        self.min_distance = min_distance
        self.two_sided = two_sided
        self.z_threshold = z_threshold

    def transform(self, dataset):
        """Create coordinate peaks from statistical images.

        Parameters
        ----------
        dataset : :obj:`~nimare.nimads.Studyset` or :obj:`~nimare.dataset.Dataset`
            Collection with z maps and/or p maps
            that can be converted to coordinates.

        Returns
        -------
        dataset : same type as input when possible
            Collection with coordinates generated from
            images and metadata indicating origin
            of coordinates ('original' or 'nimare').

        .. warning::
            Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed
            in a future release. Prefer :class:`~nimare.nimads.Studyset`.
        """
        dataset = normalize_collection(dataset)

        # relevant variables from dataset
        space = dataset.space
        images_df = dataset.images
        metadata = dataset.metadata.copy()

        # conform space specification
        if "mni" in space.lower() or "ale" in space.lower():
            coordinate_space = "MNI"
        elif "tal" in space.lower():
            coordinate_space = "TAL"
        else:
            coordinate_space = None

        coordinates_dict = {}
        cluster_threshold = 0 if self.cluster_threshold is None else self.cluster_threshold
        for _, row in images_df.iterrows():
            if row["id"] in list(dataset.coordinates["id"]) and self.merge_strategy == "fill":
                continue

            z_path = row.get("z")
            p_path = row.get("p")
            has_z = isinstance(z_path, (str, os.PathLike)) and str(z_path).strip() != ""
            has_p = isinstance(p_path, (str, os.PathLike)) and str(p_path).strip() != ""

            if has_z:
                clusters = get_clusters_table(
                    nib.funcs.squeeze_image(nib.load(z_path)),
                    self.z_threshold,
                    cluster_threshold,
                    two_sided=self.two_sided,
                    min_distance=self.min_distance,
                )
            elif has_p:
                LGR.info(
                    f"No Z map for {row['id']}, using p map "
                    "(p-values will be treated as positive z-values)"
                )
                if self.two_sided:
                    LGR.warning(f"Cannot use two_sided threshold using a p map for {row['id']}")

                p_threshold = 1 - z_to_p(self.z_threshold)
                nimg = nib.funcs.squeeze_image(nib.load(p_path))
                inv_nimg = nib.Nifti1Image(
                    1 - nimg.get_fdata(dtype=DEFAULT_FLOAT_DTYPE),
                    nimg.affine,
                    nimg.header,
                )
                clusters = get_clusters_table(
                    inv_nimg,
                    p_threshold,
                    cluster_threshold,
                    two_sided=False,
                    min_distance=self.min_distance,
                )
                # Peak stat p-values are reported as 1 - p in get_clusters_table
                clusters["Peak Stat"] = p_to_z(1 - clusters["Peak Stat"])
            else:
                LGR.warning(f"No Z or p map for {row['id']}, skipping...")
                continue

            # skip entry if no clusters are found
            if clusters.empty:
                LGR.warning(
                    f"No clusters were found for {row['id']} at a threshold of {self.z_threshold}"
                )
                continue

            if self.remove_subpeaks:
                # subpeaks are identified as 1a, 1b, etc
                # while peaks are kept as 1, 2, 3, etc,
                # so removing all non-int rows will
                # keep main peaks while removing subpeaks
                clusters = clusters[clusters["Cluster ID"].apply(lambda x: isinstance(x, int))]

            coordinates_dict[row["study_id"]] = {
                "contrasts": {
                    row["contrast_id"]: {
                        "coords": {
                            "space": coordinate_space,
                            "x": list(clusters["X"]),
                            "y": list(clusters["Y"]),
                            "z": list(clusters["Z"]),
                            "z_stat": list(clusters["Peak Stat"]),
                        },
                        "metadata": {"coordinate_source": "nimare"},
                    }
                }
            }

        # only the generated coordinates ('demolish')
        coordinates_df = _dict_to_coordinates(coordinates_dict, space)
        meta_df = _dict_to_df(
            pd.DataFrame(dataset.ids),
            coordinates_dict,
            "metadata",
        )

        if "coordinate_source" in meta_df.columns:
            metadata["coordinate_source"] = meta_df["coordinate_source"]
        else:
            # nimare did not overwrite any coordinates
            metadata["coordinate_source"] = ["original"] * metadata.shape[0]

        if self.merge_strategy != "demolish":
            original_idxs = ~dataset.coordinates["id"].isin(coordinates_df["id"])
            old_coordinates_df = dataset.coordinates[original_idxs]
            coordinates_df = pd.concat([coordinates_df, old_coordinates_df], ignore_index=True)

            # specify original coordinates
            original_ids = set(old_coordinates_df["id"])
            metadata.loc[metadata["id"].isin(original_ids), "coordinate_source"] = "original"

        if "z_stat" in coordinates_df.columns:
            # ensure z_stat is treated as float
            coordinates_df["z_stat"] = coordinates_df["z_stat"].astype(float)

            # Raise warning if coordinates dataset contains both positive and negative z_stats
            if ((coordinates_df["z_stat"].values >= 0).any()) and (
                (coordinates_df["z_stat"].values < 0).any()
            ):
                warnings.warn(
                    "Coordinates dataset contains both positive and negative z_stats. "
                    "The algorithms currently implemented in NiMARE are designed for "
                    "one-sided tests. This might lead to unexpected results."
                )

        # Append the generated foci to the analyses they belong to, and record
        # where they came from as a metadata column.
        row_of = dataset.row_of_id()
        extra_cols = [
            c
            for c in coordinates_df.columns
            if c not in ("id", "study_id", "contrast_id", "x", "y", "z", "space")
        ]
        positions, xyz = [], []
        point_values = {name: [] for name in extra_cols}
        for i, (analysis_id, x, y, z) in enumerate(
            zip(
                coordinates_df["id"],
                coordinates_df["x"],
                coordinates_df["y"],
                coordinates_df["z"],
            )
        ):
            row = row_of.get(str(analysis_id))
            if row is None:
                continue
            positions.append(row)
            xyz.append([x, y, z])
            for name in extra_cols:
                value = coordinates_df[name].iloc[i]
                point_values[name].append(None if value != value else value)

        if self.merge_strategy == "demolish":
            base = dataset.select_points(
                np.zeros(dataset.store.n_points, dtype=bool)
            ).materialize_points()
        else:
            base = dataset
        new_dataset = (
            base.with_points(
                positions,
                xyz,
                space=space,
                kind="center of mass",
                values=point_values or None,
            )
            if positions
            else base
        )
        source = metadata.set_index("id")["coordinate_source"]
        new_dataset = new_dataset.with_metadata(
            "coordinate_source",
            [source.get(str(key), "original") for key in new_dataset.ids],
        )
        return new_dataset


class StandardizeField(NiMAREBase):
    """Standardize metadata fields."""

    def __init__(self, fields):
        self.fields = fields  # the fields to be standardized

    def transform(self, dataset):
        """Standardize metadata fields.

        .. warning::
            Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed
            in a future release. Prefer :class:`~nimare.nimads.Studyset`.
        """
        dataset = normalize_collection(dataset)

        categorical_metadata, numerical_metadata = [], []
        for metadata_name in self.fields:
            if np.array_equal(
                dataset.annotations_df[metadata_name],
                dataset.annotations_df[metadata_name].astype(str),
            ):
                categorical_metadata.append(metadata_name)
            elif np.array_equal(
                dataset.annotations_df[metadata_name],
                dataset.annotations_df[metadata_name].astype(float),
            ):
                numerical_metadata.append(metadata_name)
        if len(categorical_metadata) > 0:
            LGR.warning(f"Categorical metadata {categorical_metadata} can't be standardized.")
        if len(numerical_metadata) == 0:
            raise ValueError("No numerical metadata found.")

        annot_df = dataset.annotations_df
        moderators = annot_df[numerical_metadata].astype(float)
        standardized = moderators - np.mean(moderators, axis=0)
        standardized /= np.std(standardized, axis=0)
        labels = ["standardized_" + moderator for moderator in numerical_metadata]
        return dataset.with_annotation(
            "standardized",
            labels,
            standardized.to_numpy(dtype=float),
            note_key_types={label: "number" for label in labels},
        )


def sample_sizes_to_dof(sample_sizes):
    """Calculate degrees of freedom from a list of sample sizes using a simple heuristic.

    .. versionadded:: 0.0.4

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
    """Calculate appropriate sample size from a list of sample sizes using a simple heuristic.

    .. versionadded:: 0.0.4

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

    .. versionadded:: 0.0.3

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

    .. versionadded:: 0.0.3

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
    varcope = se**2
    return varcope


def samplevar_dataset_to_varcope(samplevar_dataset, sample_size):
    """Convert "sample variance of the dataset" to "sampling variance".

    .. versionadded:: 0.0.3

    Parameters
    ----------
    samplevar_dataset : array_like
        Sample variance of the dataset (i.e., variance of the individual observations in a single
        sample). Can be calculated with ``np.var``.
    sample_size : int
        Sample size

    Returns
    -------
    varcope : array_like
        Sampling variance of the parameter (i.e., variance of sampling distribution for the
        parameter).

    Notes
    -----
    Sampling variance is sample variance divided by sample size.
    """
    varcope = samplevar_dataset / sample_size
    return varcope


def t_and_varcope_to_beta(t, varcope):
    """Convert t-statistic to parameter estimate using sampling variance.

    .. versionadded:: 0.0.3

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

    .. versionadded:: 0.0.4

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


#: ``sqrt(2)`` and ``log(2)``, which every tail below is expressed in terms of.
_SQRT2 = np.sqrt(2.0)
_LOG2 = np.log(2.0)


def _log_erfc(y):
    """Return ``log(erfc(y))``, staying finite where ``erfc`` itself underflows.

    Every normal and chi-squared tail NiMARE reports reduces to one of these.
    :func:`scipy.special.erfc` is accurate to a relative 1e-16 wherever it is
    representable, so its logarithm is accurate to about that absolutely, and it costs a
    third of what :func:`scipy.special.log_ndtr` does -- which matters, because these run
    once per voxel. Past ``y = 25.4`` the value is denormal or zero, and there only
    ``log_ndtr`` still carries it; the two agree to 2e-14 relative across the join.
    """
    shape = np.shape(y)
    y = np.atleast_1d(np.asarray(y, dtype=float))
    tail = special.erfc(y)
    with np.errstate(divide="ignore"):
        out = np.log(tail)

    underflowed = tail < np.finfo(float).tiny
    if underflowed.any():
        out[underflowed] = _LOG2 + special.log_ndtr(-y[underflowed] * _SQRT2)

    return out.reshape(shape)


def z_to_nlogp(z, tail="two"):
    """Convert z-values to ``nlogp``, the natural logarithm of the p-value.

    .. versionadded:: 0.21.0

    The log-space counterpart of :func:`z_to_p`.

    Parameters
    ----------
    z : array_like
        Z-statistics.
    tail : {'one', 'two'}, optional
        Whether p-values come from a one-tailed or two-tailed test. Default is 'two'.

    Returns
    -------
    nlogp : array_like
        Natural logarithms of the p-values, matching SciPy's ``logsf``. Natural and not
        negated, unlike the ``logp`` NiMARE's *maps* hold, which is ``-log10(p)``.

    See Also
    --------
    z_to_p : The same conversion, returning the p-value itself.
    nlogp_to_z : The inverse.
    """
    z = np.asarray(z, dtype=float)
    if tail == "two":
        # Twice the upper tail is exactly erfc(|z| / sqrt(2)), so doubling costs nothing.
        nlogp = _log_erfc(np.abs(z) / _SQRT2)
    elif tail == "one":
        nlogp = _log_erfc(z / _SQRT2) - _LOG2
    else:
        raise ValueError('Argument "tail" must be one of ["one", "two"]')

    # The cap is against log(1), which the doubled tail can exceed only by rounding at a z
    # of essentially zero. It is not a floor on the tail.
    nlogp = np.minimum(nlogp, 0.0)
    if nlogp.shape == ():
        nlogp = nlogp[()]
    return nlogp


def z_to_p(z, tail="two"):
    """Convert z-values to p-values.

    .. versionadded:: 0.0.8

    Parameters
    ----------
    z : array_like
        Z-statistics
    tail : {'one', 'two'}, optional
        Whether p-values come from one-tailed or two-tailed test. Default is
        'two'.

    Returns
    -------
    p : array_like
        P-values, floored at the smallest positive double. Use :func:`z_to_nlogp` where the
        tail runs past that.
    """
    p = np.exp(z_to_nlogp(z, tail=tail))
    p = _clip_p_values(p, dtype=np.asarray(p).dtype, copy=False)
    if p.shape == ():
        p = p[()]
    return p


def nlogp_to_z(nlogp, tail="two"):
    """Convert ``nlogp``, the natural logarithm of a p-value, to (unsigned) z-values.

    .. versionadded:: 0.21.0

    The log-space counterpart of :func:`p_to_z`.

    Parameters
    ----------
    nlogp : array_like
        Natural logarithms of the p-values, as :func:`z_to_nlogp` returns them.
    tail : {'one', 'two'}, optional
        Whether the p-values come from a one-tailed or two-tailed test. Default is 'two'.

    Returns
    -------
    z : array_like
        Z-statistics (unsigned).

    See Also
    --------
    p_to_z : The same conversion for a p-value that is already a number.
    z_to_nlogp : The inverse.
    """
    nlogp = np.asarray(nlogp, dtype=float)
    if tail == "two":
        # p_to_z halves a two-tailed p before inverting; in log space that is a subtraction,
        # which cannot underflow. The trailing addition turns the -0.0 that p == 1 produces
        # back into 0.0.
        z = -special.ndtri_exp(nlogp - np.log(2.0)) + 0.0
    elif tail == "one":
        z = np.maximum(-special.ndtri_exp(nlogp), 0.0)
    else:
        raise ValueError('Argument "tail" must be one of ["one", "two"]')

    z = np.asarray(z)
    if z.shape == ():
        z = z[()]
    return z


def p_to_z(p, tail="two"):
    """Convert p-values to (unsigned) z-values.

    .. versionchanged:: 0.21.0

        Evaluated in log space, and the input is no longer routed through a float32
        p-value, which bounded ``z`` at 14.12. The bound is now the float64 one, 38.5;
        :func:`nlogp_to_z` has no bound.

    .. versionadded:: 0.0.3

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
    p = _clip_p_values(p, dtype=np.float64)
    return nlogp_to_z(np.log(p), tail=tail)


def t_to_nlogp(t_values, dof, tail="two"):
    """Convert t-statistics to ``nlogp``, the natural logarithm of the p-value.

    .. versionadded:: 0.21.0

    The only place NiMARE evaluates the t tail, and unbounded where the p-value itself
    underflows to zero.

    Parameters
    ----------
    t_values : array_like
        T-statistics.
    dof : int
        Degrees of freedom.
    tail : {'one', 'two'}, optional
        Whether p-values come from a one-tailed or two-tailed test. Default is 'two'.

    Returns
    -------
    nlogp : array_like
        Natural logarithms of the p-values, as :func:`z_to_nlogp` returns them.
    """
    t_values = np.asarray(t_values, dtype=float)
    if tail == "two":
        nlogp = _LOG2 + _log_t_sf(np.abs(t_values), dof)
    elif tail == "one":
        nlogp = _log_t_sf(t_values, dof)
    else:
        raise ValueError('Argument "tail" must be one of ["one", "two"]')

    nlogp = np.minimum(nlogp, 0.0)
    if nlogp.shape == ():
        nlogp = nlogp[()]
    return nlogp


def t_to_z(t_values, dof):
    """Convert t-statistics to z-statistics.

    .. versionchanged:: 0.21.0

        The tail is evaluated in log space, so ``|z|`` is no longer bounded at 8.13 by an
        epsilon floor on the internal p-value, nor by ``scipy.stats.t.logsf`` underflowing at
        high ``dof``. Values below those bounds are unchanged.

    .. versionadded:: 0.0.3

    Parameters
    ----------
    t_values : array_like
        T-statistics
    dof : int
        Degrees of freedom

    Returns
    -------
    z_values : array_like
        Z-statistics, carrying the sign of ``t_values``.

    Notes
    -----
    The t and normal tail probabilities are matched, which is the transform of
    :footcite:t:`hughett2008accurate`. Non-finite input propagates.

    References
    ----------
    .. footbibliography::
    """
    t_values = np.asarray(t_values, dtype=float)

    nlogp = t_to_nlogp(np.abs(t_values), dof, tail="one")
    z_values = np.sign(t_values) * nlogp_to_z(nlogp, tail="one") + 0.0

    if z_values.shape == ():
        z_values = z_values[()]
    return z_values


def _log_t_power_law(dof):
    """Return ``log C`` in the far t tail, where ``sf(t) -> C * t ** -dof``.

    Both t conversions bottom out in this expansion. The neglected term is of relative order
    ``dof / t**2``, so it is exact to machine precision by the time a p-value has stopped
    being representable, and useless for a shallow tail.
    """
    return (
        special.gammaln((dof + 1) / 2.0)
        - special.gammaln(dof / 2.0)
        - 0.5 * np.log(np.pi)
        + ((dof - 2) / 2.0) * np.log(dof)
    )


def _log_t_sf(t, dof):
    """Return ``log(sf(t))`` for the t distribution, finite where SciPy's underflows.

    :meth:`scipy.stats.rv_continuous.logsf` computes the survival function and takes its
    logarithm afterwards, so it returns ``-inf`` as soon as that underflows -- at ``t = 96``
    for ``dof = 500``, and ``t = 591`` for ``dof = 200``, which would put an infinity in a
    z map. Past those points :func:`_log_t_power_law` carries the tail.
    """
    shape = np.shape(t)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    with np.errstate(divide="ignore"):
        out = np.asarray(stats.t.logsf(t, dof), dtype=float).copy()

    underflowed = ~np.isfinite(out) & np.isfinite(t) & (t > 0)
    if underflowed.any():
        out[underflowed] = _log_t_power_law(dof) - dof * np.log(t[underflowed])

    return out.reshape(shape)


def _asymptotic_t_isf(nlogp, dof):
    """Invert the far t tail from an ``nlogp``, where a p-value cannot reach.

    Inverts :func:`_log_t_power_law`: ``log t = (log C - nlogp) / dof``. Accurate to 1e-15
    in ``log t`` at the smallest representable p-value for ``dof`` up to 20, and to 1e-7 at
    ``dof = 100``; correspondingly poor for a shallow tail, which is the region
    :meth:`scipy.stats.rv_continuous.isf` handles.
    """
    # An overflow here is the honest answer: at dof = 1 and |z| = 50 the matched t is 1e544.
    with np.errstate(over="ignore"):
        return np.exp((_log_t_power_law(dof) - nlogp) / dof)


def z_to_t(z_values, dof):
    """Convert z-statistics to t-statistics.

    .. versionchanged:: 0.21.0

        Evaluated from an ``nlogp`` rather than from a p-value floored at machine epsilon,
        which had saturated ``t`` from ``|z| = 8.13`` on.

    .. versionadded:: 0.0.3

    Parameters
    ----------
    z_values : array_like
        Z-statistics
    dof : int
        Degrees of freedom

    Returns
    -------
    t_values : array_like
        T-statistics, carrying the sign of ``z_values``.

    Notes
    -----
    The t and normal tail probabilities are matched, as in :func:`t_to_z`. SciPy has no
    log-space inverse t, so two routes are tried and the one that reproduces the tail it was
    inverted from is kept: :meth:`scipy.stats.rv_continuous.isf`, which needs a representable
    p-value, and :func:`_asymptotic_t_isf`, which does not. Round-tripping ``t_to_z`` is exact
    to 1e-11 for the degrees of freedom a meta-analysis produces, and to 1% in the narrow
    band where the p-value has underflowed but the expansion is not yet tight.

    An infinity therefore means the matched ``t`` exceeds a double -- 1e544 at ``dof = 1``,
    ``|z| = 50`` -- rather than that the conversion gave up.

    References
    ----------
    .. footbibliography::
    """
    z_values = np.asarray(z_values, dtype=float)

    # As in t_to_z: invert the one-tailed tail of |z| and reapply the sign.
    nlogp = z_to_nlogp(np.abs(z_values), tail="one")
    magnitude = _asymptotic_t_isf(nlogp, dof)

    # Keep whichever candidate actually reproduces the tail it was inverted from. The
    # expansion is exact deep in the tail and loose in the shallow part; SciPy's inverse t is
    # the reverse, needs a representable p-value besides, and degrades before it runs out of
    # one -- 142% off at dof = 100, p = 4e-202. Scoring both against the forward tail picks
    # the right one at every dof without a tuned crossover.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        from_p = stats.t.isf(np.exp(nlogp), dof)
        magnitude = np.where(
            np.abs(_log_t_sf(from_p, dof) - nlogp) <= np.abs(_log_t_sf(magnitude, dof) - nlogp),
            from_p,
            magnitude,
        )

    t_values = np.sign(z_values) * magnitude + 0.0

    t_values = np.asarray(t_values)
    if t_values.shape == ():
        t_values = t_values[()]
    return t_values


def chi2_to_nlogp(chi2_values, dof):
    """Convert chi-squared statistics to ``nlogp``, the natural logarithm of the p-value.

    .. versionadded:: 0.21.0

    The upper tail, which is the only one a chi-squared test uses.

    Parameters
    ----------
    chi2_values : array_like
        Chi-squared statistics.
    dof : :obj:`int`
        Degrees of freedom.

    Returns
    -------
    nlogp : array_like
        Natural logarithms of the p-values, as :func:`z_to_nlogp` returns them.

    Notes
    -----
    One degree of freedom, the common case here, is evaluated through :func:`_log_erfc`.
    The general case defers to :func:`pymare.stats.log_chi2_sf`, which is equally accurate
    but reaches the non-underflowing majority of values through
    :meth:`scipy.stats.rv_continuous.logsf` and so costs about a microsecond each -- two
    orders of magnitude more, which voxelwise use cannot afford.
    """
    chi2_values = np.asarray(chi2_values, dtype=float)
    if np.ndim(dof) == 0 and dof == 1:
        # A chi-squared on one degree of freedom is a squared standard normal, so its upper
        # tail is the two-tailed normal tail at sqrt(x), i.e. erfc(sqrt(x / 2)). The cap is
        # against log(1), which rounding can reach at a statistic of essentially zero.
        return np.minimum(_log_erfc(np.sqrt(chi2_values / 2.0)), 0.0)

    return log_chi2_sf(chi2_values, dof)


def t_to_d(t_values, sample_sizes):
    """Convert t-statistics to Cohen's d.

    Parameters
    ----------
    t_values : array_like
        T-statistics
    sample_sizes : array_like
        Sample sizes

    Returns
    -------
    d_values : array_like
        Cohen's d
    """
    d_values = t_values / np.sqrt(sample_sizes)
    return d_values


def d_to_g(d, N, return_variance=False):
    """Convert Cohen's d to Hedges' g.

    Parameters
    ----------
    d : array_like
        Cohen's d
    N : array_like
        Sample sizes
    return_variance : bool, optional
        Whether to return the variance of Hedges' g. Default is False.

    Returns
    -------
    g_values : array_like
        Hedges' g
    """
    # Calculate bias correction h(N)
    h = 1 - (3 / (4 * (N - 1) - 1))

    if return_variance:
        return d * h, ((N - 1) * (1 + N * d**2) * (h**2) / (N * (N - 3))) - d**2

    return d * h
