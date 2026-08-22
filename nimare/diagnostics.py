"""Methods for diagnosing problems in meta-analytic datasets or analyses."""

import copy
import logging
import warnings
from abc import abstractmethod

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn.maskers import NiftiLabelsMasker
from nilearn.reporting import get_clusters_table
from tqdm.auto import tqdm

from nimare.base import NiMAREBase
from nimare.meta.cbma.base import (
    CBMAEstimator,
    PairwiseCBMAEstimator,
    _approximate_z_from_ma,
)
from nimare.meta.cbma.utils import (
    _threshold_z_clusters,
    generate_subset_schedule,
    resolve_subset_size,
)
from nimare.meta.ibma import IBMAEstimator
from nimare.results import MetaResult
from nimare.studyset import normalize_collection
from nimare.studyset.layout import harmonized_coordinates
from nimare.utils import (
    DEFAULT_FLOAT_DTYPE,
    _check_ncores,
    _filter_kwargs,
    _mask_coverage_to_null_ijk,
    _mask_img_to_bool,
    get_masker,
    mm2vox,
)

LGR = logging.getLogger(__name__)

POSTAIL_LBL = "PositiveTail"  # Label assigned to positive tail clusters
NEGTAIL_LBL = "NegativeTail"  # Label assigned to negative tail clusters


def _tail_mappings():
    """Return tail/sign mappings for consistent diagnostics labeling."""
    tail_to_sign = {"positive": POSTAIL_LBL, "negative": NEGTAIL_LBL}
    sign_to_tail = {POSTAIL_LBL: "positive", NEGTAIL_LBL: "negative"}
    return tail_to_sign, sign_to_tail


def _get_target_value_map(result):
    """Select the map to use for per-cluster contribution calculations."""
    # CBMAs have "stat" maps, while most IBMAs have "est" maps. ALESubtraction has
    # "stat_desc-group1MinusGroup2" maps, while MKDAChi2 has "z_desc-association" maps.
    # Fisher's and Stouffer's only have "z" maps though.
    target_value_keys = ("stat", "est", "stat_desc-group1MinusGroup2", "z_desc-association", "z")
    for target_value_key in target_value_keys:
        if target_value_key in result.maps:
            return target_value_key

    available_maps = ", ".join(sorted(result.maps.keys()))
    raise ValueError(
        "No supported map found for per-cluster contribution calculations. "
        f"Expected one of {target_value_keys}; available maps are: {available_maps}."
    )


def _resolve_target_threshold(target_threshold, voxel_thresh):
    """Resolve diagnostics threshold aliases."""
    if target_threshold is not None and voxel_thresh is not None:
        raise ValueError(
            "Only one of 'target_threshold' and deprecated 'voxel_thresh' may be provided."
        )

    if voxel_thresh is not None:
        warnings.warn(
            "'voxel_thresh' is deprecated for diagnostics and will be removed in a future "
            "release. Use 'target_threshold' to threshold the selected target image before "
            "diagnostics table/support generation.",
            FutureWarning,
            stacklevel=3,
        )
        return voxel_thresh

    return target_threshold


def _is_cluster_corrected_target(target_image):
    """Determine whether a target image is a corrected cluster-level map."""
    return "_level-cluster" in target_image and "_corr-" in target_image


def _strip_cluster_stat_description(map_name):
    """Remove cluster-size/mass description components from a map name."""
    if "_desc-" not in map_name:
        return map_name

    prefix, remainder = map_name.split("_desc-", 1)
    description, separator, suffix = remainder.partition("_")

    if description in {"size", "mass"}:
        return prefix + (f"_{suffix}" if separator else "")

    for cluster_stat in ("Size", "Mass", "size", "mass"):
        if description.endswith(cluster_stat):
            description = description[: -len(cluster_stat)]
            break

    if not description:
        return prefix + (f"_{suffix}" if separator else "")

    return f"{prefix}_desc-{description}" + (f"_{suffix}" if separator else "")


def _peak_value_map_from_cluster_target(target_image):
    """Derive the original statistic map from a corrected cluster target name."""
    uncorrected_target = target_image.split("_corr-", 1)[0]
    uncorrected_target = uncorrected_target.replace("_level-cluster", "")
    return _strip_cluster_stat_description(uncorrected_target)


def _get_peak_value_map_for_cluster_table(result, target_image):
    """Select the original map used for peak statistics in corrected-cluster tables."""
    peak_value_map = _peak_value_map_from_cluster_target(target_image)
    if "_level-cluster" not in peak_value_map and peak_value_map in result.maps:
        return peak_value_map

    available_maps = ", ".join(sorted(result.maps.keys()))
    raise ValueError(
        "No supported original z/statistic map found for corrected-cluster table peaks. "
        f"Expected '{peak_value_map}' derived from target image '{target_image}'. "
        f"Available maps are: {available_maps}."
    )


def _pad_background_border(img):
    """Add a one-voxel zero border so cluster labeling always has a background label.

    :func:`nilearn.reporting.get_clusters_table` derives cluster ids with
    ``np.unique(label_map)[1:]``, which assumes a background (0) label is present. When
    every voxel of a map survives thresholding, that assumption fails: the only label is
    dropped, and building the label maps raises an IndexError. A zero border restores the
    assumption, and shifting the affine to match keeps every voxel's world coordinates.
    """
    data = np.asanyarray(img.dataobj)
    padded_data = np.pad(data, 1, mode="constant", constant_values=0)

    affine = img.affine.copy()
    affine[:3, 3] -= img.affine[:3, :3] @ np.ones(3)
    return nib.Nifti1Image(padded_data, affine)


def _crop_background_border(img, reference_img):
    """Undo :func:`_pad_background_border` for an image derived from a padded one."""
    data = np.asanyarray(img.dataobj)[1:-1, 1:-1, 1:-1]
    return nib.Nifti1Image(data, reference_img.affine, reference_img.header)


def _needs_background_border(img, threshold):
    """Determine whether thresholding ``img`` would leave no background voxels."""
    data = np.asanyarray(img.dataobj)
    threshold = 0 if threshold is None else abs(threshold)
    return bool((data > threshold).all() or (data < -threshold).all())


def _get_clusters_table(img, threshold, cluster_threshold, two_sided, return_label_maps=False):
    """Call nilearn's ``get_clusters_table``, guarding the no-background-voxel case."""
    padded = _needs_background_border(img, threshold)
    result = get_clusters_table(
        _pad_background_border(img) if padded else img,
        threshold,
        cluster_threshold,
        two_sided=two_sided,
        return_label_maps=return_label_maps,
    )

    if not (padded and return_label_maps):
        return result

    clusters_table, label_maps = result
    return clusters_table, [_crop_background_border(lm, img) for lm in label_maps]


def _get_cluster_support_data(label_maps, shape):
    """Convert one or more cluster label maps to a binary support array."""
    support = np.zeros(shape, dtype=bool)
    for label_map in label_maps:
        support |= np.asanyarray(label_map.dataobj) > 0
    return support


def _get_clusters_table_and_label_maps(
    result,
    target_img,
    target_image,
    threshold,
    cluster_threshold,
):
    """Create diagnostics clusters from target image and peaks from original statistics."""
    target_data = target_img.get_fdata(dtype=DEFAULT_FLOAT_DTYPE)
    if hasattr(result.estimator, "two_sided"):
        # Only present in Fisher's and Stouffer's estimators
        two_sided = getattr(result.estimator, "two_sided")
    else:
        two_sided = (target_data < 0).any()

    clusters_table, label_maps = _get_clusters_table(
        target_img,
        0 if threshold is None else threshold,
        cluster_threshold,
        two_sided,
        return_label_maps=True,
    )

    if clusters_table.empty or not label_maps or not _is_cluster_corrected_target(target_image):
        return clusters_table, label_maps

    peak_value_map = _get_peak_value_map_for_cluster_table(result, target_image)
    peak_img = result.get_map(peak_value_map, return_type="image")
    peak_data = peak_img.get_fdata(dtype=DEFAULT_FLOAT_DTYPE)
    support_data = _get_cluster_support_data(label_maps, peak_data.shape)
    masked_peak_data = np.where(support_data, peak_data, 0).astype(DEFAULT_FLOAT_DTYPE, copy=False)
    masked_peak_img = nib.Nifti1Image(masked_peak_data, peak_img.affine, peak_img.header)

    peak_clusters_table = _get_clusters_table(
        masked_peak_img,
        0,
        0,
        two_sided or (masked_peak_data < 0).any(),
    )
    return peak_clusters_table, label_maps


def _count_foci_per_cluster(label_arr, clust_ids, ijk):
    """Count how many of ``ijk`` fall within one voxel of each cluster in ``label_arr``.

    A focus counts towards a cluster when some voxel of that cluster lies less than one
    voxel away from it, which is the same rule as measuring every cluster voxel's distance
    to every focus and keeping the foci with a hit. Doing it that way costs one pass over the
    whole volume per cluster, so a map with a few hundred clusters spends minutes on
    distances that a focus's own neighbourhood already settles: only integer voxels strictly
    inside a unit ball around the focus can qualify, and there are at most 27 of those.

    Parameters
    ----------
    label_arr : 3D :obj:`numpy.ndarray`
        Cluster label map. Zero is background.
    clust_ids : :obj:`list`
        Cluster labels to count for, in the order the counts are returned in.
    ijk : (N, 3) array_like
        Matrix subscripts of the foci, which need not be integers.

    Returns
    -------
    :obj:`numpy.ndarray`
        One count per entry of ``clust_ids``.
    """
    counts = dict.fromkeys(clust_ids, 0)
    ijk = np.atleast_2d(np.asarray(ijk, dtype=float))
    if ijk.size == 0:
        return np.array([counts[c_val] for c_val in clust_ids])

    shape = np.asarray(label_arr.shape)
    # The 27 integer offsets around a focus's containing voxel. Every voxel within a unit
    # distance of any point in that voxel is among them.
    offsets = np.stack(np.meshgrid(*([np.arange(-1, 2)] * 3), indexing="ij"), axis=-1)
    offsets = offsets.reshape(-1, 3)

    for focus in ijk:
        candidates = np.floor(focus).astype(np.int64) + offsets
        in_bounds = np.all((candidates >= 0) & (candidates < shape), axis=1)
        candidates = candidates[in_bounds]
        if not candidates.size:
            continue

        near = np.sum(np.square(candidates - focus), axis=1) < 1.0
        candidates = candidates[near]
        if not candidates.size:
            continue

        labels = label_arr[candidates[:, 0], candidates[:, 1], candidates[:, 2]]
        for c_val in np.unique(labels):
            if c_val in counts:
                counts[c_val] += 1

    return np.array([counts[c_val] for c_val in clust_ids])


def _cluster_masker_kwargs():
    """Return standardized kwargs for label-based cluster summaries."""
    return _filter_kwargs(
        NiftiLabelsMasker,
        {
            "standardize": False,
            "detrend": False,
            "smoothing_fwhm": None,
            "dtype": DEFAULT_FLOAT_DTYPE,
        },
    )


def _get_masker_voxel_count(masker):
    """Infer and cache the voxel count represented by a masker."""
    cached_count = getattr(masker, "_nimare_mask_voxel_count", None)
    if cached_count is not None:
        return cached_count

    mask_img = getattr(masker, "mask_img_", None)
    if mask_img is None:
        return None

    try:
        mask_data = np.asanyarray(mask_img.dataobj).astype(bool, copy=False)
        cached_count = int(mask_data.sum())
    except Exception:
        return None

    setattr(masker, "_nimare_mask_voxel_count", cached_count)
    return cached_count


def _is_voxelwise_masker(masker, n_features):
    """Determine whether a masker output is voxelwise in masked-array space."""
    n_mask_voxels = _get_masker_voxel_count(masker)
    if n_mask_voxels == n_features:
        return True

    try:
        probe_values = np.zeros(int(n_features), dtype=DEFAULT_FLOAT_DTYPE)
        round_trip_values = np.squeeze(masker.transform(masker.inverse_transform(probe_values)))
    except Exception:
        return False

    return round_trip_values.ndim == 1 and round_trip_values.shape[0] == n_features


def _cluster_ids(label_arr):
    """Return the cluster labels in a label map, in ascending order.

    Selects the positive labels rather than dropping the first unique value, which would
    discard a real cluster in a map that has no background label left.
    """
    label_arr = np.asanyarray(label_arr)
    return np.unique(label_arr[label_arr > 0]).tolist()


def _build_cluster_summary_context(masker, label_map, label_vector, cluster_ids):
    """Precompute cluster summaries in array space when possible.

    The returned context carries ``cluster_ids`` so every consumer reads the same list.
    """
    label_vector = np.squeeze(np.asarray(label_vector))
    if _is_voxelwise_masker(masker, label_vector.shape[0]):
        rounded_labels = np.rint(label_vector).astype(np.int32, copy=False)
        positive_indices = np.flatnonzero(rounded_labels > 0)
        positive_labels = rounded_labels[positive_indices]

        order = np.argsort(positive_labels, kind="stable")
        sorted_indices = positive_indices[order]
        sorted_labels = positive_labels[order]
        unique_labels, starts = np.unique(sorted_labels, return_index=True)
        grouped_indices = np.split(sorted_indices, starts[1:]) if unique_labels.size else []
        label_to_indices = dict(zip(unique_labels.tolist(), grouped_indices))

        cluster_indices = [
            label_to_indices.get(int(c_id), np.array([], dtype=np.int32)).astype(
                np.int32, copy=False
            )
            for c_id in cluster_ids
        ]
        if all(cluster_idx.size > 0 for cluster_idx in cluster_indices):
            return {
                "mode": "masked_array",
                "cluster_ids": list(cluster_ids),
                "cluster_indices": cluster_indices,
            }

    cluster_masker = NiftiLabelsMasker(label_map, **_cluster_masker_kwargs())
    cluster_masker.fit(label_map)
    return {
        "mode": "image",
        "cluster_ids": list(cluster_ids),
        "cluster_masker": cluster_masker,
    }


def _summarize_cluster_values(values, masker, cluster_summary_context):
    """Reduce per-feature values to cluster-level means."""
    if cluster_summary_context["mode"] == "masked_array":
        return np.array(
            [
                np.mean(values[cluster_idx])
                for cluster_idx in cluster_summary_context["cluster_indices"]
            ],
            dtype=DEFAULT_FLOAT_DTYPE,
        )

    stat_prop_img = masker.inverse_transform(values)
    stat_prop_values = cluster_summary_context["cluster_masker"].transform(stat_prop_img)
    return stat_prop_values.flatten()


def _infer_label_map_tails(label_maps, clusters_table, n_clusters):
    """Infer tail labels from label maps and cluster statistics."""
    inferred_tail = "positive"
    mixed_signs = False

    if len(label_maps) == 2:
        return ["positive", "negative"], inferred_tail, mixed_signs

    if len(label_maps) == 1 and n_clusters > 0:
        peak_stats = clusters_table["Peak Stat"].astype(float)
        has_pos = (peak_stats > 0).any()
        has_neg = (peak_stats < 0).any()
        if has_pos and not has_neg:
            inferred_tail = "positive"
        elif has_neg and not has_pos:
            inferred_tail = "negative"
        else:
            mixed_signs = True
        return [inferred_tail], inferred_tail, mixed_signs

    return [inferred_tail], inferred_tail, mixed_signs


class Diagnostics(NiMAREBase):
    """Base class for diagnostic methods.

    .. versionchanged:: 0.1.2

        * New parameter display_second_group, which controls whether the second group is displayed.

    .. versionchanged:: 0.1.0

        * Transform now returns a MetaResult object.

    .. versionadded:: 0.0.14

    Parameters
    ----------
    target_image : :obj:`str`, optional
        The meta-analytic map for which clusters will be characterized.
        The default is z because log-p will not always have value of zero for non-cluster voxels.
    voxel_thresh : :obj:`float` or None, optional
        Deprecated alias for ``target_threshold``. Prefer ``target_threshold`` for new code.
    cluster_threshold : :obj:`int` or None, optional
        Cluster size threshold, in :term:`voxels<voxel>`.
        If None, then no cluster size threshold will be applied. Default=None.
    n_cores : :obj:`int`, optional
        Number of cores to use for parallelization.
        If <=0, defaults to using all available cores.
        Default is 1.
    target_threshold : :obj:`float` or None, optional
        Threshold applied to ``target_image`` before defining diagnostics clusters and tables.
        For unthresholded Monte Carlo cluster-corrected maps, this should generally be the
        corrected significance threshold in the target map's units. This is distinct from
        :class:`~nimare.correct.FWECorrector` ``voxel_thresh``, which is the cluster-forming
        threshold used during correction. Default=None.

    """

    def __init__(
        self,
        target_image="z_desc-size_level-cluster_corr-FWE_method-montecarlo",
        voxel_thresh=None,
        cluster_threshold=None,
        display_second_group=False,
        n_cores=1,
        target_threshold=None,
    ):
        self.target_image = target_image
        self.voxel_thresh = voxel_thresh
        self.target_threshold = _resolve_target_threshold(target_threshold, voxel_thresh)
        self.cluster_threshold = cluster_threshold
        self.display_second_group = display_second_group
        self.n_cores = _check_ncores(n_cores)

    @abstractmethod
    def _transform(
        self,
        expid,
        label_map,
        sign,
        result,
        target_value_map=None,
        cluster_summary_context=None,
    ):
        """Apply transform to study ID and label map.

        Must return a 1D array with the contribution of `expid` in each cluster of `label_map`.
        """

    def _batch_tail_contexts(self, tail_contexts):
        """Group tails whose per-experiment computation is interchangeable.

        Returns a list of batches, each a list of indices into ``tail_contexts``. Tails land
        in the same batch when they are fit over the same experiments and, for pairwise
        estimators, over the same group. Batching lets ``_transform_batch`` do a diagnostic's
        expensive per-experiment work once for every tail it applies to.
        """
        batches = {}
        for i_context, context in enumerate(tail_contexts):
            # Pairwise estimators refit a different group per tail, so their tails can never
            # share work. Non-pairwise estimators refit the same experiments for both tails.
            sign = context["sign"] if self._is_pairwaise_estimator else None
            batches.setdefault((sign, tuple(context["meta_ids"])), []).append(i_context)

        return list(batches.values())

    def _transform_batch(
        self, expid, tail_contexts, result, target_value_map=None, image_cache=None
    ):
        """Apply transform to one study ID for each tail in a batch.

        Returns one 1D array per entry in ``tail_contexts``, in the same order. Subclasses
        whose per-experiment work does not vary across a batch should override this to do
        that work once and derive every tail's contributions from it.

        ``image_cache`` is a store the caller shares across every experiment of one
        :meth:`transform`, for subclasses that would otherwise redo work per experiment that
        does not depend on the experiment. Diagnostics that read no images ignore it.
        """
        return [
            self._transform(
                expid,
                context["label_map"],
                context["sign"],
                result,
                target_value_map,
                context["cluster_summary_context"],
            )
            for context in tail_contexts
        ]

    def transform(self, result):
        """Apply the analysis to a MetaResult.

        Parameters
        ----------
        result : :obj:`~nimare.results.MetaResult`
            A MetaResult produced by a coordinate- or image-based meta-analysis.

        Returns
        -------
        :obj:`~nimare.results.MetaResult`
            Results of Diagnostics fitting.

        Notes
        -----
        This method adds two new keys to ``maps`` and ``tables`` attributes of the
        MetaResult object.

            -   ``<target_image>_diag-<Jackknife|FocusCounter>_tab-counts`` :
                :obj:`pandas.DataFrame` or None.
                A DataFrame with information about relative contributions of each experiment
                to each cluster in the thresholded map.
                There is one row for each experiment.
                There is one column for each cluster, with column names being
                ``PostiveTail``/``NegativeTail`` indicating the sign (+/-) of the cluster's
                statistical values, plus an integer indicating the cluster's associated value
                in the ``label_maps[0]``/``label_maps[1]`` output.
                If no clusters are found or a pairwise Estimator was used, ``None`` is returned.
            -   ``<target_image>_tab-clust`` : :obj:`pandas.DataFrame`
                A DataFrame with information about each cluster.
                There is one row for each cluster.
                The columns in this table include: ``Cluster ID`` (the cluster id, plus a letter
                for subpeaks only), ``X``/``Y``/``Z`` (coordinate for the center of mass),
                ``Max Stat`` (statistical value of the peak), and ``Cluster Size (mm3)``
                (the size of the cluster, in cubic millimeters).
                If no clusters are found, this table will be empty.
            -   ``label_<target_image>_tail-<positive|negative>`` : :obj:`numpy.ndarray`
                Label maps.
                Each cluster in the map has a single value, which corresponds to the cluster number
                of the column name in ``contribution_table``.
                If target_image has negative values after thresholding, first and second maps
                correspond to positive and negative tails.
                If no clusters are found, this list will be empty.
        """
        self._is_pairwaise_estimator = issubclass(type(result.estimator), PairwiseCBMAEstimator)
        masker = result.estimator.masker
        diag_name = self.__class__.__name__

        # One store per call, shared by every per-experiment job below. Keeping it local means
        # it is released with this call, rather than living on through the diagnostic that
        # ``result`` keeps a reference to. Its entries are only valid while the input files
        # are unchanged, which is another reason not to let it outlive the call.
        image_cache = {}

        # Collect the thresholded cluster map
        if self.target_image in result.maps:
            target_img = result.get_map(self.target_image, return_type="image")
        else:
            available_maps = [f"'{m}'" for m in result.maps.keys()]
            raise ValueError(
                f"Target image ('{self.target_image}') not present in result. "
                f"Available maps in result are: {', '.join(available_maps)}."
            )

        # Get clusters table and label maps
        cluster_threshold = 0 if self.cluster_threshold is None else self.cluster_threshold

        clusters_table, label_maps = _get_clusters_table_and_label_maps(
            result,
            target_img,
            self.target_image,
            self.target_threshold,
            cluster_threshold,
        )

        n_clusters = clusters_table.shape[0]
        if n_clusters == 0:
            LGR.warning("No clusters found")
        else:
            LGR.info(f"{n_clusters} clusters found")
            # Make sure cluster IDs are strings
            clusters_table = clusters_table.astype({"Cluster ID": "str"})
            # Rename the clusters_table cluster IDs to match the contribution table columns
            clusters_table["Cluster ID"] = [
                (
                    f"{POSTAIL_LBL} {row['Cluster ID']}"
                    if row["Peak Stat"] > 0
                    else f"{NEGTAIL_LBL} {row['Cluster ID']}"
                )
                for _, row in clusters_table.iterrows()
            ]

        # Define bids-like names for tables and maps
        image_name = "_".join(self.target_image.split("_")[1:])
        image_name = f"_{image_name}" if image_name else image_name
        clusters_table_name = f"{self.target_image}_tab-clust"
        contribution_table_name = f"{self.target_image}_diag-{diag_name}_tab-counts"

        tail_to_sign, sign_to_tail = _tail_mappings()
        label_map_tails, inferred_tail, mixed_signs = _infer_label_map_tails(
            label_maps, clusters_table, n_clusters
        )
        if mixed_signs:
            LGR.warning(
                "Mixed-sign clusters detected but only one label map was returned; "
                "assuming %s tail.",
                inferred_tail,
            )

        label_map_names = [f"label{image_name}_tail-{tail}" for tail in label_map_tails]

        # Check number of clusters
        if n_clusters == 0:
            result.tables[clusters_table_name] = clusters_table
            result.tables[contribution_table_name] = None
            result.maps[label_map_names[0]] = None

            result.diagnostics.append(self)
            return result

        tables_dict = {clusters_table_name: clusters_table}
        maps_dict = {
            label_map_name: np.squeeze(masker.transform(label_map))
            for label_map_name, label_map in zip(label_map_names, label_maps)
        }

        # Use study IDs in inputs_ instead of dataset, because we don't want to try fitting the
        # estimator to a study that might have been filtered out by the estimator's criteria.
        # For pairwise estimators, use id1 for positive tail and id2 for negative tail.
        # Run diagnostics with id2 for pairwise estimators and display_second_group=True.
        if self._is_pairwaise_estimator:
            if len(label_maps) == 2:
                if self.display_second_group:
                    meta_ids_lst = [
                        result.estimator.inputs_["id1"],
                        result.estimator.inputs_["id2"],
                    ]
                    signs = [POSTAIL_LBL, NEGTAIL_LBL]
                else:
                    meta_ids_lst = [result.estimator.inputs_["id1"]]
                    signs = [POSTAIL_LBL]
            else:
                # Single-tail pairwise outputs are assigned to a specific group based on tail,
                # regardless of display_second_group (which only applies when two tails exist).
                single_tail = label_map_tails[0]
                meta_ids_lst = [
                    (
                        result.estimator.inputs_["id1"]
                        if single_tail == "positive"
                        else result.estimator.inputs_["id2"]
                    )
                ]
                signs = [tail_to_sign[single_tail]]
        elif len(label_maps) == 2:
            # Non pairwise estimator with two tails (IBMA estimators)
            meta_ids_lst = [result.estimator.inputs_["id"], result.estimator.inputs_["id"]]
            signs = [POSTAIL_LBL, NEGTAIL_LBL]
        else:
            # Non pairwise estimator with one tail (CBMA estimators)
            meta_ids_lst = [result.estimator.inputs_["id"]]
            signs = [tail_to_sign[label_map_tails[0]]]

        target_value_map = _get_target_value_map(result)

        # Build one context per tail up front, so that tails which share the same
        # per-experiment computation can be batched together below.
        tail_contexts = []
        for sign, label_map, label_map_name, meta_ids in zip(
            signs, label_maps, label_map_names, meta_ids_lst
        ):
            cluster_ids = _cluster_ids(label_map.dataobj)
            tail_contexts.append(
                {
                    "sign": sign,
                    "label_map": label_map,
                    "meta_ids": list(meta_ids),
                    "cluster_ids": cluster_ids,
                    "cluster_summary_context": _build_cluster_summary_context(
                        masker,
                        label_map,
                        maps_dict[label_map_name],
                        cluster_ids,
                    ),
                }
            )

        contribution_tables = [None] * len(tail_contexts)
        for batch in self._batch_tail_contexts(tail_contexts):
            batch_contexts = [tail_contexts[i_context] for i_context in batch]
            meta_ids = batch_contexts[0]["meta_ids"]

            # One job per experiment, covering every tail in the batch. Diagnostics whose
            # per-experiment work does not depend on the tail (Jackknife's leave-one-out
            # refit) therefore do that work once instead of once per tail.
            contributions = [
                r
                for r in tqdm(
                    Parallel(return_as="generator", n_jobs=self.n_cores)(
                        delayed(self._transform_batch)(
                            expid,
                            batch_contexts,
                            result,
                            target_value_map,
                            image_cache,
                        )
                        for expid in meta_ids
                    ),
                    total=len(meta_ids),
                )
            ]

            for i_batch, i_context in enumerate(batch):
                context = tail_contexts[i_context]
                cols = [f"{context['sign']} {int(c_id)}" for c_id in context["cluster_ids"]]
                contribution_table = pd.DataFrame(index=list(meta_ids), columns=cols)
                contribution_table.index.name = "id"

                # Add results to table
                for expid, batch_values in zip(meta_ids, contributions):
                    contribution_table.loc[expid] = batch_values[i_batch]

                contribution_tables[i_context] = contribution_table.reset_index()

        tails = [sign_to_tail[sign] for sign in signs]
        if not self._is_pairwaise_estimator and len(contribution_tables) == 2:
            # Merge POSTAIL_LBL and NEGTAIL_LBL tables for IBMA
            contribution_table = (
                contribution_tables[0].merge(contribution_tables[1], how="outer").fillna(0)
            )
            tables_dict[contribution_table_name] = contribution_table
        else:
            # Plot separate tables for CBMA
            for tail, contribution_table in zip(tails, contribution_tables):
                tables_dict[f"{contribution_table_name}_tail-{tail}"] = contribution_table

        # Save tables and maps to result
        result.tables.update(tables_dict)
        result.maps.update(maps_dict)

        # Add diagnostics class to result, since more than one can be run
        result.diagnostics.append(self)
        return result


class Jackknife(Diagnostics):
    """Run a jackknife analysis on a meta-analysis result.

    .. versionchanged:: 0.1.2

        * Support for pairwise meta-analyses.

    .. versionchanged:: 0.0.14

        * New parameter: `cluster_threshold`.
        * Return clusters table.

    .. versionchanged:: 0.0.13

        * Change cluster neighborhood from faces+edges to faces, to match Nilearn.

    .. versionadded:: 0.0.11

    Notes
    -----
    This analysis characterizes the relative contribution of each experiment in a meta-analysis
    to the resulting clusters by looping through experiments, calculating the Estimator's summary
    statistic for all experiments *except* the target experiment, dividing the resulting test
    summary statistics by the summary statistics from the original meta-analysis, and finally
    averaging the resulting proportion values across all voxels in each cluster.
    """

    def _leave_one_out_values(self, expid, sign, result, target_value_map, image_cache=None):
        """Refit the Estimator without ``expid`` and return voxelwise proportional reductions.

        Parameters
        ----------
        expid : :obj:`str`
            Study ID to leave out.
        sign : :obj:`str`
            The sign of the label map. Only pairwise Estimators use this, to decide which
            group ``expid`` is dropped from; the refit is otherwise tail-independent.
        result : :obj:`~nimare.results.MetaResult`
            A MetaResult produced by a coordinate- or image-based meta-analysis.
        target_value_map : :obj:`str`
            Name of the map used for per-cluster contribution calculations.
        image_cache : :obj:`dict` or None
            Store shared with the other refits of the same :meth:`transform`, so that the
            input images are masked once rather than once per left-out study. None masks them
            afresh, which is what a single refit outside ``transform`` wants.

        Returns
        -------
        voxelwise_stat_prop_values : 1D :obj:`numpy.ndarray`
            Voxelwise proportional reduction of the statistic after removing ``expid``.
        masker
            The masker the values are expressed in.
        """
        # We need to copy the estimator because it will otherwise overwrite the original version
        # with one missing a study in its inputs.
        estimator = copy.deepcopy(result.estimator)

        # Every refit here masks the same files, so hand each copy the one store shared by
        # this call. It is attached after the copy so that ``deepcopy`` does not duplicate it,
        # and so that the caller's own estimator is left as it was.
        share_cache = getattr(estimator, "share_masked_image_cache", None)
        if share_cache is not None and image_cache is not None:
            share_cache(image_cache)

        if self._is_pairwaise_estimator:
            all_ids = estimator.inputs_["id1"] if sign == POSTAIL_LBL else estimator.inputs_["id2"]
        else:
            all_ids = estimator.inputs_["id"]

        stat_values = result.get_map(target_value_map, return_type="array")

        # Fit Estimator to all studies except the target study
        other_ids = [id_ for id_ in all_ids if id_ != expid]
        if self._is_pairwaise_estimator:
            if sign == POSTAIL_LBL:
                temp_dset = estimator.dataset1.slice(other_ids)
                temp_result = estimator.fit(temp_dset, estimator.dataset2)
            else:
                temp_dset = estimator.dataset2.slice(other_ids)
                temp_result = estimator.fit(estimator.dataset1, temp_dset)
        else:
            temp_dset = estimator.dataset.slice(other_ids)
            temp_result = estimator.fit(temp_dset)

        # Collect the target values (e.g., ALE values) from the N-1 meta-analysis
        temp_stat_vals = temp_result.get_map(target_value_map, return_type="array")

        # Voxelwise proportional reduction of each statistic after removal of the experiment
        with np.errstate(divide="ignore", invalid="ignore"):
            prop_values = np.true_divide(temp_stat_vals, stat_values)
            prop_values = np.nan_to_num(prop_values)

        return 1 - prop_values, estimator.masker

    def _transform_batch(
        self, expid, tail_contexts, result, target_value_map=None, image_cache=None
    ):
        """Apply transform to one study ID for each tail in a batch.

        The leave-one-out refit is the expensive part and does not vary within a batch, so it
        is run once and summarized against every tail's clusters. For a two-tailed IBMA this
        halves the number of refits.
        """
        if any(context["cluster_summary_context"] is None for context in tail_contexts):
            raise ValueError("Jackknife requires a precomputed cluster_summary_context.")

        target_value_map = target_value_map or _get_target_value_map(result)
        voxelwise_stat_prop_values, masker = self._leave_one_out_values(
            expid,
            tail_contexts[0]["sign"],
            result,
            target_value_map,
            image_cache,
        )
        return [
            _summarize_cluster_values(
                voxelwise_stat_prop_values,
                masker,
                context["cluster_summary_context"],
            )
            for context in tail_contexts
        ]

    def _transform(
        self,
        expid,
        label_map,
        sign,
        result,
        target_value_map=None,
        cluster_summary_context=None,
    ):
        """Apply transform to study ID and label map.

        Parameters
        ----------
        expid : :obj:`str`
            Study ID.
        label_map : :class:`nibabel.Nifti1Image`
            The cluster label map image.
        sign : :obj:`str`
            The sign of the label map.
        result : :obj:`~nimare.results.MetaResult`
            A MetaResult produced by a coordinate- or image-based meta-analysis.

        Returns
        -------
        stat_prop_values : 1D :obj:`numpy.ndarray`
            1D array with the contribution of `expid` in each cluster of `label_map`.
        """
        context = {
            "sign": sign,
            "label_map": label_map,
            "cluster_summary_context": cluster_summary_context,
        }
        return self._transform_batch(expid, [context], result, target_value_map)[0]


class FocusCounter(Diagnostics):
    """Run a focus-count analysis on a coordinate-based meta-analysis result.

    .. versionchanged:: 0.1.2

        * Support for pairwise meta-analyses.

    .. versionchanged:: 0.0.14

        * New parameter: `cluster_threshold`.
        * Return clusters table.

    .. versionchanged:: 0.0.13

        Change cluster neighborhood from faces+edges to faces, to match Nilearn.

    .. versionadded:: 0.0.12

    Notes
    -----
    This analysis characterizes the relative contribution of each experiment in a meta-analysis
    to the resulting clusters by counting the number of peaks from each experiment that fall within
    each significant cluster.

    Warnings
    --------
    This method only works for coordinate-based meta-analyses.
    """

    def _transform(
        self,
        expid,
        label_map,
        sign,
        result,
        target_value_map=None,
        cluster_summary_context=None,
    ):
        """Apply transform to study ID and label map.

        Parameters
        ----------
        expid : :obj:`str`
            Study ID.
        label_map : :class:`nibabel.Nifti1Image`
            The cluster label map image.
        sign : :obj:`str`
            The sign of the label map.
        result : :obj:`~nimare.results.MetaResult`
            A MetaResult produced by a coordinate- or image-based meta-analysis.

        Returns
        -------
        stat_prop_values : 1D :obj:`numpy.ndarray`
            1D array with the contribution of `expid` in each cluster of `label_map`.
        """
        if issubclass(type(result.estimator), IBMAEstimator):
            raise ValueError("This method only works for coordinate-based meta-analyses.")

        affine = label_map.affine
        label_arr = np.asanyarray(label_map.dataobj)
        clust_ids = (
            _cluster_ids(label_arr)
            if cluster_summary_context is None
            else cluster_summary_context["cluster_ids"]
        )

        if self._is_pairwaise_estimator:
            coordinates_df = (
                result.estimator.inputs_["coordinates1"]
                if sign == POSTAIL_LBL
                else result.estimator.inputs_["coordinates2"]
            )
        else:
            coordinates_df = result.estimator.inputs_["coordinates"]

        coords = coordinates_df.loc[coordinates_df["id"] == expid]
        ijk = mm2vox(coords[["x", "y", "z"]], affine)

        return _count_foci_per_cluster(label_arr, clust_ids, ijk)


class ResampledStability(NiMAREBase):
    """Estimate voxelwise stability of thresholded results under dataset resampling.

    Determine the stability of a meta-analytic result by applying a resampling policy to the
    input dataset and then characterizing the stability of the resulting
    meta-analytic map's voxelwise and/or clusterwise significance.
    Based on the implementation in :footcite:t:`Frahm_Monimu_Hoffstaedter`.

    Parameters
    ----------
    target_image : 'str', optional
        The meta-analytic map for which stability will be characterized.
    resampling_policy : {"leave_1_out", "leave_k_out", "subsample"}, optional
        The resampling policy to use.
    k : int, optional
        The number of studies to leave out for each replicate when
        ``resampling_policy="leave_k_out"``.
        Must be between 1 and n-1, where n is the number of studies in the meta-analysis.
    target_n : int, optional
        The number of studies to include in each replicate when ``resampling_policy="subsample"``.
        Must be between 1 and n, where n is the number of studies in the meta-analysis.
        Default is n (i.e., subsamples are the same size as the original dataset).
    n_resamples : int, optional
        The number of resampled replicates to generate.
        If None, all possible unique replicates will be generated,
        up to a maximum of 1000 (to avoid combinatorial explosion).
    random_state : int or None, optional
        Random seed for reproducibility when random sampling is used in the resampling policy.
        Default is None.
    voxel_thresh : float or None, optional
        An optional voxel-level threshold that may be applied to the ``target_image``
        to define clusters.
        This can be None if the ``target_image`` is already thresholded
        (e.g., a cluster-level corrected map). Default is None.

        .. note::
            Unlike :class:`Diagnostics`, this class has not been renamed to
            ``target_threshold``, because the value is also reused as the cluster-forming
            threshold when the ``"subsample"`` policy re-runs Monte Carlo FWE correction,
            where it is read as a p-value. The two uses want different units and should be
            split before either is renamed.
    cluster_threshold : int or None, optional
        Cluster size threshold, in voxels.
        If None, then no cluster size threshold will be applied.
        Default is None.
    mask_coverage : {"gm", "brain"}, optional
        Voxel set used as the randomisation prior for the ``"subsample"``
        policy.  ``"gm"`` restricts random foci to grey-matter voxels (mask
        image intensity > 0.1); ``"brain"`` uses all non-zero voxels.
        Default is ``"gm"``.
    alpha : float, optional
        Family-wise error rate for the Monte Carlo cluster-size threshold used
        in the ``"subsample"`` policy.  The ``(1 - alpha)`` percentile of the
        permutation null distribution is applied. Default is 0.05.
    n_cores : int, optional
        Number of cores to use for parallelization.
        If <=0, defaults to using all available cores. Default is 1.
    generate_description : bool, optional
        Whether to append boilerplate text and extract references for the returned result.
        Default is True.
    """

    def __init__(
        self,
        target_image="z_desc-size_level-cluster_corr-FWE_method-montecarlo",
        resampling_policy="subsample",
        k=None,
        target_n=None,
        n_resamples=None,
        random_state=None,
        voxel_thresh=None,
        cluster_threshold=None,
        mask_coverage="gm",
        alpha=0.05,
        n_cores=1,
        generate_description=True,
    ):
        if mask_coverage not in ("gm", "brain"):
            raise ValueError("mask_coverage must be 'gm' or 'brain'.")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1; got {alpha}.")
        self.target_image = target_image
        self.resampling_policy = resampling_policy
        self.k = k
        self.target_n = target_n
        self.n_resamples = n_resamples
        self.random_state = random_state
        self.voxel_thresh = voxel_thresh
        self.cluster_threshold = cluster_threshold
        self.mask_coverage = mask_coverage
        self.alpha = alpha
        self.n_cores = _check_ncores(n_cores)
        self.generate_description = generate_description

    def _resolve_subsets(self, n_studies):
        """Build a replicate schedule in study-index space."""
        target_n = resolve_subset_size(
            self.resampling_policy,
            n_studies,
            k=self.k,
            target_n=self.target_n,
        )
        subsets = generate_subset_schedule(
            n_studies,
            target_n,
            n_samples=self.n_resamples,
            random_state=self.random_state,
        )
        return subsets, target_n

    def _extract_binary_support(self, result):
        """Convert the selected target image into a binary support vector."""
        if self.target_image not in result.maps:
            if result.corrector is None:
                raise ValueError(
                    f"Target image '{self.target_image}' is not present in replicate results."
                )
            result = result.corrector.transform(result)
            if self.target_image not in result.maps:
                raise ValueError(
                    f"Target image '{self.target_image}' is not present even after "
                    "reapplying the original corrector."
                )

        if self.voxel_thresh is None and self.cluster_threshold is None:
            values = result.get_map(self.target_image, return_type="array")
            return (np.abs(values) > 0).astype(DEFAULT_FLOAT_DTYPE, copy=False)

        target_img = result.get_map(self.target_image, return_type="image")
        data = target_img.get_fdata(dtype=DEFAULT_FLOAT_DTYPE)
        two_sided = getattr(result.estimator, "two_sided", bool((data < 0).any()))
        stat_threshold = self.voxel_thresh or 0
        cluster_threshold = 0 if self.cluster_threshold is None else self.cluster_threshold
        _, label_maps = get_clusters_table(
            target_img,
            stat_threshold,
            cluster_threshold,
            two_sided=two_sided,
            return_label_maps=True,
        )
        if not label_maps:
            return np.zeros(
                result.masker.transform(target_img).shape[-1], dtype=DEFAULT_FLOAT_DTYPE
            )

        support = np.zeros_like(data, dtype=bool)
        for label_map in label_maps:
            support |= np.asanyarray(label_map.dataobj) > 0
        return np.squeeze(
            result.masker.transform(nib.Nifti1Image(support.astype(np.int8), target_img.affine))
        ).astype(DEFAULT_FLOAT_DTYPE, copy=False)

    def _fit_replicate(self, kept_ids, result):
        """Refit the estimator on one retained-id subset and return binary support."""
        estimator = copy.deepcopy(result.estimator)
        dataset = estimator.dataset.slice(list(kept_ids))
        replicate_result = estimator.fit(dataset)
        if result.corrector is not None and self.target_image not in replicate_result.maps:
            replicate_result = result.corrector.transform(replicate_result)
        return self._extract_binary_support(replicate_result)

    def _fit_cbma_subset_replicate(
        self,
        subset_idx,
        ma_maps,
        estimator,
        study_ids,
        cluster_threshold,
        precomputed_null=None,
        mask_arr=None,
    ):
        """Compute one CBMA replicate from cached MA maps for a retained-study subset."""
        subset_ma = ma_maps[subset_idx, :]
        subset_study_ids = study_ids[subset_idx]
        _, z_values = _approximate_z_from_ma(
            estimator, subset_ma, subset_study_ids, precomputed_null=precomputed_null
        )
        z_values, _ = _threshold_z_clusters(
            z_values,
            estimator.masker,
            voxel_thresh=self.voxel_thresh or 0.001,
            cluster_size_threshold=cluster_threshold,
            mask_arr=mask_arr,
        )
        return (z_values > 0).astype(DEFAULT_FLOAT_DTYPE, copy=False)

    def _cbma_subset_stability(self, result, subsets, target_n):
        """Run cached-MA stability analysis for any single-sample CBMA estimator."""
        # Deep copy protects result.estimator from _collect_ma_maps side-effects
        # (ALE sets _study_max_ma_values during map collection).
        estimator = copy.deepcopy(result.estimator)
        ma_maps = estimator._collect_ma_maps()
        montecarlo_iters = (
            result.corrector.parameters.get("n_iters", 5000)
            if result.corrector is not None
            else 5000
        )
        cluster_forming_threshold = self.voxel_thresh or 0.001
        study_ids = np.array(estimator.inputs_["id"])
        sample_space = _mask_coverage_to_null_ijk(
            estimator.masker, mask_coverage=self.mask_coverage
        ).astype(np.int32, copy=False)

        # Build the full-dataset approximate null once and reuse it for every
        # subsample and null-MA iteration (mirrors JALE's hx_conv reuse).
        full_null_temp = copy.deepcopy(estimator)
        full_null_temp.null_distributions_ = {}
        full_null_temp._prepare_subsample_null(ma_maps)
        full_null_temp._compute_approximate_z_values(ma_maps)
        precomputed_null = full_null_temp.null_distributions_

        # Precompute boolean mask array once to avoid NiBabel round-trip in hot loops.
        mask_arr = _mask_img_to_bool(estimator.masker.mask_img)

        rng = np.random.RandomState(self.random_state)
        null_cluster_sizes = np.zeros(montecarlo_iters, dtype=np.int32)
        for i_iter in range(montecarlo_iters):
            null_ma, subset_ids = estimator._generate_random_null_ma(target_n, sample_space, rng)
            _, null_z = _approximate_z_from_ma(
                estimator, null_ma, subset_ids, precomputed_null=precomputed_null
            )
            _, null_cluster_sizes[i_iter] = _threshold_z_clusters(
                null_z,
                estimator.masker,
                voxel_thresh=cluster_forming_threshold,
                cluster_size_threshold=None,
                mask_arr=mask_arr,
            )

        cluster_threshold = np.percentile(null_cluster_sizes, 100.0 * (1.0 - self.alpha))

        running_sum = None
        n_done = 0
        for support in tqdm(
            Parallel(return_as="generator", n_jobs=self.n_cores)(
                delayed(self._fit_cbma_subset_replicate)(
                    subset_idx,
                    ma_maps,
                    estimator,
                    study_ids,
                    cluster_threshold,
                    precomputed_null=precomputed_null,
                    mask_arr=mask_arr,
                )
                for subset_idx in subsets
            ),
            total=len(subsets),
        ):
            running_sum = (
                support.astype(np.float64) if running_sum is None else running_sum + support
            )
            n_done += 1

        return (running_sum / n_done).astype(DEFAULT_FLOAT_DTYPE, copy=False)

    def _finalize_result(self, result, stability_map, n_resamples_used, target_n_used):
        """Attach stability map and summary table to a copied result object."""
        result = self._copy_result_for_diagnostic(result)
        map_name = f"{self.target_image}_diag-ResampledStability"
        result.maps[map_name] = stability_map
        result.tables[f"{map_name}_tab-summary"] = pd.DataFrame(
            [
                {
                    "target_image": self.target_image,
                    "resampling_policy": self.resampling_policy,
                    "n_resamples": n_resamples_used,
                    "target_n": target_n_used,
                    "k": self.k,
                    "random_state": self.random_state,
                }
            ]
        )
        result.diagnostics.append(self)
        if self.generate_description:
            result.description_ += (
                " Voxelwise stability of thresholded results was estimated by repeatedly "
                "resampling the input dataset, recomputing thresholded support maps, and "
                "averaging the binary support across resamples. This diagnostic follows the "
                "resampling-based stability approach implemented "
                "in JALE \\citep{Frahm_Monimu_Hoffstaedter}."
            )
        return result

    @staticmethod
    def _copy_result_for_diagnostic(result):
        """Return a lightweight MetaResult copy suitable for adding diagnostic outputs."""
        new = object.__new__(MetaResult)
        new.estimator = result.estimator
        new.corrector = result.corrector
        new.diagnostics = list(result.diagnostics)
        new.masker = result.masker
        new.maps = dict(result.maps)
        new.tables = dict(result.tables)
        new.metadata = dict(getattr(result, "metadata", {}))
        new._set_description(result.description_)
        return new

    def transform(self, result):
        """Apply the resampling diagnostic to a fitted meta-analytic result."""
        if issubclass(type(result.estimator), PairwiseCBMAEstimator):
            raise ValueError(
                "ResampledStability currently supports single-sample estimators only."
            )
        if not isinstance(result.estimator, (CBMAEstimator, IBMAEstimator)):
            raise ValueError(
                "ResampledStability only supports CBMA and single-sample IBMA estimators."
            )

        if not hasattr(result.estimator, "dataset") or result.estimator.dataset is None:
            raise ValueError(
                "ResampledStability requires a fitted estimator with a retained dataset."
            )

        all_ids = list(result.estimator.inputs_["id"])
        subsets, target_n_used = self._resolve_subsets(len(all_ids))

        if isinstance(result.estimator, CBMAEstimator):
            stability_map = self._cbma_subset_stability(result, subsets, target_n_used)
            return self._finalize_result(
                result,
                stability_map,
                n_resamples_used=len(subsets),
                target_n_used=target_n_used,
            )

        kept_id_lists = [[all_ids[i] for i in subset] for subset in subsets]

        running_sum = None
        n_done = 0
        for support in tqdm(
            Parallel(return_as="generator", n_jobs=self.n_cores)(
                delayed(self._fit_replicate)(kept_ids, result) for kept_ids in kept_id_lists
            ),
            total=len(kept_id_lists),
        ):
            running_sum = (
                support.astype(np.float64) if running_sum is None else running_sum + support
            )
            n_done += 1

        stability_map = (running_sum / n_done).astype(DEFAULT_FLOAT_DTYPE, copy=False)
        return self._finalize_result(
            result,
            stability_map,
            n_resamples_used=len(kept_id_lists),
            target_n_used=target_n_used,
        )


class FocusFilter(NiMAREBase):
    """Remove coordinates outside of the collection mask.

    .. versionadded:: 0.0.13

    Parameters
    ----------
    mask : :obj:`str`, :class:`~nibabel.nifti1.Nifti1Image`, \
    :class:`~nilearn.maskers.NiftiMasker` or similar, or None, optional
        Mask(er) to use. If None, uses the masker of the collection provided in ``transform``.

    Notes
    -----
    This filter removes any coordinates outside of the brain mask.
    It does not remove studies without coordinates in the brain mask, since an input collection
    does not need to have coordinates for all studies (e.g., some may only have images).
    """

    def __init__(self, mask=None):
        if mask is not None:
            mask = get_masker(mask)

        self.masker = mask

    def transform(self, dataset):
        """Apply the filter to a Studyset/Dataset collection.

        Parameters
        ----------
        dataset : :obj:`~nimare.nimads.Studyset`, \
                or :obj:`~nimare.dataset.Dataset`
            The collection to filter.

        Returns
        -------
        dataset : :obj:`~nimare.dataset.Dataset` or :obj:`~nimare.nimads.Studyset`
            The filtered collection.

        .. warning::
            Support for :class:`~nimare.dataset.Dataset` inputs is deprecated and will be removed
            in NiMARE 1.0.0. Prefer :class:`~nimare.nimads.Studyset`.
        """
        filtered = normalize_collection(dataset)

        masker = self.masker or filtered.masker
        mask_array = np.asarray(masker.mask_img_.dataobj)
        affine = masker.mask_img.affine

        # Coordinates in the studyset's own space, at store level: this selects
        # *foci*, and every analysis is kept, so the selection is a point mask
        # rather than a rewritten table.
        xyz, _, _ = harmonized_coordinates(filtered.store, filtered.space)
        ijk = mm2vox(xyz, affine)
        shape = np.asarray(mask_array.shape)
        in_bounds = np.all((ijk >= 0) & (ijk < shape), axis=1)
        keep = np.zeros(len(ijk), dtype=bool)
        if in_bounds.any():
            inside = ijk[in_bounds]
            keep[in_bounds] = mask_array[inside[:, 0], inside[:, 1], inside[:, 2]] == 1

        LGR.info(
            f"{int((~keep).sum())}/{len(keep)} coordinates fall outside of the mask. "
            "Removing them."
        )
        return filtered.select_points(keep)
