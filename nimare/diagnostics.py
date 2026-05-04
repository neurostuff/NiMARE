"""Methods for diagnosing problems in meta-analytic datasets or analyses."""

import copy
import logging
from abc import abstractmethod
from itertools import combinations

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn.maskers import NiftiLabelsMasker
from nilearn.reporting import get_clusters_table
from scipy.spatial.distance import cdist
from scipy.special import comb
from tqdm.auto import tqdm

from nimare.base import NiMAREBase
from nimare.dataset import Dataset
from nimare.meta.cbma.base import CBMAEstimator, PairwiseCBMAEstimator
from nimare.meta.ibma import IBMAEstimator
from nimare.nimads import Studyset
from nimare.studyset import normalize_collection
from nimare.utils import (
    DEFAULT_FLOAT_DTYPE,
    _check_ncores,
    _filter_kwargs,
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


def _build_cluster_summary_context(masker, label_map, label_vector, cluster_ids):
    """Precompute cluster summaries in array space when possible."""
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
                "cluster_indices": cluster_indices,
            }

    cluster_masker = NiftiLabelsMasker(label_map, **_cluster_masker_kwargs())
    cluster_masker.fit(label_map)
    return {
        "mode": "image",
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
        An optional voxel-level threshold that may be applied to the ``target_image`` to define
        clusters. This can be None if the ``target_image`` is already thresholded
        (e.g., a cluster-level corrected map).
        Default is None.
    cluster_threshold : :obj:`int` or None, optional
        Cluster size threshold, in :term:`voxels<voxel>`.
        If None, then no cluster size threshold will be applied. Default=None.
    n_cores : :obj:`int`, optional
        Number of cores to use for parallelization.
        If <=0, defaults to using all available cores.
        Default is 1.

    """

    def __init__(
        self,
        target_image="z_desc-size_level-cluster_corr-FWE_method-montecarlo",
        voxel_thresh=None,
        cluster_threshold=None,
        display_second_group=False,
        n_cores=1,
    ):
        self.target_image = target_image
        self.voxel_thresh = voxel_thresh
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
        stat_threshold = self.voxel_thresh or 0
        cluster_threshold = 0 if self.cluster_threshold is None else self.cluster_threshold

        if hasattr(result.estimator, "two_sided"):
            # Only present in Fisher's and Stouffer's estimators
            two_sided = getattr(result.estimator, "two_sided")
        else:
            two_sided = (target_img.get_fdata(dtype=DEFAULT_FLOAT_DTYPE) < 0).any()

        clusters_table, label_maps = get_clusters_table(
            target_img,
            stat_threshold,
            cluster_threshold,
            two_sided=two_sided,
            return_label_maps=True,
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

        contribution_tables = []
        for sign, label_map, label_map_name, meta_ids in zip(
            signs, label_maps, label_map_names, meta_ids_lst
        ):
            label_arr = np.asanyarray(label_map.dataobj)
            cluster_ids = sorted(list(np.unique(label_arr)[1:]))
            rows = list(meta_ids)
            cluster_summary_context = _build_cluster_summary_context(
                masker,
                label_map,
                maps_dict[label_map_name],
                cluster_ids,
            )

            # Create contribution table
            cols = [f"{sign} {int(c_id)}" for c_id in cluster_ids]
            contribution_table = pd.DataFrame(index=rows, columns=cols)
            contribution_table.index.name = "id"

            contributions = [
                r
                for r in tqdm(
                    Parallel(return_as="generator", n_jobs=self.n_cores)(
                        delayed(self._transform)(
                            expid,
                            label_map,
                            sign,
                            result,
                            target_value_map,
                            cluster_summary_context,
                        )
                        for expid in meta_ids
                    ),
                    total=len(meta_ids),
                )
            ]

            # Add results to table
            for expid, stat_prop_values in zip(meta_ids, contributions):
                contribution_table.loc[expid] = stat_prop_values

            contribution_tables.append(contribution_table.reset_index())

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
        # We need to copy the estimator because it will otherwise overwrite the original version
        # with one missing a study in its inputs.
        estimator = copy.deepcopy(result.estimator)

        if self._is_pairwaise_estimator:
            all_ids = estimator.inputs_["id1"] if sign == POSTAIL_LBL else estimator.inputs_["id2"]
        else:
            all_ids = estimator.inputs_["id"]

        target_value_map = target_value_map or _get_target_value_map(result)
        stat_values = result.get_map(target_value_map, return_type="array")
        if cluster_summary_context is None:
            raise ValueError("Jackknife requires a precomputed cluster_summary_context.")

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

        voxelwise_stat_prop_values = 1 - prop_values
        return _summarize_cluster_values(
            voxelwise_stat_prop_values,
            estimator.masker,
            cluster_summary_context,
        )


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
        clust_ids = sorted(list(np.unique(label_arr)[1:]))

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

        focus_counts = []
        for c_val in clust_ids:
            cluster_mask = label_arr == c_val
            cluster_idx = np.vstack(np.where(cluster_mask))
            distances = cdist(cluster_idx.T, ijk)
            distances = distances < 1
            distances = np.any(distances, axis=0)
            n_included_voxels = np.sum(distances)
            focus_counts.append(n_included_voxels)

        return np.array(focus_counts)


class ResampledStability(NiMAREBase):
    """Estimate voxelwise stability of thresholded results under dataset resampling.

    Notes
    -----
    For ALE analyses with ``resampling_policy="subsample"``, the subsample
    path follows the same cluster-thresholding formulation used by related ALE
    software. See that project's README for background and references.

    Parameters
    ----------
    prior_space : {"gm", "brain"}, optional
        Voxel set used as the randomization prior when ``resampling_policy``
        is ``"subsample"`` and the estimator is ALE. ``"gm"`` restricts random
        foci to grey-matter voxels (mask image intensity > 0.1); ``"brain"``
        uses all non-zero voxels. Default is ``"gm"``.
    alpha : float, optional
        Family-wise error rate used to determine the cluster-size threshold for
        the ALE subsample path. The ``(1 - alpha)`` percentile of
        the permutation null cluster-size distribution is used. Default is
        0.05.
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
        prior_space="gm",
        alpha=0.05,
        n_cores=1,
    ):
        if prior_space not in ("gm", "brain"):
            raise ValueError("prior_space must be 'gm' or 'brain'.")
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
        self.prior_space = prior_space
        self.alpha = alpha
        self.n_cores = _check_ncores(n_cores)

    def _resolve_subsets(self, n_studies):
        """Build a replicate schedule in study-index space."""
        policy = self.resampling_policy
        rng = np.random.RandomState(self.random_state)

        if policy == "leave_1_out":
            return [np.delete(np.arange(n_studies), i) for i in range(n_studies)]

        if policy == "leave_k_out":
            if self.k is None:
                raise ValueError("resampling_policy='leave_k_out' requires k.")
            if not 0 < self.k < n_studies:
                raise ValueError(f"k must be between 1 and {n_studies - 1}; got {self.k}.")
            target_n = n_studies - int(self.k)
        elif policy == "subsample":
            target_n = self.target_n or n_studies
            if not 0 < target_n <= n_studies:
                raise ValueError(f"target_n must be between 1 and {n_studies}; got {target_n}.")
        else:
            raise ValueError(
                "resampling_policy must be one of 'leave_1_out', 'leave_k_out', or 'subsample'."
            )

        max_combinations = int(comb(n_studies, target_n, exact=True))
        if self.n_resamples is None:
            if max_combinations > 1000:
                raise ValueError(
                    "This resampling schedule is too large to enumerate exhaustively. "
                    "Set n_resamples to sample a subset of unique schedules."
                )
            n_resamples = max_combinations
        else:
            n_resamples = min(int(self.n_resamples), max_combinations)

        if n_resamples == max_combinations and max_combinations <= 1000:
            return [
                np.asarray(idx, dtype=np.int32) for idx in combinations(range(n_studies), target_n)
            ]

        subsets = set()
        while len(subsets) < n_resamples:
            subset = tuple(np.sort(rng.choice(n_studies, size=target_n, replace=False)))
            subsets.add(subset)
        return [np.asarray(subset, dtype=np.int32) for subset in sorted(subsets)]

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

    def _fit_ale_subsample_replicate(self, sample_idx, ma_maps, estimator, cluster_threshold):
        """Compute one probabilistic ALE replicate from cached MA maps."""
        from nimare.meta.cbma.ale import (
            _ale_approximate_z_from_ma,
            _threshold_z_clusters,
        )

        subset_ma = ma_maps[sample_idx, :]
        _, z_values = _ale_approximate_z_from_ma(estimator, subset_ma)
        z_values, _ = _threshold_z_clusters(
            z_values,
            estimator.masker,
            voxel_thresh=self.voxel_thresh or 0.001,
            cluster_size_threshold=cluster_threshold,
        )
        return (z_values > 0).astype(DEFAULT_FLOAT_DTYPE, copy=False)

    def _ale_subsample_stability(self, result):
        """Run probabilistic ALE stability analysis for ALE estimators."""
        from nimare.meta.cbma.ale import (
            _ale_approximate_z_from_ma,
            _generate_unique_subsamples,
            _masked_prior_columns,
            _random_ale_ma_from_metadata,
            _study_metadata_from_coordinates,
            _threshold_z_clusters,
        )

        estimator = copy.deepcopy(result.estimator)
        ma_maps = estimator._collect_ma_maps()
        n_studies = ma_maps.shape[0]
        target_n = self.target_n or n_studies
        n_resamples = self.n_resamples or 2500
        montecarlo_iters = (
            result.corrector.parameters.get("n_iters", 5000)
            if result.corrector is not None
            else 5000
        )
        cluster_forming_threshold = self.voxel_thresh or 0.001
        sample_sizes, num_foci = _study_metadata_from_coordinates(estimator.inputs_["coordinates"])
        prior_img, _ = _masked_prior_columns(estimator.masker, prior_space=self.prior_space)
        sample_space = np.vstack(np.where(prior_img)).T.astype(np.int32, copy=False)

        rng = np.random.RandomState(self.random_state)
        null_cluster_sizes = np.zeros(montecarlo_iters, dtype=np.int32)
        for i_iter in range(montecarlo_iters):
            if target_n < n_studies:
                subset = rng.permutation(n_studies)[:target_n]
                subset_sample_sizes = sample_sizes[subset]
                subset_num_foci = num_foci[subset]
            else:
                subset_sample_sizes = sample_sizes
                subset_num_foci = num_foci
            null_ma = _random_ale_ma_from_metadata(
                estimator.masker.mask_img,
                subset_sample_sizes,
                subset_num_foci,
                sample_space,
                rng,
            )
            _, null_z = _ale_approximate_z_from_ma(estimator, null_ma)
            _, null_cluster_sizes[i_iter] = _threshold_z_clusters(
                null_z,
                estimator.masker,
                voxel_thresh=cluster_forming_threshold,
                cluster_size_threshold=None,
            )

        cluster_threshold = np.percentile(null_cluster_sizes, 100.0 * (1.0 - self.alpha))
        samples = _generate_unique_subsamples(
            n_studies,
            target_n,
            n_resamples,
            random_state=self.random_state,
        )
        replicate_maps = [
            values
            for values in tqdm(
                Parallel(return_as="generator", n_jobs=self.n_cores)(
                    delayed(self._fit_ale_subsample_replicate)(
                        sample_idx,
                        ma_maps,
                        estimator,
                        cluster_threshold,
                    )
                    for sample_idx in samples
                ),
                total=len(samples),
            )
        ]
        return np.mean(np.vstack(replicate_maps), axis=0).astype(DEFAULT_FLOAT_DTYPE, copy=False)

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

        if result.estimator.__class__.__name__ == "ALE" and self.resampling_policy == "subsample":
            stability_map = self._ale_subsample_stability(result)
            result = result.copy()
            map_name = f"{self.target_image}_diag-ResampledStability"
            table_name = f"{map_name}_tab-summary"
            result.maps[map_name] = stability_map
            result.tables[table_name] = pd.DataFrame(
                [
                    {
                        "target_image": self.target_image,
                        "resampling_policy": self.resampling_policy,
                        "n_resamples": int(self.n_resamples or 2500),
                        "target_n": self.target_n,
                        "k": self.k,
                        "random_state": self.random_state,
                    }
                ]
            )
            result.diagnostics.append(self)
            return result

        all_ids = list(result.estimator.inputs_["id"])
        subsets = self._resolve_subsets(len(all_ids))
        kept_id_lists = [[all_ids[i] for i in subset] for subset in subsets]

        replicate_maps = [
            values
            for values in tqdm(
                Parallel(return_as="generator", n_jobs=self.n_cores)(
                    delayed(self._fit_replicate)(kept_ids, result) for kept_ids in kept_id_lists
                ),
                total=len(kept_id_lists),
            )
        ]
        stability_map = np.mean(np.vstack(replicate_maps), axis=0).astype(
            DEFAULT_FLOAT_DTYPE,
            copy=False,
        )

        result = result.copy()
        map_name = f"{self.target_image}_diag-ResampledStability"
        table_name = f"{map_name}_tab-summary"
        result.maps[map_name] = stability_map
        result.tables[table_name] = pd.DataFrame(
            [
                {
                    "target_image": self.target_image,
                    "resampling_policy": self.resampling_policy,
                    "n_resamples": len(kept_id_lists),
                    "target_n": None if not kept_id_lists else len(kept_id_lists[0]),
                    "k": self.k,
                    "random_state": self.random_state,
                }
            ]
        )
        result.diagnostics.append(self)
        return result


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
            in a future release. Prefer :class:`~nimare.nimads.Studyset`.
        """
        if isinstance(dataset, Dataset):
            filtered = dataset
        elif isinstance(dataset, Studyset):
            filtered = dataset.copy()
        else:
            filtered = normalize_collection(dataset).copy()

        masker = self.masker or filtered.masker
        # use 0 or 1 to indicate if voxels are in the mask
        masker_array = masker.mask_img_.dataobj

        # Get matrix indices for collection coordinates
        dset_xyz = filtered.coordinates[["x", "y", "z"]].values

        # mm2vox automatically rounds the coordinates
        dset_ijk = mm2vox(dset_xyz, masker.mask_img.affine)

        # Only retain coordinates inside the brain mask
        def check_coord(coord):
            try:
                return masker_array[coord[0], coord[1], coord[2]] == 1
            except IndexError:
                return False

        keep_idx = [i for i, coord in enumerate(dset_ijk) if check_coord(coord)]

        LGR.info(
            f"{dset_ijk.shape[0] - len(keep_idx)}/{dset_ijk.shape[0]} coordinates fall outside of "
            "the mask. Removing them."
        )

        # Only retain coordinates inside the brain mask
        filtered.coordinates = filtered.coordinates.iloc[keep_idx]

        return filtered
