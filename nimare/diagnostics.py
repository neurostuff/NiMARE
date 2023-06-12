"""Methods for diagnosing problems in meta-analytic datasets or analyses."""
import copy
import logging
from abc import abstractmethod

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn import input_data
from nilearn.reporting import get_clusters_table
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

from nimare.base import NiMAREBase
from nimare.meta.cbma.base import PairwiseCBMAEstimator
from nimare.meta.ibma import IBMAEstimator
from nimare.utils import _check_ncores, get_masker, mm2vox, tqdm_joblib

LGR = logging.getLogger(__name__)


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
    def _transform(self, expid, label_map, result):
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
        two_sided = (target_img.get_fdata() < 0).any()
        clusters_table, label_maps = get_clusters_table(
            target_img,
            stat_threshold,
            self.cluster_threshold,
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
                f"PositiveTail {row['Cluster ID']}"
                if row["Peak Stat"] > 0
                else f"NegativeTail {row['Cluster ID']}"
                for _, row in clusters_table.iterrows()
            ]

        # Define bids-like names for tables and maps
        image_name = "_".join(self.target_image.split("_")[1:])
        image_name = f"_{image_name}" if image_name else image_name
        clusters_table_name = f"{self.target_image}_tab-clust"
        contribution_table_name = f"{self.target_image}_diag-{diag_name}_tab-counts"
        label_map_names = (
            [f"label{image_name}_tail-positive", f"label{image_name}_tail-negative"]
            if len(label_maps) == 2
            else [f"label{image_name}_tail-positive"]
        )

        # Check number of clusters
        if n_clusters == 0:
            result.tables[clusters_table_name] = clusters_table
            result.tables[contribution_table_name] = None
            result.maps[label_map_names[0]] = None

            result.diagnostics.append(self)
            return result

        # Use study IDs in inputs_ instead of dataset, because we don't want to try fitting the
        # estimator to a study that might have been filtered out by the estimator's criteria.
        # Use only id1 for pairwise estimators.
        if self._is_pairwaise_estimator:
            if self.display_second_group and len(label_maps) == 2:
                # Run diagnostics with id2 for pairwise estimators and display_second_group
                meta_ids_lst = [result.estimator.inputs_["id1"], result.estimator.inputs_["id2"]]
                signs = ["PositiveTail", "NegativeTail"]
            else:
                meta_ids_lst = [result.estimator.inputs_["id1"]]
                signs = ["PositiveTail"]
        elif len(label_maps) == 2:
            meta_ids_lst = [result.estimator.inputs_["id"], result.estimator.inputs_["id"]]
            signs = ["PositiveTail", "NegativeTail"]
        else:
            meta_ids_lst = [result.estimator.inputs_["id"]]
            signs = ["PositiveTail"]

        contribution_tables = []
        for sign, label_map, meta_ids in zip(signs, label_maps, meta_ids_lst):
            rows = list(meta_ids)
            cluster_ids = sorted(list(np.unique(label_map.get_fdata())[1:]))

            # Create contribution table
            cols = [f"{sign} {int(c_id)}" for c_id in cluster_ids]
            contribution_table = pd.DataFrame(index=rows, columns=cols)
            contribution_table.index.name = "id"

            with tqdm_joblib(tqdm(total=len(meta_ids))):
                contributions = Parallel(n_jobs=self.n_cores)(
                    delayed(self._transform)(expid, label_map, sign, result) for expid in meta_ids
                )

            # Add results to table
            for expid, stat_prop_values in zip(meta_ids, contributions):
                contribution_table.loc[expid] = stat_prop_values

            contribution_tables.append(contribution_table.reset_index())

        if len(contribution_tables) == 2:
            # Merge PositiveTail and NegativeTail tables
            contribution_table = (
                contribution_tables[0].merge(contribution_tables[1], how="outer").fillna(0)
            )
        else:
            # Only export PositiveTail table for pairwise estimators
            contribution_table = contribution_tables[0]

        # Save tables and maps to result
        diag_tables_dict = {
            clusters_table_name: clusters_table,
            contribution_table_name: contribution_table,
        }
        diag_maps_dict = {
            label_map_name: np.squeeze(masker.transform(label_map))
            for label_map_name, label_map in zip(label_map_names, label_maps)
        }

        result.tables.update(diag_tables_dict)
        result.maps.update(diag_maps_dict)

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

    def _transform(self, expid, label_map, sign, result):
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
            all_ids = estimator.inputs_["id1"]
        else:
            all_ids = estimator.inputs_["id"]

        original_masker = estimator.masker

        # Mask using a labels masker, so that we can easily get the mean value for each cluster
        cluster_masker = input_data.NiftiLabelsMasker(label_map)
        cluster_masker.fit(label_map)

        # CBMAs have "stat" maps, while most IBMAs have "est" maps.
        # Fisher's and Stouffer's only have "z" maps though.
        if "est" in result.maps:
            target_value_map = "est"
        elif "stat" in result.maps:
            target_value_map = "stat"
        elif "stat_desc-group1MinusGroup2" in result.maps:
            target_value_map = "stat_desc-group1MinusGroup2"
        elif "z_desc-specificity" in result.maps:
            target_value_map = "z_desc-specificity"
        else:
            target_value_map = "z"

        stat_values = result.get_map(target_value_map, return_type="array")

        # Fit Estimator to all studies except the target study
        other_ids = [id_ for id_ in all_ids if id_ != expid]
        if self._is_pairwaise_estimator:
            temp_dset = (
                estimator.dataset1.slice(other_ids)
                if sign == "PositiveTail"
                else estimator.dataset2.slice(other_ids)
            )
            temp_result = estimator.fit(temp_dset, estimator.dataset2)
        else:
            temp_dset = estimator.dataset.slice(other_ids)
            temp_result = estimator.fit(temp_dset)

        # Collect the target values (e.g., ALE values) from the N-1 meta-analysis
        temp_stat_img = temp_result.get_map(target_value_map, return_type="image")
        temp_stat_vals = np.squeeze(original_masker.transform(temp_stat_img))

        # Voxelwise proportional reduction of each statistic after removal of the experiment
        with np.errstate(divide="ignore", invalid="ignore"):
            prop_values = np.true_divide(temp_stat_vals, stat_values)
            prop_values = np.nan_to_num(prop_values)

        voxelwise_stat_prop_values = 1 - prop_values
        stat_prop_img = original_masker.inverse_transform(voxelwise_stat_prop_values)
        stat_prop_values = cluster_masker.transform(stat_prop_img)

        return stat_prop_values.flatten()


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

    def _transform(self, expid, label_map, sign, result):
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
        label_arr = label_map.get_fdata()
        clust_ids = sorted(list(np.unique(label_arr)[1:]))

        if self._is_pairwaise_estimator:
            coordinates_df = (
                result.estimator.inputs_["coordinates1"]
                if sign == "PositiveTail"
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


class FocusFilter(NiMAREBase):
    """Remove coordinates outside of the Dataset's mask from the Dataset.

    .. versionadded:: 0.0.13

    Parameters
    ----------
    mask : :obj:`str`, :class:`~nibabel.nifti1.Nifti1Image`, \
    :class:`~nilearn.maskers.NiftiMasker` or similar, or None, optional
        Mask(er) to use. If None, uses the masker of the Dataset provided in ``transform``.

    Notes
    -----
    This filter removes any coordinates outside of the brain mask.
    It does not remove studies without coordinates in the brain mask, since a Dataset does not
    need to have coordinates for all studies (e.g., some may only have images).
    """

    def __init__(self, mask=None):
        if mask is not None:
            mask = get_masker(mask)

        self.masker = mask

    def transform(self, dataset):
        """Apply the filter to a Dataset.

        Parameters
        ----------
        dataset : :obj:`~nimare.dataset.Dataset`
            The Dataset to filter.

        Returns
        -------
        dataset : :obj:`~nimare.dataset.Dataset`
            The filtered Dataset.
        """
        masker = self.masker or dataset.masker

        # Get matrix indices for in-brain voxels in the mask
        mask_ijk = np.vstack(np.where(masker.mask_img.get_fdata())).T

        # Get matrix indices for Dataset coordinates
        dset_xyz = dataset.coordinates[["x", "y", "z"]].values

        # mm2vox automatically rounds the coordinates
        dset_ijk = mm2vox(dset_xyz, masker.mask_img.affine)

        # Check if each coordinate in Dataset is within the mask
        # If it is, log that coordinate in keep_idx
        keep_idx = [
            i
            for i, coord in enumerate(dset_ijk)
            if len(np.where((mask_ijk == coord).all(axis=1))[0])
        ]
        LGR.info(
            f"{dset_ijk.shape[0] - len(keep_idx)}/{dset_ijk.shape[0]} coordinates fall outside of "
            "the mask. Removing them."
        )

        # Only retain coordinates inside the brain mask
        dataset.coordinates = dataset.coordinates.iloc[keep_idx]

        return dataset
