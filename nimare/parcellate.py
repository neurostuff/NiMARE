"""Parcellation tools."""
import datetime
import inspect
import logging
import os
from tempfile import mkstemp

import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans

from .base import NiMAREBase
from .extract.utils import _get_dataset_dir
from .meta.base import CBMAEstimator
from .meta.cbma.ale import ALE
from .results import MetaResult
from .utils import add_metadata_to_dataframe, check_type, listify, use_memmap, vox2mm

LGR = logging.getLogger(__name__)


class CoordCBP(NiMAREBase):
    """Perform coordinate-based coactivation-based parcellation.

    .. versionadded:: 0.0.10

    Parameters
    ----------
    target_mask : :obj:`nibabel.Nifti1.Nifti1Image`
        Mask of target of parcellation.
        Currently must be in same space/resolution as Dataset mask.
    n_clusters : :obj:`list` of :obj:`int`
        Number of clusters to evaluate in clustering.
        Metrics will be calculated for each cluster-count in the list, to allow users to select
        the optimal cluster solution.
    r : :obj:`float` or :obj:`list` of :obj:`float` or None, optional
        Radius (in mm) within which to find studies.
        If a list of values is provided, then MACMs and clustering will be performed across all
        values, and a selection procedure will be performed to identify the optimal ``r``.
        Mutually exclusive with ``n``. Default is None.
    n : :obj:`int` or :obj:`list` of :obj:`int` or None, optional
        Number of closest studies to identify.
        If a list of values is provided, then MACMs and clustering will be performed across all
        values, and a selection procedure will be performed to identify the optimal ``n``.
        Mutually exclusive with ``r``. Default is None.
    meta_estimator : :obj:`nimare.meta.base.CBMAEstimator`, optional
        CBMA Estimator with which to run the MACMs.
        Default is :obj:`nimare.meta.cbma.ale.ALE`.
    target_image : :obj:`str`, optional
        Name of meta-analysis results image to use for clustering.
        Default is "ale", which is specific to the ALE estimator.

    Notes
    -----
    This approach deviates in a number of respects from the method described in
    Chase et al. (2020), including the fact that the scikit-learn implementation of the K-means
    clustering algorithm only supports Euclidean distance, so correlation distance is not used.
    """

    _required_inputs = {"coordinates": ("coordinates", None)}

    def __init__(
        self,
        target_mask,
        n_clusters,
        r=None,
        n=None,
        meta_estimator=None,
        target_image="stat",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if meta_estimator is None:
            meta_estimator = ALE()
        else:
            meta_estimator = check_type(meta_estimator, CBMAEstimator)

        if r and n:
            raise ValueError("Only one of 'r' and 'n' may be provided.")
        elif not r and not n:
            raise ValueError("Either 'r' or 'n' must be provided.")

        self.meta_estimator = meta_estimator
        self.target_image = target_image
        self.n_clusters = listify(n_clusters)
        self.filter_selection = isinstance(r, list) or isinstance(n, list)
        self.filter_type = "r" if r else "n"
        self.alpha = 0.05
        self.r = listify(r)
        self.n = listify(n)

    def _preprocess_input(self, dataset):
        """Mask required input images using either the dataset's mask or the estimator's.

        Also, insert required metadata into coordinates DataFrame.
        """
        super()._preprocess_input(dataset)

        # All extra (non-ijk) parameters for a kernel should be overrideable as
        # parameters to __init__, so we can access them with get_params()
        kt_args = list(self.meta_estimator.kernel_transformer.get_params().keys())

        # Integrate "sample_size" from metadata into DataFrame so that
        # kernel_transformer can access it.
        if "sample_size" in kt_args:
            self.inputs_["coordinates"] = add_metadata_to_dataframe(
                dataset,
                self.inputs_["coordinates"],
                metadata_field="sample_sizes",
                target_column="sample_size",
                filter_func=np.mean,
            )

    @use_memmap(LGR, n_files=1)
    def _fit(self, dataset):
        """Perform coordinate-based coactivation-based parcellation on dataset.

        Parameters
        ----------
        dataset : :obj:`nimare.dataset.Dataset`
            Dataset to analyze.
        """
        self.dataset = dataset
        self.masker = self.masker or dataset.masker

        # Loop through voxels in target_mask, selecting studies for each and running MACMs (no MCC)
        target_ijk = np.vstack(np.where(self.target_mask.get_fdata()))
        target_xyz = vox2mm(target_ijk, self.masker.mask_img.affine)
        n_target_voxels = target_xyz.shape[1]
        n_mask_voxels = self.masker.transform(self.masker.mask_img).shape[0]

        n_filters = len(getattr(self, self.filter_type))
        labels = np.zeros((n_filters, len(self.n_clusters), n_target_voxels), dtype=int)
        silhouettes = np.zeros((n_filters, len(self.n_clusters)), dtype=float)
        kwargs = {"r": None, "n": None}

        # Use a memmapped 2D array
        start_time = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        dataset_dir = _get_dataset_dir("temporary_files", data_dir=None)
        _, memmap_filename = mkstemp(
            prefix=self.__class__.__name__,
            suffix=start_time,
            dir=dataset_dir,
        )
        data = np.memmap(
            memmap_filename,
            dtype=float,
            mode="w+",
            shape=(n_target_voxels, n_mask_voxels),
        )
        ratios = np.zeros((n_filters, len(self.n_clusters)), dtype=float)

        for i_filter in range(n_filters):
            kwargs[self.filter_type] = getattr(self, self.filter_type)[i_filter]
            for j_coord in n_target_voxels:
                xyz = target_xyz[:, j_coord]
                macm_ids = dataset.get_studies_by_coordinate(xyz, **kwargs)
                coord_dset = dataset.slice(macm_ids)

                # This seems like a somewhat inelegant solution
                # Check if the meta method is a pairwise estimator
                if "dataset2" in inspect.getfullargspec(self.meta_estimator.fit).args:
                    unselected_ids = sorted(list(set(dataset.ids) - set(macm_ids)))
                    unselected_dset = dataset.slice(unselected_ids)
                    self.meta_estimator.fit(coord_dset, unselected_dset)
                else:
                    self.meta_estimator.fit(coord_dset)

                data[j_coord, :] = self.meta_estimator.results.get_map(
                    self.target_image,
                    return_type="array",
                )  # data is overwritten across filters

            # Perform clustering
            for j_cluster, cluster_count in enumerate(self.n_clusters):
                kmeans = KMeans(
                    n_clusters=cluster_count,
                    init="k-means++",
                    n_init=10,
                    random_state=0,
                    algorithm="elkan",
                ).fit(data)
                labels[i_filter, j_cluster, :] = kmeans.labels_

                silhouettes[i_filter, j_cluster] = self._silhouette(data, kmeans.labels_)

                # Necessary calculations for ratios metric
                avg_intracenter_distance = np.mean(
                    distance.pdist(
                        kmeans.cluster_centers_,
                        metric="euclidean",
                    )
                )
                total_voxel_center_distance = 0
                for k_cluster_val in range(cluster_count):
                    cluster_val_voxels = np.where(kmeans.labels_ == k_cluster_val)[0]
                    cluster_val_center = kmeans.cluster_centers_[k_cluster_val, :]
                    cluster_val_features = data[cluster_val_voxels, :]
                    cluster_val_distances = distance.cdist(
                        cluster_val_center,
                        cluster_val_features,
                        metric="euclidean",
                    )
                    total_voxel_center_distance += np.sum(cluster_val_distances)
                avg_voxel_center_distance = total_voxel_center_distance / n_target_voxels
                ratio = avg_voxel_center_distance / avg_intracenter_distance
                ratios[i_filter, j_cluster] = ratio

        # Identify range of optimal filters
        retained_filter_idx = self._filter_selection(labels)
        labels = labels[retained_filter_idx, ...]
        silhouettes = silhouettes[retained_filter_idx, ...]
        ratios = ratios[retained_filter_idx, ...]

        # Calculate metrics
        vm_results = self._voxel_misclassification(labels, alpha=self.alpha)  # boolx2
        vi_results = self._variation_of_information(labels, self.n_clusters)  # floatx2
        s_results = silhouettes  # float
        nvp_results = self._nondominant_voxel_percentage()  # ?
        cdr_results = self._cluster_distance_ratio(ratios)  # float
        vote = np.nansum(
            np.dstack((vm_results, vi_results, s_results, nvp_results, cdr_results)),
            axis=2,
        )
        assert vote.shape == (n_filters, len(self.n_clusters))

        # Clean up MACM data memmap
        LGR.info(f"Removing temporary file: {memmap_filename}")
        os.remove(memmap_filename)

        images = {"labels": labels}
        return images

    def _filter_selection(self, labels):
        """Select a range of optimal filter values based on consistency of cluster assignment.

        Parameters
        ----------
        labels : :obj:`numpy.ndarray` of shape (n_filters, nclusters, n_voxels)
            Labeling results from a range of KMeans clustering runs.

        Notes
        -----
        From Chase et al. (2020):
        We implemented a two-step procedure that involved a decision on those filter sizes to be
        included in the final analysis and subsequently a decision on the optimal cluster solution.
        In the first step, we examined the consistency of the cluster assignment for the individual
        voxels across the cluster solutions of the co-occurrence maps performed at different filter
        sizes. We selected a filter range with the lowest number of deviants, that is, number of
        voxels that were assigned differently compared with the solution from the majority of
        filters. In other words, we identified those filter sizes which produced solutions most
        similar to the consensus-solution across all filter sizes
        """
        from scipy import stats

        n_filters, n_clusters, n_voxels = labels.shape
        deviant_proportions = np.zeros((n_filters, n_clusters))
        for i_cluster in range(n_clusters):
            cluster_labels = labels[:, i_cluster, :]
            cluster_labels_mode = stats.mode(cluster_labels, axis=0)[0]
            is_deviant = cluster_labels != cluster_labels_mode
            deviant_proportion = is_deviant.mean(axis=1)
            assert deviant_proportion.size == n_filters
            deviant_proportions[:, i_cluster] = deviant_proportion
        # Z-score within each cluster solution
        deviant_z = stats.zscore(deviant_proportions, axis=0)
        filter_deviant_z = deviant_z.sum(axis=1)
        min_deviants_filter = np.where(filter_deviant_z == np.min(filter_deviant_z))[0]

        # NOTE: This is not the end, but I don't know how filters were selected based on this array
        return min_deviants_filter

    def _voxel_misclassification(self, selected_filter_labels, alpha=0.05):
        """Calculate voxel misclassification metric.

        Parameters
        ----------
        labels : :obj:`numpy.ndarray` of shape (n_selected_filters, nclusters, n_voxels)
            Labeling results from a range of KMeans clustering runs, limited to those filters
            that survived the filter selection procedure.

        Notes
        -----
        From Chase et al. (2020):
        First, misclassified voxels (deviants) were examined as a topological criterion,
        with optimal K parcellations being those in which the percentage of deviants was not
        significantly increased compared to the K − 1 solution and but where the K + 1 was
        associated with significantly increased deviants.

        TS: Deviants are presumably calculated only from the range of filters selected in the
        filter selection step.

        TS: To measure significance, we could do a series of pair-wise proportion z-tests
        Left side null hypothesis: proportion from K is <= K - 1
        Left side alternative: K > K -1
        Accept on null
        z, p = proportions_ztest([k, k-1], [n, n], alternative="larger")
        p >= alpha --> YAY
        Right side null hypothesis: proportion from K is >= K + 1
        Right side alternative: K < K + 1
        Accept on rejected null
        z, p = proportions_ztest([k, k+1], [n, n], alternative="smaller")
        p < alpha --> YAY
        """
        from scipy import stats
        from statsmodels.stats.proportion import proportions_ztest

        n_filters, n_clusters, n_voxels = selected_filter_labels.shape
        count = n_voxels * n_filters
        deviant_counts = np.zeros((n_filters, n_clusters))
        for i_cluster in range(n_clusters):
            cluster_labels = selected_filter_labels[:, i_cluster, :]
            cluster_labels_mode = stats.mode(cluster_labels, axis=0)[0]
            is_deviant = cluster_labels != cluster_labels_mode
            deviant_count = is_deviant.sum(axis=1)
            assert deviant_count.size == n_filters
            deviant_counts[:, i_cluster] = deviant_count

        decision_criteria = np.empty((n_clusters, 2), dtype=bool)

        for i_cluster in range(n_clusters):
            cluster_deviant_count = np.sum(deviant_counts[:, i_cluster])

            # Compare to k - 1
            if i_cluster > 0:
                m1_cluster_deviant_count = np.sum(deviant_counts[:, i_cluster - 1])
                _, p = proportions_ztest(
                    count=[cluster_deviant_count, m1_cluster_deviant_count],
                    nobs=[count, count],
                    alternative="larger",
                )
                decision_criteria[i_cluster, 0] = p >= alpha
            else:
                decision_criteria[i_cluster, 0] = np.nan

            if i_cluster < (n_clusters - 1):
                p1_cluster_deviant_count = np.sum(deviant_counts[:, i_cluster + 1])
                _, p = proportions_ztest(
                    count=[cluster_deviant_count, p1_cluster_deviant_count],
                    nobs=[count, count],
                    alternative="smaller",
                )
                decision_criteria[i_cluster, 1] = p < alpha
            else:
                decision_criteria[i_cluster, 1] = np.nan

        decision_results = np.all(decision_criteria, axis=1)
        good_cluster_idx = np.where(decision_results)[0]
        if not good_cluster_idx:
            LGR.warning("No cluster solution found according to VM metric.")

        return good_cluster_idx

    def _variation_of_information(self, labels, cluster_counts):
        """Calculate variation of information metric.

        Parameters
        ----------
        labels : :obj:`numpy.ndarray` of shape (n_filters, n_clusters, n_voxels)
        cluster_counts : :obj:`list` of :obj:`int`
            The set of K values tested. Must be n_clusters items long.

        Returns
        -------
        vi_values : :obj:`numpy.ndarray` of shape (n_filters, n_clusters, 2)
            Variation of information values for each filter and each cluster, where the last
            dimension has two values: the VI value for K vs. K - 1 and the value for K vs. K + 1.
            K values with no K - 1 or K + 1 have a NaN in the associated cell.

        Notes
        -----
        From Chase et al. (2020):
        Second, the variation of information (VI) metric was employed as an information-theoretic
        criterion to assess the similarity of cluster assignments for each filter size between
        the current solution and the neighboring (K − 1 and K + 1) solutions.
        """
        from .utils import variation_of_information

        n_filters, n_clusters, _ = labels.shape
        assert len(cluster_counts) == n_clusters

        vi_values = np.empty((n_filters, n_clusters, 2))
        for i_filter in range(n_filters):
            filter_labels = labels[i_filter, :, :]
            for j_cluster, cluster_count in enumerate(cluster_counts):
                cluster_partition = [
                    np.where(filter_labels[j_cluster, :] == k_cluster_num)[0]
                    for k_cluster_num in range(cluster_count)
                ]

                # Calculate VI between K and K - 1
                if j_cluster > 0:
                    cluster_m1_partition = [
                        np.where(filter_labels[j_cluster - 1, :] == k_cluster_num)[0]
                        for k_cluster_num in range(cluster_counts[j_cluster - 1])
                    ]

                    vi_values[i_filter, j_cluster, 0] = variation_of_information(
                        cluster_partition,
                        cluster_m1_partition,
                    )
                else:
                    vi_values[i_filter, j_cluster, 0] = np.nan

                # Calculate VI between K and K + 1
                if j_cluster < (labels.shape[1] - 1):
                    cluster_p1_partition = [
                        np.where(filter_labels[j_cluster + 1, :] == k_cluster_num)[0]
                        for k_cluster_num in range(cluster_counts[j_cluster + 1])
                    ]

                    vi_values[i_filter, j_cluster, 1] = variation_of_information(
                        cluster_partition,
                        cluster_p1_partition,
                    )
                else:
                    vi_values[i_filter, j_cluster, 1] = np.nan

        return vi_values

    def _average_silhouette(self, data, labels):
        """Calculate average silhouette score.

        Notes
        -----
        From Chase et al. (2020):
        Third, the silhouette value averaged across voxels for each filter size was considered a
        cluster separation criterion.
        """
        from sklearn.metrics import silhouette_score

        silhouette = silhouette_score(data, labels, metric="euclidean", random_state=None)
        # Average across voxels
        return silhouette

    def _nondominant_voxel_percentage(self):
        """Calculate percentage-of-voxels-not-with-parent metric.

        Notes
        -----
        From Chase et al. (2020):
        Fourth, we assessed the percentage of voxels not related to the dominant parent cluster
        compared with the K − 1 solution as a second topological criterion.
        This measure corresponds to the percentage voxels that are not present in hierarchy, K,
        compared with the previous K − 1 solution, and is related to the hierarchy index.
        """
        pass

    def _cluster_distance_ratio(self, ratios):
        """Calculate change-in-inter/intra-cluster-distance metric.

        Parameters
        ----------
        ratios

        Notes
        -----
        From Chase et al. (2020):
        Finally, the change in inter- versus intra-cluster distance ratio was computed as a second
        cluster separation criterion.
        This ratio is the first derivative of the ratio between the average distance of a given
        voxel to its own cluster center and the average distance between the cluster centers.
        """
        # TODO: Calculate first derivative?

        return ratios

    def fit(self, dataset, drop_invalid=True):
        """Perform coordinate-based coactivation-based parcellation on dataset.

        Parameters
        ----------
        dataset : :obj:`nimare.dataset.Dataset`
            Dataset to analyze.
        drop_invalid : :obj:`bool`, optional
            Whether to automatically ignore any studies without the required data or not.
            Default is True.
        """
        self._validate_input(dataset, drop_invalid=drop_invalid)
        self._preprocess_input(dataset)
        maps = self._fit(dataset)

        if hasattr(self, "masker") and self.masker is not None:
            masker = self.masker
        else:
            masker = dataset.masker

        self.results = MetaResult(self, masker, maps)
        return self.results
