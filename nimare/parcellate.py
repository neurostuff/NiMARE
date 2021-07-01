"""Parcellation tools."""
import datetime
import inspect
import logging
import os
from tempfile import mkstemp

import numpy as np
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

        # Clean up MACM data memmap
        LGR.info(f"Removing temporary file: {memmap_filename}")
        os.remove(memmap_filename)

        images = {"labels": labels}
        return images

    def _filter_selection(self, labels):
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

    def _silhouette(self, data, labels):
        """Calculate silhouette score."""
        from sklearn.metrics import silhouette_score
        silhouette = silhouette_score(data, labels, metric="euclidean", random_state=None)
        return silhouette

    def _voxel_misclassification(self):
        pass

    def _variation_of_information(self, X, Y):
        """Calculate variation of information metric.

        Parameters
        ----------
        X, Y : :obj:`list` of :obj:`list`
            Partitions to compare

        Notes
        -----
        From https://gist.github.com/jwcarr/626cbc80e0006b526688

        Examples
        --------
        >>> X = [[1, 2, 3], [4, 5, 6, 7]]
        >>> Y = [[1, 2, 3, 4], [5, 6, 7]]
        >>> variation_of_information(X, Y)
        0.9271749993818661
        """
        from math import log

        n = float(sum([len(x) for x in X]))
        sigma = 0.0
        for x in X:
            p = len(x) / n
            for y in Y:
                q = len(y) / n
                r = len(set(x) & set(y)) / n
                if r > 0.0:
                    sigma += r * (log(r / p, 2) + log(r / q, 2))
        return abs(sigma)

    def _nondominant_voxel_percentage(self):
        pass

    def _cluster_distance_ratio(self):
        pass

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
