"""Parcellation tools."""
import inspect
import logging

import numpy as np
from sklearn.cluster import KMeans

from .base import NiMAREBase
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
    r : :obj:`float` or None, optional
        Radius (in mm) within which to find studies. Mutually exclusive with ``n``.
        Default is None.
    n : :obj:`int` or None, optional
        Number of closest studies to identify. Mutually exclusive with ``r``.
        Default is None.
    meta_estimator : :obj:`nimare.meta.base.CBMAEstimator`, optional
        CBMA Estimator with which to run the MACMs.
        Default is :obj:`nimare.meta.cbma.ale.ALE`.
    target_image : :obj:`str`, optional
        Name of meta-analysis results image to use for clustering.
        Default is "ale", which is specific to the ALE estimator.
    """

    _required_inputs = {"coordinates": ("coordinates", None)}

    def __init__(
        self,
        target_mask,
        n_clusters,
        r=None,
        n=None,
        meta_estimator=None,
        target_image="ale",
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
        self.r = r
        self.n = n

    def _preprocess_input(self, dataset):
        """Mask required input images using either the dataset's mask or the estimator's.

        Also, insert required metadata into coordinates DataFrame.
        """
        super()._preprocess_input(dataset)

        # All extra (non-ijk) parameters for a kernel should be overrideable as
        # parameters to __init__, so we can access them with get_params()
        kt_args = list(self.kernel_transformer.get_params().keys())

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
        self.null_distributions_ = {}

        # Loop through voxels in target_mask, selecting studies for each and running MACMs (no MCC)
        target_ijk = np.vstack(np.where(self.target_mask.get_fdata()))
        target_xyz = vox2mm(target_ijk, self.masker.mask_img.affine)
        for i_coord in target_xyz.shape[1]:
            xyz = target_xyz[:, i_coord]
            macm_ids = dataset.get_studies_by_coordinate(xyz, r=self.r, n=self.n)
            coord_dset = dataset.slice(macm_ids)

            # This seems like a somewhat inelegant solution
            # Check if the meta method is a pairwise estimator
            if "dataset2" in inspect.getfullargspec(self.meta_estimator.fit).args:
                unselected_ids = sorted(list(set(dataset.ids) - set(macm_ids)))
                unselected_dset = dataset.slice(unselected_ids)
                self.meta_estimator.fit(coord_dset, unselected_dset)
            else:
                self.meta_estimator.fit(coord_dset)

            macm_data = self.meta_estimator.results.get_map(self.target_image, return_type="array")
            if i_coord == 0:
                data = np.zeros((target_xyz.shape[1], len(macm_data)))

            data[i_coord, :] = macm_data

        # Perform clustering
        labels = np.zeros((len(self.n_clusters), len(macm_data)), dtype=int)
        for i_cluster, cluster_count in enumerate(self.n_clusters):
            kmeans = KMeans(n_clusters=cluster_count, random_state=0).fit(data)
            labels[i_cluster, :] = kmeans.labels_

        images = {"labels": labels}
        return images

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
