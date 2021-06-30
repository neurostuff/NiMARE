"""Parcellation tools."""
import inspect
import logging

import numpy as np
from sklearn.cluster import KMeans

from .base import NiMAREBase
from .meta.base import CBMAEstimator
from .results import MetaResult
from .utils import add_metadata_to_dataframe, check_type, listify, use_memmap, vox2mm

LGR = logging.getLogger(__name__)


class CoordCBP(NiMAREBase):
    """Base class for parcellators in :mod:`nimare.parcellate`.

    .. versionadded:: 0.0.10

    """

    _required_inputs = {"coordinates": ("coordinates", None)}

    def __init__(
        self,
        target_mask,
        meta_estimator,
        target_image,
        n_clusters,
        r=None,
        n=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Get kernel transformer
        meta_estimator = check_type(meta_estimator, CBMAEstimator)
        self.meta_estimator = meta_estimator
        self.target_image = target_image
        self.n_clusters = listify(n_clusters)

        if r and n:
            raise ValueError("Only one of 'r' and 'n' may be provided.")
        elif not r and not n:
            raise ValueError("Either 'r' or 'n' must be provided.")
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
        """
        Perform coordinate-based meta-analysis on dataset.

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

        # Correlate voxel-wise MACM results
        data = np.corrcoef(data)

        # Convert voxel-wise correlation matrix to distances
        data = 1 - data

        # Perform clustering
        labels = np.zeros((len(self.n_clusters), len(macm_data)), dtype=int)
        for i_cluster, cluster_count in enumerate(self.n_clusters):
            kmeans = KMeans(n_clusters=cluster_count, random_state=0).fit(data)
            labels[i_cluster, :] = kmeans.labels_

        images = {"labels": labels}
        return images

    def fit(self, dataset, drop_invalid=True):
        self._validate_input(dataset, drop_invalid=drop_invalid)
        self._preprocess_input(dataset)
        maps = self._fit(dataset)

        if hasattr(self, "masker") and self.masker is not None:
            masker = self.masker
        else:
            masker = dataset.masker

        self.results = MetaResult(self, masker, maps)
        return self.results
