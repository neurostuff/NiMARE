"""Methods for diagnosing problems in meta-analytic datasets or analyses."""
import copy
import logging
from functools import partial
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.funcs import squeeze_image
from nilearn import image, input_data
from scipy import ndimage
from tqdm.auto import tqdm

from .base import NiMAREBase
from .utils import vox2mm

LGR = logging.getLogger(__name__)


class Jackknife(NiMAREBase):
    """Run a jackknife analysis on a meta-analysis result.

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
    n_cores : :obj:`int`, optional
        Number of cores to use for parallelization.
        If <=0, defaults to using all available cores.
        Default is 1.

    Notes
    -----
    This analysis characterizes the relative contribution of each experiment in a meta-analysis
    to the resulting clusters by looping through experiments, calculating the Estimator's summary
    statistic for all experiments *except* the target experiment, dividing the resulting test
    summary statistics by the summary statistics from the original meta-analysis, and finally
    averaging the resulting proportion values across all voxels in each cluster.

    Warning
    -------
    Pairwise meta-analyses, like ALESubtraction and MKDAChi2, are not yet supported in this
    method.
    """

    def __init__(
        self,
        target_image="z_desc-size_level-cluster_corr-FWE_method-montecarlo",
        voxel_thresh=None,
        n_cores=1,
    ):
        self.target_image = target_image
        self.voxel_thresh = voxel_thresh
        self.n_cores = self._check_ncores(n_cores)

    def transform(self, result):
        """Apply the analysis to a MetaResult.

        Parameters
        ----------
        result : :obj:`nimare.results.MetaResult`
            A MetaResult produced by a coordinate- or image-based meta-analysis.

        Returns
        -------
        contribution_table : :obj:`pandas.DataFrame`
            A DataFrame with information about relative contributions of each experiment to each
            cluster in the thresholded map.
            There is one row for each experiment, as well as one more row at the top of the table
            (below the header), which has the center of mass of each cluster.
            The centers of mass are not guaranteed to fall within the actual clusters, but can
            serve as a useful heuristic for identifying them.
            There is one column for each cluster, with column names being integers indicating the
            cluster's associated value in the ``labeled_cluster_img`` output.
        labeled_cluster_img : :obj:`nibabel.nifti1.Nifti1Image`
            The labeled, thresholded map that is used to identify clusters characterized by this
            analysis.
            Each cluster in the map has a single value, which corresponds to the cluster's column
            name in ``contribution_table``.
        """
        if not hasattr(result.estimator, "dataset"):
            raise AttributeError(
                "MetaResult was not generated by an Estimator with a `dataset` attribute. "
                "This may be because the Estimator was a pairwise Estimator. "
                "The Jackknife method does not currently work with pairwise Estimators."
            )
        dset = result.estimator.dataset
        # We need to copy the estimator because it will otherwise overwrite the original version
        # with one missing a study in its inputs.
        estimator = copy.deepcopy(result.estimator)
        original_masker = estimator.masker

        # Collect the thresholded cluster map
        if self.target_image in result.maps:
            target_img = result.get_map(self.target_image, return_type="image")
        else:
            available_maps = [f"'{m}'" for m in result.maps.keys()]
            raise ValueError(
                f"Target image ('{self.target_image}') not present in result. "
                f"Available maps in result are: {', '.join(available_maps)}."
            )

        if self.voxel_thresh:
            thresh_img = image.threshold_img(target_img, self.voxel_thresh)
        else:
            thresh_img = target_img

        thresh_arr = thresh_img.get_fdata()

        # CBMAs have "stat" maps, while most IBMAs have "est" maps.
        # Fisher's and Stouffer's only have "z" maps though.
        if "est" in result.maps:
            target_value_map = "est"
        elif "stat" in result.maps:
            target_value_map = "stat"
        else:
            target_value_map = "z"

        stat_values = result.get_map(target_value_map, return_type="array")

        # Use study IDs in inputs_ instead of dataset, because we don't want to try fitting the
        # estimator to a study that might have been filtered out by the estimator's criteria.
        meta_ids = estimator.inputs_["id"]
        rows = ["Center of Mass"] + list(meta_ids)

        # Let's label the clusters in the thresholded map so we can use it as a NiftiLabelsMasker
        # This won't work when the Estimator's masker isn't a NiftiMasker... :(
        conn = np.zeros((3, 3, 3), int)
        conn[:, :, 1] = 1
        conn[:, 1, :] = 1
        conn[1, :, :] = 1
        labeled_cluster_arr, n_clusters = ndimage.measurements.label(thresh_arr, conn)
        labeled_cluster_img = nib.Nifti1Image(
            labeled_cluster_arr,
            affine=target_img.affine,
            header=target_img.header,
        )

        if n_clusters == 0:
            LGR.warning("No clusters found")
            contribution_table = pd.DataFrame(index=rows)
            return contribution_table, labeled_cluster_img

        # Identify center of mass for each cluster
        # This COM may fall outside the cluster, but it is a useful heuristic for identifying them
        cluster_ids = list(range(1, n_clusters + 1))
        cluster_coms = ndimage.center_of_mass(
            labeled_cluster_arr,
            labeled_cluster_arr,
            cluster_ids,
        )
        cluster_coms = np.array(cluster_coms)
        cluster_coms = vox2mm(cluster_coms, target_img.affine)

        cluster_com_strs = []
        for i_peak in range(len(cluster_ids)):
            x, y, z = cluster_coms[i_peak, :].astype(int)
            xyz_str = f"({x}, {y}, {z})"
            cluster_com_strs.append(xyz_str)

        # Mask using a labels masker, so that we can easily get the mean value for each cluster
        cluster_masker = input_data.NiftiLabelsMasker(labeled_cluster_img)
        cluster_masker.fit(labeled_cluster_img)

        # Create empty contribution table
        contribution_table = pd.DataFrame(index=rows, columns=cluster_ids)
        contribution_table.index.name = "Cluster ID"
        contribution_table.loc["Center of Mass"] = cluster_com_strs

        # For the private method that does the work of the analysis, we have a lot of constant
        # parameters, and only one that varies (the ID of the study being removed),
        # so we use functools.partial.
        transform_method = partial(
            self._transform,
            all_ids=meta_ids,
            dset=dset,
            estimator=estimator,
            target_value_map=target_value_map,
            stat_values=stat_values,
            original_masker=original_masker,
            cluster_masker=cluster_masker,
        )
        with Pool(self.n_cores) as p:
            jackknife_results = list(tqdm(p.imap(transform_method, meta_ids), total=len(meta_ids)))

        # Add the results to the table
        for expid, stat_prop_values in jackknife_results:
            contribution_table.loc[expid] = stat_prop_values

        return contribution_table, labeled_cluster_img

    def _transform(
        self,
        expid,
        all_ids,
        dset,
        estimator,
        target_value_map,
        stat_values,
        original_masker,
        cluster_masker,
    ):
        # Fit Estimator to all studies except the target study
        other_ids = [id_ for id_ in all_ids if id_ != expid]
        temp_dset = dset.slice(other_ids)
        temp_result = estimator.fit(temp_dset)

        # Collect the target values (e.g., ALE values) from the N-1 meta-analysis
        temp_stat_img = temp_result.get_map(target_value_map, return_type="image")
        temp_stat_vals = np.squeeze(original_masker.transform(temp_stat_img))

        # Voxelwise proportional reduction of each statistic after removal of the experiment
        with np.errstate(divide="ignore", invalid="ignore"):
            prop_values = np.true_divide(temp_stat_vals, stat_values)
            prop_values = np.nan_to_num(prop_values)

        voxelwise_stat_prop_values = 1 - prop_values

        # Now get the cluster-wise mean of the proportion values
        # pending resolution of https://github.com/nilearn/nilearn/issues/2724
        try:
            stat_prop_img = original_masker.inverse_transform(voxelwise_stat_prop_values)
        except IndexError:
            stat_prop_img = squeeze_image(
                original_masker.inverse_transform([voxelwise_stat_prop_values])
            )

        stat_prop_values = cluster_masker.transform(stat_prop_img)
        return expid, stat_prop_values
