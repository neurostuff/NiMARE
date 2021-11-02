"""Methods for diagnosing problems in meta-analytic datasets or analyses."""
import copy
import logging
from abc import ABCMeta

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image, input_data
from scipy import ndimage
from tqdm.auto import tqdm

from .utils import vox2mm

LGR = logging.getLogger(__name__)


class Jackknife(metaclass=ABCMeta):
    """Run a jackknife analysis on a meta-analysis result.

    Parameters
    ----------
    target_image
        The default is z because log-p will not always have value of zero for non-cluster voxels.
    voxel_thresh
    n_cores

    Notes
    -----
    This analysis characterizes the relative contribution of each experiment in a meta-analysis
    to the resulting clusters by looping through experiments, calculating the Estimator's summary
    statistic for all experiments *except* the target experiment, dividing the resulting test
    summary statistics by the summary statistics from the original meta-analysis, and finally
    averaging the resulting proportion values across all voxels in each cluster.
    """

    def __init__(
        self,
        target_image="z_level-cluster_corr-FWE_method-montecarlo",
        voxel_thresh=None,
        n_cores=1,
    ):
        self.target_image = target_image
        self.voxel_thresh = voxel_thresh
        self.n_cores = n_cores

    def transform(self, result):
        """Apply the analysis to a MetaResult.

        Parameters
        ----------
        result : :obj:`nimare.results.MetaResult`
            A MetaResult produced by a coordinate- or image-based meta-analysis.
            Multiple comparisons correction must be performed prior to applying the Jackknife.

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
        dset = result.estimator.dataset
        # We need to copy the estimator because it will otherwise overwrite the original version
        # with one missing a study in its inputs.
        estimator = copy.deepcopy(result.estimator)
        original_masker = estimator.masker

        target_img = result.get_map(self.target_image, return_type="image")
        if self.voxel_thresh:
            thresh_img = image.threshold_img(target_img, self.voxel_thresh)
        else:
            thresh_img = target_img

        # CBMAs have "stat" maps, while most IBMAs have "est" maps.
        # Fisher's and Stouffer's only have "z" maps though.
        if "est" in result.maps:
            target_value_map = "est"
        elif "stat" in result.maps:
            target_value_map = "stat"
        else:
            target_value_map = "z"

        stat_values = result.get_map(target_value_map, return_type="array")

        thresh_arr = thresh_img.get_fdata()

        # Should probably go off IDs in estimator.inputs_ instead,
        # in case some experiments are filtered out based on available data.
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

        cluster_ids = list(range(1, n_clusters + 1))
        coms = ndimage.center_of_mass(labeled_cluster_arr, labeled_cluster_arr, cluster_ids)
        coms = np.array(coms)
        coms = vox2mm(coms, target_img.affine)

        cluster_peak_strs = []
        for i_peak in range(len(cluster_ids)):
            x, y, z = coms[i_peak, :].astype(int)
            xyz_str = f"({x}, {y}, {z})"
            cluster_peak_strs.append(xyz_str)

        # Mask using a labels masker
        cluster_masker = input_data.NiftiLabelsMasker(labeled_cluster_img)
        cluster_masker.fit(labeled_cluster_img)

        # Compile contribution table
        contribution_table = pd.DataFrame(index=rows, columns=cluster_ids)
        contribution_table.index.name = "Cluster ID"
        contribution_table.loc["Center of Mass"] = cluster_peak_strs

        for i_expid in tqdm(range(len(meta_ids))):
            expid = meta_ids[i_expid]
            other_ids = [id_ for id_ in meta_ids if id_ != expid]
            temp_dset = dset.slice(other_ids)
            temp_result = estimator.fit(temp_dset)
            temp_stat_img = temp_result.get_map(target_value_map, return_type="image")

            temp_stat_vals = np.squeeze(original_masker.transform(temp_stat_img))
            # Voxelwise proportional reduction of each statistic after removal of the experiment
            with np.errstate(divide="ignore", invalid="ignore"):
                prop_values = np.true_divide(temp_stat_vals, stat_values)
                prop_values = np.nan_to_num(prop_values)

            voxelwise_stat_prop_values = 1 - prop_values
            stat_prop_img = original_masker.inverse_transform(voxelwise_stat_prop_values)

            # Now get the cluster-wise mean of the proportion values
            stat_prop_values = cluster_masker.transform(stat_prop_img)
            contribution_table.loc[expid] = stat_prop_values

        return contribution_table, labeled_cluster_img
