"""Methods for diagnosing problems in meta-analytic datasets or analyses."""
from abc import ABCMeta

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import input_data
from scipy import ndimage
from tqdm.auto import tqdm

from .utils import vox2mm


class Jackknife(metaclass=ABCMeta):
    """Run a jackknife analysis on a meta-analysis result."""

    def __init__(self):
        ...

    def transform(self, result):
        dset = result.estimator.dataset
        estimator = result.estimator
        original_masker = estimator.masker
        # Using z because log-p will not always have value of zero for non-cluster voxels
        cfwe_img = result.get_map(
            "z_level-cluster_corr-FWE_method-montecarlo",
            return_type="image",
        )
        stat_values = result.get_map("stat", return_type="array")
        cfwe_arr = cfwe_img.get_fdata()

        # Let's label the clusters in the cFWE map so we can use it as a NiftiLabelsMasker
        # This won't work when the Estimator's masker isn't a NiftiMasker... :(
        conn = np.zeros((3, 3, 3), int)
        conn[:, :, 1] = 1
        conn[:, 1, :] = 1
        conn[1, :, :] = 1
        cfwe_arr_labeled, n_clusters = ndimage.measurements.label(cfwe_arr, conn)
        cluster_ids = list(range(1, n_clusters + 1))
        coms = ndimage.center_of_mass(cfwe_arr_labeled, cfwe_arr_labeled, cluster_ids)
        coms = np.array(coms)
        coms = vox2mm(coms, cfwe_img.affine)

        cfwe_img_labeled = nib.Nifti1Image(
            cfwe_arr_labeled,
            affine=cfwe_img.affine,
            header=cfwe_img.header,
        )

        cluster_peak_strs = []
        for i_peak in range(len(cluster_ids)):
            x, y, z = coms[i_peak, :].astype(int)
            xyz_str = f"({x}, {y}, {z})"
            cluster_peak_strs.append(xyz_str)

        # Mask using a labels masker
        cluster_masker = input_data.NiftiLabelsMasker(cfwe_img_labeled)
        cluster_masker.fit(cfwe_img_labeled)

        # Should probably go off IDs in estimator.inputs_ instead,
        # in case some experiments are filtered out based on available data.
        rows = ["Center of Mass"] + list(dset.ids)
        output = pd.DataFrame(index=rows, columns=cluster_ids)
        output.index.name = "Cluster ID"
        output.loc["Center of Mass"] = cluster_peak_strs

        for i_expid in tqdm(range(len(dset.ids))):
            expid = dset.ids[i_expid]
            other_ids = [id_ for id_ in dset.ids if id_ != expid]
            temp_dset = dset.slice(other_ids)
            temp_result = estimator.fit(temp_dset)
            temp_stat_img = temp_result.get_map("stat", return_type="image")

            temp_stat_vals = np.squeeze(original_masker.transform(temp_stat_img))
            # Voxelwise proportional reduction of each statistic after removal of the experiment
            with np.errstate(divide="ignore", invalid="ignore"):
                prop_values = np.true_divide(temp_stat_vals, stat_values)
                prop_values = np.nan_to_num(prop_values)

            voxelwise_stat_prop_values = 1 - prop_values
            stat_prop_img = original_masker.inverse_transform(voxelwise_stat_prop_values)

            # Now get the cluster-wise mean of the proportion values
            stat_prop_values = cluster_masker.transform(stat_prop_img)
            output.loc[expid] = stat_prop_values

        return output, cfwe_img_labeled
