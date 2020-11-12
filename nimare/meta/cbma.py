"""
CBMA methods from the ALE and MKDA families.
"""
import os
import logging
import multiprocessing as mp
from abc import abstractmethod
import inspect

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm.auto import tqdm

from .. import references
from ..base import MetaEstimator
from ..stats import null_to_p
from ..transforms import p_to_z
from ..utils import round2
from .kernel import ALEKernel

LGR = logging.getLogger(__name__)


class CBMAEstimator(MetaEstimator):
    """Base class for coordinate-based meta-analysis methods.

    Parameters
    ----------
    kernel_transformer : :obj:`nimare.base.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset. Default is
        ALEKernel.
    *args
        Optional arguments to the :obj:`nimare.base.MetaEstimator` __init__
        (called automatically).
    **kwargs
        Optional keyword arguments to the :obj:`nimare.base.MetaEstimator`
        __init__ (called automatically).
    """

    def __init__(self, kernel_transformer, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get kernel transformer
        kernel_args = {
            k.split("kernel__")[1]: v for k, v in kwargs.items() if k.startswith("kernel__")
        }

        # Allow both instances and classes for the kernel transformer input.
        from .kernel import KernelTransformer

        if not issubclass(type(kernel_transformer), KernelTransformer) and not issubclass(
            kernel_transformer, KernelTransformer
        ):
            raise ValueError(
                'Argument "kernel_transformer" must be a kind of ' "KernelTransformer"
            )
        elif not inspect.isclass(kernel_transformer) and kernel_args:
            LGR.warning(
                'Argument "kernel_transformer" has already been '
                "initialized, so kernel arguments will be ignored: "
                "{}".format(", ".join(kernel_args.keys()))
            )
        elif inspect.isclass(kernel_transformer):
            kernel_transformer = kernel_transformer(**kernel_args)
        self.kernel_transformer = kernel_transformer

    def _preprocess_input(self, dataset):
        """Mask required input images using either the dataset's mask or the
        estimator's. Also, insert required metadata into coordinates DataFrame.
        """
        super()._preprocess_input(dataset)

        # All extra (non-ijk) parameters for a kernel should be overrideable as
        # parameters to __init__, so we can access them with get_params()
        kt_args = list(self.kernel_transformer.get_params().keys())

        # Integrate "sample_size" from metadata into DataFrame so that
        # kernel_transformer can access it.
        if "sample_size" in kt_args:
            if "sample_sizes" in dataset.get_metadata():
                # Extract sample sizes and make DataFrame
                sample_sizes = dataset.get_metadata(field="sample_sizes", ids=dataset.ids)
                # we need an extra layer of lists
                sample_sizes = [[ss] for ss in sample_sizes]
                sample_sizes = pd.DataFrame(
                    index=dataset.ids, data=sample_sizes, columns=["sample_sizes"]
                )
                sample_sizes["sample_size"] = sample_sizes["sample_sizes"].apply(np.mean)
                # Merge sample sizes df into coordinates df
                self.inputs_["coordinates"] = self.inputs_["coordinates"].merge(
                    right=sample_sizes,
                    left_on="id",
                    right_index=True,
                    sort=False,
                    validate="many_to_one",
                    suffixes=(False, False),
                    how="left",
                )
            else:
                LGR.warning(
                    'Metadata field "sample_sizes" not found. '
                    "Set a constant sample size as a kernel transformer "
                    "argument, if possible."
                )

    def compute_summarystat(self, data):
        """Compute OF scores from data.

        Parameters
        ----------
        data : array, pandas.DataFrame, or list of img_like
            Data from which to estimate ALE scores.
            The data can be:
            (1) a 1d contrast-len or 2d contrast-by-voxel array of MA values,
            (2) a DataFrame containing coordinates to produce MA values,
            or (3) a list of imgs containing MA values.

        Returns
        -------
        stat_values : 1d array
            OF values. One value per voxel.
        """
        if isinstance(data, pd.DataFrame):
            ma_values = self.kernel_transformer.transform(
                data, masker=self.masker, return_type="array"
            )
        elif isinstance(data, list):
            ma_values = self.masker.transform(data)
        elif isinstance(data, np.ndarray):
            ma_values = data
        elif not isinstance(data, np.ndarray):
            raise ValueError('Unsupported data type "{}"'.format(type(data)))

        # Apply weights before returning
        return self._compute_summarystat(ma_values)

    @abstractmethod
    def _compute_summarystat(self, ma_values):
        pass

    def _compute_null_empirical(self, ma_maps, n_iters=10000):
        """Compute uncorrected null distribution using empirical method.

        Parameters
        ----------
        ma_maps : (C x V) array
            Contrast by voxel array of MA values, after weighting with
            weight_vec.

        Notes
        -----
        This method adds one entry to the null_distributions_ dict attribute:
        "empirical_null".
        """
        n_studies, n_voxels = ma_maps.shape
        null_ijk = np.random.choice(np.arange(n_voxels), (n_iters, n_studies))
        iter_ma_values = ma_maps[np.arange(n_studies), tuple(null_ijk)].T
        null_dist = self.compute_summarystat(iter_ma_values)
        self.null_distributions_["empirical_null"] = null_dist

    def _summarystat_to_p(self, stat_values, null_method="analytic"):
        """
        Compute p- and z-values from summary statistics (e.g., ALE scores) and
        either histograms from analytic null or null distribution from
        empirical null.

        Parameters
        ----------
        stat_values : 1D array_like
            Array of summary statistic values from estimator.
        null_method : {"analytic", "empirical"}, optional
            Whether to use analytic null or empirical null.
            Default is "analytic".

        Returns
        -------
        p_values, z_values : 1D array
            P- and Z-values for statistic values.
            Same shape as stat_values.
        """

        if null_method == "analytic":
            assert "histogram_bins" in self.null_distributions_.keys()
            assert "histogram_weights" in self.null_distributions_.keys()

            step = 1 / np.mean(np.diff(self.null_distributions_["histogram_bins"]))

            # Determine p- and z-values from stat values and null distribution.
            p_values = np.ones(stat_values.shape)
            idx = np.where(stat_values > 0)[0]
            stat_bins = round2(stat_values[idx] * step)
            p_values[idx] = self.null_distributions_["histogram_weights"][stat_bins]

        elif null_method == "empirical":
            assert "empirical_null" in self.null_distributions_.keys()
            p_values = null_to_p(
                stat_values, self.null_distributions_["empirical_null"], tail="upper"
            )
        else:
            raise ValueError("Argument 'null_method' must be one of: 'analytic', 'empirical'.")

        z_values = p_to_z(p_values, tail="one")
        return p_values, z_values
