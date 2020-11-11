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
