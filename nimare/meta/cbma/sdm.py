"""CBMA methods from the Seed-based d Mapping (SDM) family."""

import logging

import numpy as np
import sparse
from joblib import Memory

from nimare import _version
from nimare.meta.cbma.base import CBMAEstimator
from nimare.meta.kernel import SDMKernel

LGR = logging.getLogger(__name__)
__version__ = _version.get_versions()["version"]


class SDM(CBMAEstimator):
    """Seed-based d Mapping (SDM) meta-analysis.

    This is a simplified implementation of the SDM algorithm that generates
    effect size maps from coordinates using an anisotropic Gaussian kernel
    and performs a random-effects meta-analysis.

    .. versionadded:: 0.3.1

    Parameters
    ----------
    kernel_transformer : :obj:`~nimare.meta.kernel.KernelTransformer`, optional
        Kernel with which to convolve coordinates from dataset.
        Default is SDMKernel with FWHM=20mm.
    memory : instance of :class:`joblib.Memory`, :obj:`str`, or :class:`pathlib.Path`
        Used to cache the output of a function. By default, no caching is done.
        If a :obj:`str` is given, it is the path to the caching directory.
    memory_level : :obj:`int`, default=0
        Rough estimator of the amount of memory used by caching.
        Higher value means more memory for caching. Zero means no caching.
    **kwargs
        Keyword arguments. Arguments for the kernel_transformer can be assigned here,
        with the prefix ``kernel__`` in the variable name.
        Another optional argument is ``mask``.

    Attributes
    ----------
    masker : :class:`~nilearn.input_data.NiftiMasker` or similar
        Masker object.
    inputs_ : :obj:`dict`
        Inputs to the Estimator. For CBMA estimators, there is only one key: coordinates.
        This is an edited version of the dataset's coordinates DataFrame.

    Notes
    -----
    This is a simplified implementation that focuses on the core SDM kernel-based approach.
    The full SDM-PSI algorithm includes additional sophisticated steps such as:

    - Multiple imputation of missing data
    - Subject-level image simulation
    - Rubin's rules for combining results across imputations
    - Advanced permutation testing procedures

    These advanced features are not yet implemented in this version.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    :class:`~nimare.meta.kernel.SDMKernel`:
        The kernel used by this estimator.
    """

    def __init__(
        self,
        kernel_transformer=SDMKernel,
        memory=Memory(location=None, verbose=0),
        memory_level=0,
        **kwargs,
    ):
        if not (isinstance(kernel_transformer, SDMKernel) or kernel_transformer == SDMKernel):
            LGR.warning(
                f"The KernelTransformer being used ({kernel_transformer}) is not optimized "
                f"for the {type(self).__name__} algorithm. "
                "Expect suboptimal performance and beware bugs."
            )

        # Add kernel transformer attribute and process keyword arguments
        super().__init__(
            kernel_transformer=kernel_transformer,
            memory=memory,
            memory_level=memory_level,
            **kwargs,
        )
        self.dataset = None

    def _generate_description(self):
        """Generate a description of the fitted Estimator.

        Returns
        -------
        str
            Description of the Estimator.
        """
        description = (
            "A Seed-based d Mapping (SDM) meta-analysis was performed "
            f"with NiMARE {__version__} "
            "(RRID:SCR_017398; \\citealt{Salo2023}), using an "
            f"{self.kernel_transformer.__class__.__name__.replace('Kernel', '')} kernel. "
            f"{self.kernel_transformer._generate_description()} "
            "The summary statistic images were combined across studies using a simple mean, "
            "which provides a basic estimate of the meta-analytic effect size at each voxel."
        )
        return description

    def _compute_summarystat_est(self, ma_values):
        """Compute summary statistic (mean) for SDM.

        Parameters
        ----------
        ma_values : array or sparse array
            Modeled activation values, typically a studies-by-voxels array.

        Returns
        -------
        stat_values : 1d array
            Mean activation values across studies.
        """
        # Compute mean across studies
        stat_values = np.mean(ma_values, axis=0)

        # Handle sparse arrays
        if isinstance(stat_values, sparse._coo.core.COO):
            mask_data = self.masker.mask_img.get_fdata().astype(bool)
            stat_values = stat_values.todense().reshape(-1)
            stat_values = stat_values[mask_data.reshape(-1)]

        return stat_values

    def _fit(self, dataset):
        """Perform SDM meta-analysis on dataset.

        Parameters
        ----------
        dataset : :obj:`~nimare.dataset.Dataset`
            Dataset to analyze.
        """
        self.dataset = dataset
        self.masker = self.masker or dataset.masker

        # Collect modeled activation maps from coordinates
        ma_values = self._collect_ma_maps(
            coords_key="coordinates",
            maps_key="ma_maps",
        )

        # Compute summary statistic using the _compute_summarystat_est method
        stat_values = self._compute_summarystat_est(ma_values)
        n_studies = ma_values.shape[0]

        # Compute standard error and z-scores
        # Convert sparse to dense if needed for std calculation
        if isinstance(ma_values, sparse._coo.core.COO):
            ma_dense = ma_values.todense()
            mask_data = self.masker.mask_img.get_fdata().astype(bool)
            ma_dense_reshaped = ma_dense.reshape(n_studies, -1)
            ma_masked = ma_dense_reshaped[:, mask_data.reshape(-1)]
        else:
            ma_masked = ma_values

        # Calculate standard deviation and standard error
        std_values = np.std(ma_masked, axis=0, ddof=1)
        se_values = std_values / np.sqrt(n_studies)

        # Avoid division by zero
        se_values[se_values == 0] = np.finfo(float).eps

        z_values = stat_values / se_values

        # Convert z to p-values using scipy (two-tailed)
        from scipy import stats
        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_values)))

        # Create output maps
        images = {
            "stat": stat_values,
            "z": z_values,
            "p": p_values,
            "dof": np.full_like(stat_values, n_studies - 1),
        }

        description = self._generate_description()

        return images, {}, description
