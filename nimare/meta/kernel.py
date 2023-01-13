"""Kernel transformers for CBMA algorithms.

Methods for estimating thresholded cluster maps from neuroimaging contrasts
(Contrasts) from sets of foci and optional additional information (e.g., sample
size and test statistic values).
"""
from __future__ import division

import logging

import nibabel as nib
import numpy as np
import pandas as pd

from nimare.base import NiMAREBase
from nimare.meta.utils import compute_ale_ma, compute_kda_ma, get_ale_kernel
from nimare.utils import _add_metadata_to_dataframe, mm2vox

LGR = logging.getLogger(__name__)


class KernelTransformer(NiMAREBase):
    """Base class for modeled activation-generating methods in :mod:`~nimare.meta.kernel`.

    .. versionchanged:: 0.0.13

            - Remove "dataset" `return_type` option.

    Coordinate-based meta-analyses leverage coordinates reported in
    neuroimaging papers to simulate the thresholded statistical maps from the
    original analyses. This generally involves convolving each coordinate with
    a kernel (typically a Gaussian or binary sphere) that may be weighted based
    on some additional measure, such as statistic value or sample size.

    Notes
    -----
    All extra (non-ijk) parameters for a given kernel should be overrideable as
    parameters to __init__, so we can access them with get_params() and also
    apply them to datasets with missing data.
    """

    def _infer_names(self, **kwargs):
        """Determine filename pattern and image type.

        The parameters used to construct the filenames come from the transformer's
        parameters (attributes saved in ``__init__()``).

        Parameters
        ----------
        **kwargs
            Additional key/value pairs to incorporate into the image name.
            A common example is the hash for the target template's affine.

        Attributes
        ----------
        filename_pattern : str
            Filename pattern for images.
        image_type : str
            Name of the corresponding column in the Dataset.images DataFrame.
        """
        params = self.get_params()
        params = dict(**params, **kwargs)

        # Determine names for kernel-specific files
        keys = sorted(params.keys())
        param_str = "_".join(f"{k}-{str(params[k])}" for k in keys)
        self.filename_pattern = (
            f"study-[[id]]_{param_str}_{self.__class__.__name__}.nii.gz".replace(
                "[[", "{"
            ).replace("]]", "}")
        )
        self.image_type = f"{param_str}_{self.__class__.__name__}"

    def transform(self, dataset, masker=None, return_type="image"):
        """Generate modeled activation images for each Contrast in dataset.

        Parameters
        ----------
        dataset : :obj:`~nimare.dataset.Dataset` or :obj:`pandas.DataFrame`
            Dataset for which to make images. Can be a DataFrame if necessary.
        masker : img_like or None, optional
            Mask to apply to MA maps. Required if ``dataset`` is a DataFrame.
            If None (and ``dataset`` is a Dataset), the Dataset's masker attribute will be used.
            Default is None.
        return_type : {'sparse', 'array', 'image'}, optional
            Whether to return a sparse matrix ('sparse'), a numpy array ('array'),
            or a list of niimgs ('image').
            Default is 'image'.

        Returns
        -------
        imgs : (C x V) :class:`numpy.ndarray` or :obj:`list` of :class:`nibabel.Nifti1Image` \
               or :class:`~nimare.dataset.Dataset`
            If return_type is 'sparse', a 4D sparse array (E x S), where E is
            the number of unique experiments, and the remaining 3 dimensions are
            equal to `shape` of the images.
            If return_type is 'array', a 2D numpy array (C x V), where C is
            contrast and V is voxel.
            If return_type is 'image', a list of modeled activation images
            (one for each of the Contrasts in the input dataset).

        Attributes
        ----------
        filename_pattern : str
            Filename pattern for MA maps. If :meth:`_infer_names` is executed.
        image_type : str
            Name of the corresponding column in the Dataset.images DataFrame.
            If :meth:`_infer_names` is executed.
        """
        if return_type not in ("sparse", "array", "image"):
            raise ValueError('Argument "return_type" must be "image", "array", or "sparse".')

        if isinstance(dataset, pd.DataFrame):
            assert (
                masker is not None
            ), "Argument 'masker' must be provided if dataset is a DataFrame."
            mask = masker.mask_img
            coordinates = dataset

            # Calculate IJK. Must assume that the masker is in same space,
            # but has different affine, from original IJK.
            coordinates[["i", "j", "k"]] = mm2vox(dataset[["x", "y", "z"]], mask.affine)
        else:
            masker = dataset.masker if not masker else masker
            mask = masker.mask_img
            coordinates = dataset.coordinates.copy()

            # Calculate IJK
            if not np.array_equal(mask.affine, dataset.masker.mask_img.affine):
                LGR.warning("Mask affine does not match Dataset affine. Assuming same space.")

            coordinates[["i", "j", "k"]] = mm2vox(coordinates[["x", "y", "z"]], mask.affine)

            # Add any metadata the Transformer might need to the coordinates DataFrame
            # This approach is probably inferior to one which uses a _required_inputs attribute
            # (like the MetaEstimators), but it should work just fine as long as individual
            # requirements are written in here.
            if (
                hasattr(self, "sample_size")
                and (self.sample_size is None)
                and ("sample_size" not in coordinates.columns)
            ):
                coordinates = _add_metadata_to_dataframe(
                    dataset,
                    coordinates,
                    metadata_field="sample_sizes",
                    target_column="sample_size",
                    filter_func=np.mean,
                )

        if return_type == "array":
            mask_data = mask.get_fdata().astype(bool)
        elif return_type == "image":
            dtype = type(self.value) if hasattr(self, "value") else float
            mask_data = mask.get_fdata().astype(dtype)

        # Generate the MA maps
        transformed_maps = self._transform(mask, coordinates)

        if return_type == "sparse":
            return transformed_maps[0]

        imgs = []
        # Loop over exp ids since sparse._coo.core.COO is not iterable
        for i_exp, _ in enumerate(transformed_maps[1]):
            kernel_data = transformed_maps[0][i_exp].todense()

            if return_type == "array":
                img = kernel_data[mask_data]
                imgs.append(img)
            elif return_type == "image":
                kernel_data *= mask_data
                img = nib.Nifti1Image(kernel_data, mask.affine)
                imgs.append(img)

        del kernel_data, transformed_maps

        if return_type == "array":
            return np.vstack(imgs)
        elif return_type == "image":
            return imgs

    def _transform(self, mask, coordinates):
        """Apply the kernel's unique transformer.

        Parameters
        ----------
        mask : niimg-like
            Mask image. Should contain binary-like integer data.
        coordinates : pandas.DataFrame
            DataFrame containing IDs and coordinates.
            The DataFrame must have the following columns: "id", "i", "j", "k".
            Additionally, individual kernels may require other columns
            (e.g., "sample_size" for ALE).

        Returns
        -------
        transformed_maps : N-length list of (3D array, str) tuples or (4D array, 1D array) tuple
            Transformed data, containing one element for each study.

            -   Case 1: A kernel that is not an (M)KDAKernel, with no memory limit.
                Each list entry is composed of a 3D array (the MA map) and the study's ID.

            -   Case 2: (M)KDAKernel, with no memory limit.
                There is a length-2 tuple with a 4D numpy array of the shape (N, X, Y, Z),
                containing all of the MA maps, and a numpy array of shape (N,) with the study IDs.
        """
        pass


class ALEKernel(KernelTransformer):
    """Generate ALE modeled activation images from coordinates and sample size.

    By default (if neither ``fwhm`` nor ``sample_size`` is provided), the FWHM of the kernel
    will be determined on a study-wise basis based on the sample sizes available in the input,
    via the method described in :footcite:t:`eickhoff2012activation`.

    .. versionchanged:: 0.0.13

            - Remove "dataset" `return_type` option.

    .. versionchanged:: 0.0.12

        * Remove low-memory option in favor of sparse arrays for kernel transformers.

    Parameters
    ----------
    fwhm : :obj:`float`, optional
        Full-width half-max for Gaussian kernel, if you want to have a
        constant kernel across Contrasts. Mutually exclusive with
        ``sample_size``.
    sample_size : :obj:`int`, optional
        Sample size, used to derive FWHM for Gaussian kernel based on
        formulae from Eickhoff et al. (2012). This sample size overwrites
        the Contrast-specific sample sizes in the dataset, in order to hold
        kernel constant across Contrasts. Mutually exclusive with ``fwhm``.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, fwhm=None, sample_size=None):
        if fwhm is not None and sample_size is not None:
            raise ValueError('Only one of "fwhm" and "sample_size" may be provided.')
        self.fwhm = fwhm
        self.sample_size = sample_size

    def _transform(self, mask, coordinates):
        ijks = coordinates[["i", "j", "k"]].values
        exp_idx = coordinates["id"].values

        use_dict = True
        kernel = None
        if self.sample_size is not None:
            sample_sizes = self.sample_size
            use_dict = False
        elif self.fwhm is None:
            sample_sizes = coordinates["sample_size"].values
        else:
            sample_sizes = None

        if self.fwhm is not None:
            assert np.isfinite(self.fwhm), "FWHM must be finite number"
            _, kernel = get_ale_kernel(mask, fwhm=self.fwhm)
            use_dict = False

        transformed = compute_ale_ma(
            mask,
            ijks,
            kernel=kernel,
            exp_idx=exp_idx,
            sample_sizes=sample_sizes,
            use_dict=use_dict,
        )

        exp_ids = np.unique(exp_idx)
        return transformed, exp_ids


class KDAKernel(KernelTransformer):
    """Generate KDA modeled activation images from coordinates.

    .. versionchanged:: 0.0.13

            - Remove "dataset" `return_type` option.

    .. versionchanged:: 0.0.12

        * Remove low-memory option in favor of sparse arrays for kernel transformers.

    Parameters
    ----------
    r : :obj:`int`, optional
        Sphere radius, in mm.
    value : :obj:`int`, optional
        Value for sphere.
    """

    _sum_overlap = True

    def __init__(self, r=10, value=1):
        self.r = float(r)
        self.value = value

    def _transform(self, mask, coordinates):

        ijks = coordinates[["i", "j", "k"]].values
        exp_idx = coordinates["id"].values
        transformed = compute_kda_ma(
            mask,
            ijks,
            self.r,
            self.value,
            exp_idx,
            sum_overlap=self._sum_overlap,
        )
        exp_ids = np.unique(exp_idx)
        return transformed, exp_ids


class MKDAKernel(KDAKernel):
    """Generate MKDA modeled activation images from coordinates.

    .. versionchanged:: 0.0.13

            - Remove "dataset" `return_type` option.

    .. versionchanged:: 0.0.12

        * Remove low-memory option in favor of sparse arrays for kernel transformers.

    Parameters
    ----------
    r : :obj:`int`, optional
        Sphere radius, in mm.
    value : :obj:`int`, optional
        Value for sphere.
    """

    _sum_overlap = False
