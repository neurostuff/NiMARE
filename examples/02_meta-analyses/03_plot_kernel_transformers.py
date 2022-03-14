"""

.. _metas_ma_maps:

===========================
KernelTransformers and CBMA
===========================

:py:class:`~nimare.meta.kernel.KernelTransformer`s are tools for converting
individual studies' coordinates into images.

For coordinate-based meta-analyses, individual studies' statistical maps are
mimicked by generating "modeled activation" (MA) maps from the coordinates.
These MA maps are used in the CBMA algorithms, although the specific method
used to generate the MA maps differs by algorithm.

This example provides an introduction to the ``KernelTransformer`` class and
a tour of available types.
"""
# sphinx_gallery_thumbnail_number = 2
###############################################################################
# Start with the necessary imports
# -----------------------------------------------------------------------------
import os

import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map

###############################################################################
# Load Dataset
# -----------------------------------------------------------------------------
from nimare.dataset import Dataset
from nimare.utils import get_resource_path

dset_file = os.path.join(get_resource_path(), "nidm_pain_dset.json")
dset = Dataset(dset_file)

###############################################################################
# Kernels ingest Datasets and can produce a few types of outputs
# -----------------------------------------------------------------------------
from nimare.meta.kernel import MKDAKernel

kernel = MKDAKernel()
image_output = kernel.transform(dset, return_type="image")
print(type(image_output))
print(type(image_output[0]))

###############################################################################

array_output = kernel.transform(dset, return_type="array")
print(type(array_output))
print(array_output.shape)

###############################################################################
# There is also an option to return an updated Dataset, with the MA maps saved
# as nifti files and references in the Dataset's images attribute.
# However, this will only work if the Dataset has a location set for its
# images.
try:
    dataset_output = kernel.transform(dset, return_type="dataset")
except ValueError as error:
    print(error)

###############################################################################
# Each kernel can accept certain parameters that control behavior
# -----------------------------------------------------------------------------
# You can see what options are available via the API documentation or through
# the help string.
help(MKDAKernel)

###############################################################################
# For example, :class:`~nimare.meta.kernel.MKDAKernel` kernel accepts an `r`
# argument to control the radius of the kernel.
kernel = MKDAKernel(r=2)
mkda_r02 = kernel.transform(dset, return_type="image")
kernel = MKDAKernel(r=6)
mkda_r06 = kernel.transform(dset, return_type="image")
kernel = MKDAKernel(r=10)
mkda_r10 = kernel.transform(dset, return_type="image")
kernel = MKDAKernel(r=14)
mkda_r14 = kernel.transform(dset, return_type="image")

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
plot_stat_map(
    mkda_r02[2],
    cut_coords=[-2, -10, -4],
    title="r=2mm",
    vmax=2,
    axes=axes[0, 0],
    draw_cross=False,
)
plot_stat_map(
    mkda_r06[2],
    cut_coords=[-2, -10, -4],
    title="r=6mm",
    vmax=2,
    axes=axes[0, 1],
    draw_cross=False,
)
plot_stat_map(
    mkda_r10[2],
    cut_coords=[-2, -10, -4],
    title="r=10mm",
    vmax=2,
    axes=axes[1, 0],
    draw_cross=False,
)
plot_stat_map(
    mkda_r14[2],
    cut_coords=[-2, -10, -4],
    title="r=14mm",
    vmax=2,
    axes=axes[1, 1],
    draw_cross=False,
)
fig.show()

###############################################################################
# There are several kernels available
# -----------------------------------------------------------------------------
# :class:`~nimare.meta.kernel.MKDAKernel` convolves coordinates with a
# sphere and takes the union across voxels.
kernel = MKDAKernel(r=10)
mkda_res = kernel.transform(dset, return_type="image")

plot_stat_map(
    mkda_res[2],
    cut_coords=[-2, -10, -4],
    title="MKDA",
    draw_cross=False,
)

###############################################################################
# :class:`~nimare.meta.kernel.KDAKernel` convolves coordinates with a
# sphere as well, but takes the *sum* across voxels.
from nimare.meta.kernel import KDAKernel

kernel = KDAKernel(r=10)
kda_res = kernel.transform(dset, return_type="image")

plot_stat_map(
    kda_res[2],
    cut_coords=[-2, -10, -4],
    title="KDA",
    draw_cross=False,
)

###############################################################################
# :class:`~nimare.meta.kernel.ALEKernel` convolves coordinates with a 3D
# Gaussian, for which the FWHM is determined by the sample size of each study.
from nimare.meta.kernel import ALEKernel

kernel = ALEKernel(sample_size=20)
ale_res = kernel.transform(dset, return_type="image")

plot_stat_map(
    ale_res[2],
    cut_coords=[-2, -10, -4],
    title="ALE",
    draw_cross=False,
)
