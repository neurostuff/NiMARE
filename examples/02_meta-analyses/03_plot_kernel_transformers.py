"""

.. _metas_kernels:

===========================
KernelTransformers and CBMA
===========================

``KernelTransformer`` classes are tools for converting individual studies'
coordinates into images.

For coordinate-based meta-analyses, individual studies' statistical maps are
mimicked by generating "modeled activation" (MA) maps from the coordinates.
These MA maps are used in the CBMA algorithms, although the specific method
used to generate the MA maps differs by algorithm.

This example provides an introduction to the ``KernelTransformer`` class and
a tour of available types.
"""

# sphinx_gallery_thumbnail_number = 2
import os

import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map

###############################################################################
# Load Studyset
# -----------------------------------------------------------------------------
from nimare.nimads import Studyset
from nimare.utils import get_resource_path

studyset_file = os.path.join(get_resource_path(), "nidm_pain_studyset.json")
studyset = Studyset(studyset_file, target="mni152_2mm")

# First, let us reduce this Studyset to only two studies.
source = studyset.to_dict()
source["studies"] = source["studies"][2:4]
studyset = Studyset(source, target="mni152_2mm")

###############################################################################
# Kernels ingest Studysets and can produce a few types of outputs
# -----------------------------------------------------------------------------
from nimare.meta.kernel import MKDAKernel

# First, the kernel should be initialized with any parameters.
kernel = MKDAKernel()

# Then, the ``transform`` method takes in the Studyset and produces the MA maps.
output = kernel.transform(studyset)

###############################################################################
# ``return_type="image"`` returns a list of 3D niimg objects.
#
# This is the default option.
image_output = kernel.transform(studyset, return_type="image")
print(type(image_output))
print(type(image_output[0]))
print(image_output[0].shape)

###############################################################################
# ``return_type="array"`` returns a 2D numpy array
array_output = kernel.transform(studyset, return_type="array")
print(type(array_output))
print(array_output.shape)

###############################################################################
# ``return_type="summary_array"`` returns a 1D summary map across studies.
summary_output = kernel.transform(studyset, return_type="summary_array")
print(type(summary_output))
print(summary_output.shape)

###############################################################################
# Each kernel can accept certain parameters that control behavior
# -----------------------------------------------------------------------------
# You can see what options are available via the API documentation or through
# the help string.
help(MKDAKernel)

###############################################################################
# For example, :class:`~nimare.meta.kernel.MKDAKernel` kernel accepts an ``r``
# argument to control the radius of the kernel.
RADIUS_VALUES = [4, 8, 12]
fig, axes = plt.subplots(ncols=3, figsize=(20, 10))

for i, radius in enumerate(RADIUS_VALUES):
    kernel = MKDAKernel(r=radius)
    ma_maps = kernel.transform(studyset, return_type="image")

    plot_stat_map(
        ma_maps[0],
        display_mode="z",
        cut_coords=[-2],
        title=f"r={radius}mm",
        axes=axes[i],
        draw_cross=False,
        annotate=False,
        colorbar=False,
        cmap="RdBu_r",
        symmetric_cbar=True,
    )

###############################################################################
# There are several kernels available
# -----------------------------------------------------------------------------
# :class:`~nimare.meta.kernel.MKDAKernel` convolves coordinates with a
# sphere and takes the union across voxels.
kernel = MKDAKernel(r=10)
ma_maps = kernel.transform(studyset, return_type="image")

plot_stat_map(
    ma_maps[0],
    cut_coords=[-2, -10, -4],
    title="MKDA",
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
)

###############################################################################
# :class:`~nimare.meta.kernel.KDAKernel` convolves coordinates with a
# sphere as well, but takes the *sum* across voxels.
from nimare.meta.kernel import KDAKernel

kernel = KDAKernel(r=10)
ma_maps = kernel.transform(studyset, return_type="image")

plot_stat_map(
    ma_maps[0],
    cut_coords=[-2, -10, -4],
    title="KDA",
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
)

###############################################################################
# :class:`~nimare.meta.kernel.ALEKernel` convolves coordinates with a 3D
# Gaussian, for which the FWHM is determined by the sample size of each study.
# This sample size will be inferred automatically, if that information is
# available in the Studyset, or it can be set as a constant value across all
# studies in the Studyset with the ``sample_size`` argument.
from nimare.meta.kernel import ALEKernel

kernel = ALEKernel(sample_size=20)
ma_maps = kernel.transform(studyset, return_type="image")

plot_stat_map(
    ma_maps[0],
    cut_coords=[-2, -10, -4],
    title="ALE",
    draw_cross=False,
    cmap="RdBu_r",
    symmetric_cbar=True,
)
