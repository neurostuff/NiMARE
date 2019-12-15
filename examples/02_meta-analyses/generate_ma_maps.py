# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _metas1:

========================================================
 Generate modeled activation maps
========================================================

"""
###############################################################################
# Start with the necessary imports
# --------------------------------
import os

import numpy as np
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map

import nimare
from nimare.tests.utils import get_test_data_path

###############################################################################
# Load Dataset
# --------------------------------------------------
dset_file = os.path.join(get_test_data_path(), 'nidm_pain_dset.json')
dset = nimare.dataset.Dataset(dset_file)

###############################################################################
# MKDA kernel maps
# --------------------------------------------------
kernel = nimare.meta.cbma.MKDAKernel(r=8)
mkda_r08 = kernel.transform(dset)
kernel = nimare.meta.cbma.MKDAKernel(r=9)
mkda_r09 = kernel.transform(dset)
kernel = nimare.meta.cbma.MKDAKernel(r=10)
mkda_r10 = kernel.transform(dset)
kernel = nimare.meta.cbma.MKDAKernel(r=11)
mkda_r11 = kernel.transform(dset)

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 17.5))
plot_stat_map(mkda_r08[2], cut_coords=[-2, -10, -4],
              title='r=8mm', vmax=2, axes=axes[0],
              draw_cross=False)
plot_stat_map(mkda_r09[2], cut_coords=[-2, -10, -4],
              title='r=9mm', vmax=2, axes=axes[1],
              draw_cross=False)
plot_stat_map(mkda_r10[2], cut_coords=[-2, -10, -4],
              title='r=10mm', vmax=2, axes=axes[2],
              draw_cross=False)
plot_stat_map(mkda_r11[2], cut_coords=[-2, -10, -4],
              title='r=11mm', vmax=2, axes=axes[3],
              draw_cross=False)
fig.show()

###############################################################################
# Show different kernel types together
# --------------------------------------------------
kernel = nimare.meta.cbma.MKDAKernel(r=10)
mkda_res = kernel.transform(dset)
kernel = nimare.meta.cbma.KDAKernel(r=10)
kda_res = kernel.transform(dset)
kernel = nimare.meta.cbma.ALEKernel(n=20)
ale_res = kernel.transform(dset)
max_conv = np.max(kda_res[2].get_data())
plot_stat_map(ale_res[2], cut_coords=[-2, -10, -4], title='ALE')
plot_stat_map(mkda_res[2], cut_coords=[-2, -10, -4], title='MKDA', vmax=max_conv)
plot_stat_map(kda_res[2], cut_coords=[-2, -10, -4], title='KDA', vmax=max_conv)
