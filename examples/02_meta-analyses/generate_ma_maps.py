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
from os.path import dirname, join
import numpy as np
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map

import nimare

###############################################################################
# Load Dataset
# --------------------------------------------------
database_file = join(dirname(nimare.__file__), 'tests/data/nidm_pain_dset.json')
ds = nimare.dataset.Dataset(database_file)

###############################################################################
# MKDA kernel maps
# --------------------------------------------------
kernel = nimare.meta.cbma.MKDAKernel(ds.coordinates, ds.mask)
mkda_r08 = kernel.transform(ids=ds.ids, r=8)
mkda_r09 = kernel.transform(ids=ds.ids, r=9)
mkda_r10 = kernel.transform(ids=ds.ids, r=10)
mkda_r11 = kernel.transform(ids=ds.ids, r=11)

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
kernel = nimare.meta.cbma.MKDAKernel(ds.coordinates, ds.mask)
mkda_res = kernel.transform(ids=ds.ids, r=10)
kernel = nimare.meta.cbma.KDAKernel(ds.coordinates, ds.mask)
kda_res = kernel.transform(ids=ds.ids, r=10)
kernel = nimare.meta.cbma.ALEKernel(ds.coordinates, ds.mask)
ale_res = kernel.transform(ids=ds.ids, n=20)
max_conv = np.max(kda_res[2].get_data())
plot_stat_map(ale_res[2], cut_coords=[-2, -10, -4], title='ALE')
plot_stat_map(mkda_res[2], cut_coords=[-2, -10, -4], title='MKDA', vmax=max_conv)
plot_stat_map(kda_res[2], cut_coords=[-2, -10, -4], title='KDA', vmax=max_conv)
