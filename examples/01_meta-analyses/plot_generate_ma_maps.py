"""

.. _meta1:

=========================================
 Generate modeled activation images
=========================================

Generate MA maps using CBMA kernels.

"""
###############################################################################
# Start with the necessary imports
# --------------------------------
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map

import nimare

###############################################################################
# Load data
# --------------------------------
database_file = op.join(nimare.utils.get_resource_path(),
                        'data/nidm_pain_dset_with_subpeaks.json')
db = nimare.dataset.dataset.Database(database_file)
ds = db.get_dataset()

###############################################################################
# Load data
# --------------------------------
kernel = nimare.meta.cbma.MKDAKernel(ds.coordinates, ds.mask)
mkda_r08 = kernel.transform(ids=ds.ids, r=8)
mkda_r09 = kernel.transform(ids=ds.ids, r=9)
mkda_r10 = kernel.transform(ids=ds.ids, r=10)
mkda_r11 = kernel.transform(ids=ds.ids, r=11)

###############################################################################
# Load data
# --------------------------------
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
