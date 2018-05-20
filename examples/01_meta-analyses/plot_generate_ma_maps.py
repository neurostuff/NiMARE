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
# KDA
# --------------------------------
kernel = nimare.meta.cbma.KDAKernel(ds.coordinates, ds.mask)
kda_res = kernel.transform(ids=ds.ids, r=10)
max_conv = np.max(kda_res[2].get_data())
plot_stat_map(kda_res[2], cut_coords=[-2, -10, -4], title='KDA', vmax=max_conv)

###############################################################################
# MKDA
# --------------------------------
kernel = nimare.meta.cbma.MKDAKernel(ds.coordinates, ds.mask)
mkda_res = kernel.transform(ids=ds.ids, r=10)
plot_stat_map(mkda_res[2], cut_coords=[-2, -10, -4], title='MKDA', vmax=max_conv)

###############################################################################
# ALE
# --------------------------------
kernel = nimare.meta.cbma.ALEKernel(ds.coordinates, ds.mask)
ale_res = kernel.transform(ids=ds.ids, n=20)
plot_stat_map(ale_res[2], cut_coords=[-2, -10, -4], title='ALE')
