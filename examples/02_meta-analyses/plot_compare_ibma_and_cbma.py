# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _metas7:

===================================================
07 Compare image and coordinate based meta-analyses
===================================================

Run IBMAs and CBMAs on a toy dataset, then compare the results qualitatively.

Collection of NIDM-Results packs downloaded from Neurovault collection 1425,
uploaded by Dr. Camille Maumet.
"""
import os

import pandas as pd
from nilearn.plotting import plot_stat_map

import nimare
from nimare.meta.cbma import ALE
from nimare.meta.ibma import DerSimonianLaird
from nimare.tests.utils import get_test_data_path
from nimare.transforms import ImagesToCoordinates, ImageTransformer

###############################################################################
# Download data
# --------------------------------
dset_dir = nimare.extract.download_nidm_pain()

###############################################################################
# Load Dataset
# --------------------------------------------------
dset_file = os.path.join(get_test_data_path(), "nidm_pain_dset.json")
dset = nimare.dataset.Dataset(dset_file)
dset.update_path(dset_dir)

# Calculate missing statistical images from the available stats.
xformer = ImageTransformer(target=["z", "varcope"])
dset = xformer.transform(dset)

# create coordinates from statistical maps
coord_gen = ImagesToCoordinates(merge_strategy="replace")
dset = coord_gen.transform(dset)

###############################################################################
# ALE (CBMA)
# -----------------------------------------------------------------------------
meta_cbma = ALE()
meta_cbma.fit(dset)
plot_stat_map(
    meta_cbma.results.get_map("z"), cut_coords=[0, 0, -8], draw_cross=False, cmap="RdBu_r"
)

###############################################################################
# DerSimonian-Laird (IBMA)
# -----------------------------------------------------------------------------
# We must resample the image data to the same MNI template as the Dataset.
meta_ibma = DerSimonianLaird(resample=True)
meta_ibma.fit(dset)
plot_stat_map(
    meta_ibma.results.get_map("z"), cut_coords=[0, 0, -8], draw_cross=False, cmap="RdBu_r"
)

###############################################################################
# Compare CBMA and IBMA Z-maps
# -----------------------------------------------------------------------------
stat_df = pd.DataFrame(
    {
        "CBMA": meta_cbma.results.get_map("z", return_type="array"),
        "IBMA": meta_ibma.results.get_map("z", return_type="array").squeeze(),
    }
)
print(stat_df.corr())
