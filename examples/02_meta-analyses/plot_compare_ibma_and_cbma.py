# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _metas7:

===================================================================
Compare image and coordinate based meta-analyses on 21 pain studies
===================================================================

Collection of NIDM-Results packs downloaded from Neurovault collection 1425,
uploaded by Dr. Camille Maumet.

"""
import os

import pandas as pd
from nilearn.plotting import plot_stat_map

import nimare
from nimare.transforms import CoordinateGenerator, transform_images
from nimare.meta.ibma import DerSimonianLaird
from nimare.meta.cbma import ALE
from nimare.tests.utils import get_test_data_path

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
dset.images = transform_images(
    dset.images, target="z", masker=dset.masker, metadata_df=dset.metadata
)
dset.images = nimare.transforms.transform_images(
    dset.images, target="varcope", masker=dset.masker, metadata_df=dset.metadata
)

# create coordinates from statistical maps
coord_gen = CoordinateGenerator(merge_strategy="replace")
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
meta_ibma = DerSimonianLaird()
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
