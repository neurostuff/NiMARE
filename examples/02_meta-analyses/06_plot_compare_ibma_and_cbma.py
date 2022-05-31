"""

.. _metas_cbma_vs_ibma:

================================================
Compare image and coordinate based meta-analyses
================================================

Run IBMAs and CBMAs on a toy dataset, then compare the results qualitatively.

Collection of NIDM-Results packs downloaded from Neurovault collection 1425,
uploaded by Dr. Camille Maumet.
"""
import os

import pandas as pd
from nilearn.plotting import plot_stat_map

from nimare.dataset import Dataset
from nimare.extract import download_nidm_pain
from nimare.meta.cbma import ALE
from nimare.meta.ibma import DerSimonianLaird
from nimare.transforms import ImagesToCoordinates, ImageTransformer
from nimare.utils import get_resource_path

###############################################################################
# Download data
# -----------------------------------------------------------------------------
dset_dir = download_nidm_pain()

###############################################################################
# Load Dataset
# -----------------------------------------------------------------------------
dset_file = os.path.join(get_resource_path(), "nidm_pain_dset.json")
dset = Dataset(dset_file)
dset.update_path(dset_dir)

# Calculate missing statistical images from the available stats.
xformer = ImageTransformer(target=["varcope"])
dset = xformer.transform(dset)

# create coordinates from statistical maps
coord_gen = ImagesToCoordinates(merge_strategy="fill")
dset = coord_gen.transform(dset)

###############################################################################
# ALE (CBMA)
# -----------------------------------------------------------------------------
meta_cbma = ALE()
cbma_results = meta_cbma.fit(dset)
plot_stat_map(
    cbma_results.get_map("z"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
)

###############################################################################
# DerSimonian-Laird (IBMA)
# -----------------------------------------------------------------------------
# We must resample the image data to the same MNI template as the Dataset.
meta_ibma = DerSimonianLaird(resample=True)
ibma_results = meta_ibma.fit(dset)
plot_stat_map(
    ibma_results.get_map("z"),
    cut_coords=[0, 0, -8],
    draw_cross=False,
    cmap="RdBu_r",
)

###############################################################################
# Compare CBMA and IBMA Z-maps
# -----------------------------------------------------------------------------
stat_df = pd.DataFrame(
    {
        "CBMA": cbma_results.get_map("z", return_type="array"),
        "IBMA": ibma_results.get_map("z", return_type="array"),
    }
)
print(stat_df.corr())
