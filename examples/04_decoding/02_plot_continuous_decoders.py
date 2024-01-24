"""

.. _decode_continuous:

============================
Continuous functional decoding
============================

Perform meta-analytic functional decoding on statistical images.

We can use the methods in ``nimare.decode.continuous`` to apply functional
characterization analysis to statistical images.
"""
import os

import nibabel as nib
import requests
from nilearn.plotting import plot_stat_map

from nimare.decode.continuous import CorrelationDecoder
from nimare.extract import fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset
from nimare.meta.cbma.mkda import MKDAChi2

###############################################################################
# Download Neurosynth with Topics
# -----------------------------------------------------------------------------
# Neurosynth's data files are stored at https://github.com/neurosynth/neurosynth-data.
out_dir = os.path.abspath("../example_data/")
os.makedirs(out_dir, exist_ok=True)

files = fetch_neurosynth(
    data_dir=out_dir,
    version="7",
    overwrite=False,
    source="abstract",
    vocab="LDA50",
)
neurosynth_db = files[0]

###############################################################################
# Convert Neurosynth database to NiMARE dataset file
# -----------------------------------------------------------------------------
neurosynth_dset = convert_neurosynth_to_dataset(
    coordinates_file=neurosynth_db["coordinates"],
    metadata_file=neurosynth_db["metadata"],
    annotations_files=neurosynth_db["features"],
)

###############################################################################
# Download a continuous map for decoding from NeuroVault
# -----------------------------------------------------------------------------
image_id = "3158"  # tfMRI motor left hand from the HCP dataset
image_info_url = f"https://neurovault.org/api/images/{image_id}/"
image_info = requests.get(image_info_url).json()

image_url = image_info["file"]
image_filename = os.path.basename(image_url)
image_path = os.path.join(out_dir, f"{image_id}_{image_filename}")

response = requests.get(image_url)
with open(image_path, "wb") as image_file:
    image_file.write(response.content)

continuous_map = nib.load(image_path)

plot_stat_map(
    continuous_map,
    cut_coords=[0, 0, -8],
    draw_cross=False,
)

###############################################################################
#
# .. _neurosynth-topic-decoder-example:
#
# Decode an statistical image using an LDA-Based continuous decoder
# -----------------------------------------------------------------------------

# Train the decoder
decoder = CorrelationDecoder(
    frequency_threshold=0.05,
    meta_estimator=MKDAChi2(),
    feature_group="LDA50_abstract_weight",
    target_image="z_desc-association",
    n_cores=1,
)
decoder.fit(neurosynth_dset)

# Decode the image
corr_df = decoder.transform(continuous_map)
corr_df.sort_values(by="r", ascending=False).head(10)
