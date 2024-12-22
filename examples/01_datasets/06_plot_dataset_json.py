"""

.. _datasets_json:

===============================================
Create a NiMARE Dataset object from a JSON file
===============================================

Here, we present the structure of the JSON/dict input format to create a
:class:`~nimare.dataset.Dataset` class from scratch.
"""

###############################################################################
# Data Structure
# -----------------------------------------------------------------------------
# A JSON file is organized as a nested dictionary structure containing neuroimaging
# study data. Each study may contain one or multiple contrasts. Each contrast could
# have coordinates, associated images, metadata, and text information. The data will
# be mapped to five core DataFrames: annotations, coordinates, images, metadata, and texts.

###############################################################################
# Study Level
# -----------------------------------------------------------------------------
# - Every study is assigned a unique identifier (study_id), which is defined by
#   the user (e.g., "pain_01.nidm").
# - Every study contains a ``contrasts`` dictionary holding all the contrasts for that
#   study. Each contrast is assigned a unique identifier (contrast_id), which is
#   also defined by the user (e.g., "1").
#
# .. code-block:: python
#
#    {
#        "<study_id>": {
#            "contrasts": {
#                "<contrast_id>": {
#                    "annotations": {...},
#                    "coords": {...},
#                    "images": {...},
#                    "metadata": {...},
#                    "text": {...},
#                }
#            }
#        }
#    }

###############################################################################
# Contrast Level
# -----------------------------------------------------------------------------
# Each contrast contains five main dictionaries:

###############################################################################
# 1. **Annotations Dictionary** (`annotations`)
# `````````````````````````````````````````````````````````````````````````````
# - Contains labels and annotations.
# - Optional for studies.
#
# .. code-block:: python
#
#    "annotations": {
#        "label": str,  # Label for the contrast
#        "description": str  # Description of the contrast
#    }

###############################################################################
# 2. **Coordinates Dictionary** (`coords`)
# `````````````````````````````````````````````````````````````````````````````
# - Includes space information and x, y, and z coordinates.
#
# .. code-block:: python
#
#    "coords": {
#        "space": str,  # e.g., "MNI"
#        "x": List[float],  # x-coordinates
#        "y": List[float],  # y-coordinates
#        "z": List[float]   # z-coordinates
#    }


###############################################################################
# 3. **Images Dictionary** (`images`)
# `````````````````````````````````````````````````````````````````````````````
# - Contains paths to statistical maps. Possible keys are "beta", "se", "t", and "z".
#
# .. code-block:: python
#
#    "images": {
#        "beta": str,     # Path to contrast image
#        "se": str,       # Path to standard error image
#        "t": str,        # Path to t-statistic image
#        "z": str         # Path to z-statistic image
#    }

###############################################################################
# 4. **Metadata Dictionary** (`metadata`)
# `````````````````````````````````````````````````````````````````````````````
# - Contains study-specific metadata.
# - Flexible schema for user-defined metadata.
#
# .. code-block:: python
#
#    "metadata": {
#        "sample_sizes": List[int]
#    }

###############################################################################
# 5. **Text Dictionary** (`text`)
# `````````````````````````````````````````````````````````````````````````````
# - Contains study/contrast text information.
# - Contains keys associated with the linked publication.
#
# .. code-block:: python
#
#    "text": {
#        "title": str,    # Study title
#        "keywords": str, # Study keywords
#        "abstract": str, # Study abstract
#        "body": str      # Main study text/content
#    }

###############################################################################
# Example JSON
# -----------------------------------------------------------------------------
# Load the example dataset JSON file

import json
import os

from nimare.utils import get_resource_path

dset_file = os.path.join(get_resource_path(), "nidm_pain_dset.json")

with open(dset_file, "r") as f_obj:
    data = json.load(f_obj)

###############################################################################
# Example of accessing coordinates for a study
study_coords = data["pain_01.nidm"]["contrasts"]["1"]["coords"]
x_coords = study_coords["x"]
y_coords = study_coords["y"]
z_coords = study_coords["z"]
print(x_coords[:5], y_coords[:5], z_coords[:5])

###############################################################################
# Example of accessing image paths
study_images = data["pain_01.nidm"]["contrasts"]["1"]["images"]
beta_image_path = study_images["beta"]
t_stat_path = study_images["t"]
print(beta_image_path, t_stat_path)

###############################################################################
# Example of accessing metadata
study_metadata = data["pain_01.nidm"]["contrasts"]["1"]["metadata"]
sample_size = study_metadata["sample_sizes"][0]
print(sample_size)

###############################################################################
# .. note::
#    Find more information about the Dataset class that can be created from this JSON file
#    in :ref:`datasets_object`.

###############################################################################
# Example JSON Structure
# -----------------------------------------------------------------------------
print(json.dumps(data, indent=4))
