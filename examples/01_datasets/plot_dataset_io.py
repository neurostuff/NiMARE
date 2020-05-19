# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _datasets1:

========================================================
 Load and work with a Dataset
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
# Datasets are stored as jsons or pkl[.gz] files
# --------------------------------------------------
# Json files are used to create Datasets, while generated Datasets are saved
# to, and loaded from, pkl[.gz] files.
# We use jsons because they are easy to edit, and thus build by hand, if
# necessary.
# We then store the generated Datasets as pkl.gz files because an initialized
# Dataset is no longer a dictionary.

# Let's start by downloading a dataset
dset_dir = nimare.extract.download_nidm_pain()

# Now we can load and save the Dataset object
dset_file = os.path.join(get_test_data_path(), 'nidm_pain_dset.json')
dset = nimare.dataset.Dataset(dset_file, target='mni152_2mm', mask=None)
dset.save('pain_dset.pkl')
dset = nimare.dataset.Dataset.load('pain_dset.pkl')
os.remove('pain_dset.pkl')  # cleanup

###############################################################################
# Much of the data in Datasets is stored as DataFrames
# --------------------------------------------------------------
# The five DataFrames in Dataset are "coordinates" (reported peaks),
# "images" (statistical maps), "metadata", "texts", and "annotations" (labels).
print('Coordinates:')
print(dset.coordinates.head())

print('Images:')
print(dset.images.head())

print('Metadata:')
print(dset.metadata.head())

print('Texts:')
print(dset.texts.head())

print('Annotations:')
print(dset.annotations.head())

###############################################################################
# There are a handful of other important Dataset attributes
# --------------------------------------------------------------
print('Study identifiers: {}'.format(dset.ids))
print('Masker: {}'.format(dset.masker))
print('Template space: {}'.format(dset.space))

###############################################################################
# Statistical images are not stored internally
# --------------------------------------------------------------
# Images are not stored within the Dataset.
# Instead, relative paths to image files are retained in the Dataset.images
# attribute.
# When loading a Dataset, you will likely need to specify the path to the images.
# To do this, you can use Dataset.update_path.
dset.update_path(dset_dir)
print(dset.images.head())

###############################################################################
# Datasets support many search methods
# --------------------------------------------------------------
# There are ``get_[X]`` and ``get_studies_by_[X]`` methods for a range of
# possible search criteria.
# The ``get_[X]`` methods allow you to search for specific metadata, while the
# ``get_studies_by_[X]`` methods let you search for study identifiers within
# the Dataset matching criteria.
#
# Note that the ``get_[X]`` methods return a value for every study in the Dataset
# by default, and for every requested study if the ``ids`` argument is provided.
# If a study does not have the data requested, the returned list will have
# ``None`` for that study.
z_images = dset.get_images(imtype='z')
z_images = [str(z) for z in z_images]
print('\n'.join(z_images))

###############################################################################
# Datasets can also search for studies matching criteria
# --------------------------------------------------------------
# ``get_studies_by_[X]`` methods return a list of study identifiers matching
# the criteria, such as reporting a peak coordinate near a search coordinate.
sel_studies = dset.get_studies_by_coordinate(xyz=[[0, 0, 0]], r=20)
print('\n'.join(sel_studies))

###############################################################################
# Datasets are meant to be mostly immutable
# --------------------------------------------------------------
# While some elements of Datasets are designed to be changeable, like the paths
# to image files, most elements are not.
# NiMARE Estimators operate on Datasets and return *new*, updated Datasets.
# If you want to reduce a Dataset based on a subset of the studies in the
# Dataset, you need to use ``Dataset.slice()``.
sub_dset = dset.slice(ids=sel_studies)
print('\n'.join(sub_dset.ids))
