"""

.. _datasets_object:

=========================
The NiMARE Dataset object
=========================

This is a brief walkthrough of the :class:`~nimare.dataset.Dataset` class and its methods.
"""

###############################################################################
# Start with the necessary imports
# -----------------------------------------------------------------------------
import os

from nimare.dataset import Dataset
from nimare.extract import download_nidm_pain
from nimare.transforms import ImageTransformer
from nimare.utils import get_resource_path

###############################################################################
# Datasets are stored as json or pkl[.gz] files
# -----------------------------------------------------------------------------
# Json files are used to create Datasets, while generated Datasets are saved
# to, and loaded from, pkl[.gz] files.
# We use jsons because they are easy to edit, and thus build by hand, if
# necessary.
# We then store the generated Datasets as pkl.gz files because an initialized
# Dataset is no longer a dictionary.

# Let's start by downloading a dataset
dset_dir = download_nidm_pain()

# Now we can load and save the Dataset object
dset_file = os.path.join(get_resource_path(), "nidm_pain_dset.json")
dset = Dataset(dset_file, target="mni152_2mm", mask=None)
dset.save("pain_dset.pkl")
dset = Dataset.load("pain_dset.pkl")
os.remove("pain_dset.pkl")  # cleanup

###############################################################################
# Much of the data in Datasets is stored as DataFrames
# -----------------------------------------------------------------------------
# The five DataFrames in Dataset are "coordinates" (reported peaks),
# "images" (statistical maps), "metadata", "texts", and "annotations" (labels).

###############################################################################
# ``Dataset.annotations`` contains labels describing studies
# `````````````````````````````````````````````````````````````````````````````
# Columns include the standard identifiers and any labels.
# The labels may be grouped together based on label source, in which case they
# should be prefixed with some string followed by two underscores.
dset.annotations.head()

###############################################################################
# ``Dataset.coordinates`` contains reported peaks
# `````````````````````````````````````````````````````````````````````````````
# Columns include the standard identifiers, as well as mm coordinates (x, y, z)
# and voxel indices (i, j, k) specific to the Dataset's masker's space.
dset.coordinates.head()

###############################################################################
# ``Dataset.images`` contains images from studies
# `````````````````````````````````````````````````````````````````````````````
# Columns include the standard identifiers, as well as paths to images grouped
# by image type (e.g., z, beta, t).

# Here we'll only show a subset of these image types to fit in the window.
columns_to_show = ["id", "study_id", "contrast_id", "beta__relative", "z__relative"]
dset.images[columns_to_show].head()

###############################################################################
# ``Dataset.metadata`` contains metadata describing studies
# `````````````````````````````````````````````````````````````````````````````
# Columns include the standard identifiers, as well as one column for each
# metadata field.
dset.metadata.head()

###############################################################################
# ``Dataset.texts`` contains texts associated with studies
# `````````````````````````````````````````````````````````````````````````````
# Columns include the standard identifiers, as well as one for each text type.
dset.texts.head()

###############################################################################
# There are a handful of other important Dataset attributes
# -----------------------------------------------------------------------------

###############################################################################
# ``Dataset.ids`` contains study identifiers
dset.ids

###############################################################################
# ``Dataset.masker`` is a nilearn Masker object
dset.masker

###############################################################################
# ``Dataset.space`` is a string
print(f"Template space: {dset.space}")

###############################################################################
# Statistical images are not stored internally
# -----------------------------------------------------------------------------
# Images are not stored within the Dataset.
# Instead, relative paths to image files are retained in the Dataset.images
# attribute.
# When loading a Dataset, you will likely need to specify the path to the images.
# To do this, you can use :func:`~nimare.dataset.Dataset.update_path`.
dset.update_path(dset_dir)
columns_to_show = ["id", "study_id", "contrast_id", "beta", "beta__relative"]
dset.images[columns_to_show].head()

###############################################################################
# Images can also be calculated based on available files
# `````````````````````````````````````````````````````````````````````````````
# When some images are available, but others are not, sometimes required images
# can be calculated from the available ones.
#
# For example, ``varcope = t / beta``, so if you have t-statistic images and
# beta images, you can also calculate varcope (variance) images.
#
# We use the :mod:`~nimare.transforms` module to perform these transformations
# (especially :class:`~nimare.transforms.ImageTransformer`)
varcope_transformer = ImageTransformer(target="varcope")
dset = varcope_transformer.transform(dset)
dset.images[["id", "varcope"]].head()

###############################################################################
# Datasets support many search methods
# -----------------------------------------------------------------------------
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
z_images = dset.get_images(imtype="z")
z_images = [str(z) for z in z_images]
print("\n".join(z_images))

###############################################################################
# Let's try to fill in missing z images
# `````````````````````````````````````````````````````````````````````````````
z_transformer = ImageTransformer(target="z")
dset = z_transformer.transform(dset)
z_images = dset.get_images(imtype="z")
z_images = [str(z) for z in z_images]
print("\n".join(z_images))

###############################################################################
# Datasets can also search for studies matching criteria
# -----------------------------------------------------------------------------
# ``get_studies_by_[X]`` methods return a list of study identifiers matching
# the criteria, such as reporting a peak coordinate near a search coordinate.
sel_studies = dset.get_studies_by_coordinate(xyz=[[0, 0, 0]], r=20)
print("\n".join(sel_studies))

###############################################################################
# Datasets are meant to be mostly immutable
# -----------------------------------------------------------------------------
# While some elements of Datasets are designed to be changeable, like the paths
# to image files, most elements are not.
# NiMARE Estimators operate on Datasets and return *new*, updated Datasets.
# If you want to reduce a Dataset based on a subset of the studies in the
# Dataset, you need to use :meth:`~nimare.dataset.Dataset.slice`.
sub_dset = dset.slice(ids=sel_studies)
print("\n".join(sub_dset.ids))
