# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""

.. _metas1:

========================================================
 Run an ALE meta-analysis with the Neurosynth database
========================================================

Build a small sample of studies about "pain" from Neurosynth and run an
ALE meta-analysis on those studies.

..note::
    This will likely change as we work to shift database querying to a remote
    database, rather than handling it locally with NiMARE.

"""
###############################################################################
# Start with the necessary imports
# --------------------------------
import os.path as op
from os import mkdir
from neurosynth.base.dataset import download
from nilearn import plotting

from nimare.io import convert_neurosynth_to_dataset
from nimare.dataset import Dataset
from nimare.meta.cbma import ALE

###############################################################################
# Load the Neurosynth database
# --------------------------------
out_dir = op.abspath('../example_data/')
if not op.isdir(out_dir):
    mkdir(out_dir)

if not op.isfile(op.join(out_dir, 'database.txt')):
    download(out_dir, unpack=True)

###############################################################################
# Convert Neurosynth database to NiMARE dataset file
# --------------------------------------------------
gz_file = op.join(out_dir, 'neurosynth_dataset.pkl.gz')
if not op.isfile(gz_file):
    dset = convert_neurosynth_to_dataset(op.join(out_dir, 'database.txt'),
                                         op.join(out_dir, 'features.txt'))
    dset.save(gz_file)
else:
    dset = Dataset.load(gz_file)

###############################################################################
# Search Neurosynth for pain studies
# --------------------------------------------------
pain_studies = dset.get_studies_by_label('pain', frequency_threshold=0.001)
print('{0} studies found.'.format(len(pain_studies)))
# Reduce to only 50 for expediency
pain_studies = pain_studies[:50]
pain_dset = dset.slice(pain_studies)

###############################################################################
# Run ALE
# --------------------------------------------------
ale = ALE(dset)
ale.fit(pain_studies, n_iters=100)

###############################################################################
# Show results
# --------------------------------------------------
fig = plotting.plot_stat_map(ale.results.images['cfwe'])
