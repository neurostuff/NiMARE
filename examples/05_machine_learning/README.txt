.. _examples-machine-learning:

Machine learning with Studysets
-------------------------------

NiMARE's machine-learning tools convert :class:`~nimare.nimads.Studyset`
objects into scikit-learn-compatible feature sets. The example below builds
modeled activation features, splits analyses by study so that none of a study's
analyses land on both sides, reduces the voxelwise features, and fits a
downstream estimator.
