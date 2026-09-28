.. _examples-machine-learning:

Machine learning with Studysets
-------------------------------

NiMARE's machine-learning tools convert :class:`~nimare.nimads.Studyset`
objects into scikit-learn-compatible bunches. The example below builds
modeled activation features, splits analyses so each study stays on one side,
reduces the voxelwise features, and fits a downstream estimator.
