"""Benchmark the IBMA estimators, with and without dependence correction."""

import os
import tempfile
from shutil import copyfile

import nimare
from nimare.meta.ibma import (
    DerSimonianLaird,
    Fishers,
    FixedEffectsHedges,
    Hedges,
    PermutedOLS,
    SampleSizeBasedLikelihood,
    Stouffers,
    VarianceBasedLikelihood,
    WeightedLeastSquares,
)
from nimare.tests.utils import get_test_data_path

ESTIMATORS = {
    "fishers": Fishers,
    "stouffers": Stouffers,
    "weighted_least_squares": WeightedLeastSquares,
    "dersimonian_laird": DerSimonianLaird,
    "hedges": Hedges,
    "fixed_effects_hedges": FixedEffectsHedges,
    "sample_size_based_likelihood": SampleSizeBasedLikelihood,
    "variance_based_likelihood": VarianceBasedLikelihood,
    "permuted_ols": PermutedOLS,
}


def _stage_dataset(dset_name, tmpdir):
    """Load a test dataset and copy its images into a temporary directory."""
    dset_file = os.path.join(get_test_data_path(), dset_name)
    dset_dir = os.path.join(get_test_data_path(), "test_pain_dataset")
    mask_file = os.path.join(dset_dir, "mask.nii.gz")
    dset = nimare.dataset.Dataset(dset_file, mask=mask_file)
    dset.update_path(dset_dir)

    for column in dset.images.columns:
        if column.endswith("__relative"):
            continue
        for path in dset.images[column].values:
            if (path is None) or not os.path.isfile(path):
                continue
            new_path = path.replace(dset_dir.rstrip(os.path.sep), str(tmpdir).rstrip(os.path.sep))
            dirname = os.path.dirname(new_path)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            copyfile(path, new_path)

    dset.update_path(tmpdir)
    return dset


class TimeIBMA:
    """Time IBMA estimators on a dataset with one image per study.

    Dependence correction never engages here, so these are the baseline costs.
    """

    def setup(self):
        """Stage the single-contrast test dataset."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dataset = _stage_dataset("test_pain_dataset.json", self.tmpdir.name)

    def teardown(self):
        """Remove the temporary directory."""
        self.tmpdir.cleanup()

    def time_fishers(self):
        """Time the Fishers estimator."""
        Fishers().fit(self.dataset)

    def time_stouffers(self):
        """Time the Stouffers estimator."""
        Stouffers().fit(self.dataset)

    def time_weighted_least_squares(self):
        """Time the WeightedLeastSquares estimator."""
        WeightedLeastSquares().fit(self.dataset)

    def time_dersimonian_laird(self):
        """Time the DerSimonianLaird estimator."""
        DerSimonianLaird().fit(self.dataset)

    def time_hedges(self):
        """Time the Hedges estimator."""
        Hedges().fit(self.dataset)


class TimeIBMADependence:
    """Time the cost of correcting for studies contributing several images.

    Each estimator is timed with ``dependence="auto"`` (the default, which
    corrects) against ``dependence="independent"`` (which does not) on a
    dataset where studies do contribute multiple contrasts. The gap is the
    price of the correction: building the correlation matrix in
    ``_preprocess_dependence``, plus the cluster-robust covariance (or, for
    Fishers, Brown's moments) in each estimator.

    The correlation matrix is ``O(k^2 * v)`` over all voxels and is expected to
    dominate, so watch that rather than the sandwich itself.
    """

    param_names = ("estimator", "dependence")
    params = (sorted(ESTIMATORS), ["auto", "independent"])

    def setup(self, estimator, dependence):  # noqa: ARG002
        """Stage the multiple-contrast test dataset."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dataset = _stage_dataset(
            "test_pain_dataset_multiple_contrasts.json", self.tmpdir.name
        )

    def teardown(self, estimator, dependence):  # noqa: ARG002
        """Remove the temporary directory."""
        self.tmpdir.cleanup()

    def time_fit(self, estimator, dependence):
        """Time a fit with and without the dependence correction."""
        ESTIMATORS[estimator](dependence=dependence).fit(self.dataset)

    def peakmem_fit(self, estimator, dependence):
        """Measure peak memory with and without the dependence correction."""
        ESTIMATORS[estimator](dependence=dependence).fit(self.dataset)


class TimeIBMADependenceLiberalMask:
    """Same comparison on the liberal-mask path.

    That path fits a separate model per bag of voxels and subsets the group
    labels for each, so it exercises different code than the aggressive mask.
    """

    param_names = ("estimator", "dependence")
    params = (["fishers", "stouffers", "weighted_least_squares"], ["auto", "independent"])

    def setup(self, estimator, dependence):  # noqa: ARG002
        """Stage the multiple-contrast test dataset."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.dataset = _stage_dataset(
            "test_pain_dataset_multiple_contrasts.json", self.tmpdir.name
        )

    def teardown(self, estimator, dependence):  # noqa: ARG002
        """Remove the temporary directory."""
        self.tmpdir.cleanup()

    def time_fit(self, estimator, dependence):
        """Time a liberal-mask fit with and without the dependence correction."""
        ESTIMATORS[estimator](aggressive_mask=False, dependence=dependence).fit(self.dataset)
