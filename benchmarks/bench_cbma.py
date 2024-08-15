"""Benchmark the CBMA estimators."""

import os

import nimare
from nimare.meta.cbma import ALE, KDA, MKDAChi2, MKDADensity
from nimare.tests.utils import get_test_data_path


class TimeCBMA:
    """Time CBMA estimators."""

    def setup(self):
        """
        Setup the data.

        Loads the dataset required for the benchmarks.
        """
        self.dataset = nimare.dataset.Dataset(
            os.path.join(get_test_data_path(), "test_pain_dataset.json")
        )

    def time_ale(self):
        """
        Time the ALE estimator.

        Fits the ALE estimator to the dataset and measures the time taken.
        """
        meta = ALE()
        meta.fit(self.dataset)

    def time_mkdadensity(self):
        """
        Time the MKDADensity estimator.

        Fits the MKDADensity estimator to the dataset and measures the time taken.
        """
        meta = MKDADensity()
        meta.fit(self.dataset)

    def time_kda(self):
        """
        Time the KDA estimator.

        Fits the KDA estimator to the dataset and measures the time taken.
        """
        meta = KDA()
        meta.fit(self.dataset)

    def time_mkdachi2(self):
        """
        Time the MKDAChi2 estimator.

        Fits the MKDAChi2 estimator to the dataset and measures the time taken.
        """
        meta = MKDAChi2()
        meta.fit(self.dataset, self.dataset)
