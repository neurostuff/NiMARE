"""Benchmark the CBMA estimators."""

import os

import nimare
from nimare.generate import create_coordinate_dataset
from nimare.meta.cbma import ALE, KDA, MKDAChi2, MKDADensity
from nimare.nimads import Studyset
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
        _, self.dataset_dense = create_coordinate_dataset(
            foci=60,
            n_studies=100,
            foci_percentage="100%",
            seed=123,
        )
        self.studyset = Studyset.from_dataset(self.dataset)
        self.studyset_dense = Studyset.from_dataset(self.dataset_dense)

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

    def time_mkdadensity_studyset(self):
        """
        Time the MKDADensity estimator on a Studyset converted from the benchmark Dataset.
        """
        meta = MKDADensity()
        meta.fit(self.studyset)

    def time_mkdadensity_dense(self):
        """
        Time the MKDADensity estimator on a denser simulated dataset.

        Fits the MKDADensity estimator to a dataset with >=50 foci/study and
        >=40 studies to showcase compute_kda_ma improvements.
        """
        meta = MKDADensity()
        meta.fit(self.dataset_dense)

    def time_mkdadensity_dense_studyset(self):
        """
        Time the MKDADensity estimator on a dense Studyset input.
        """
        meta = MKDADensity()
        meta.fit(self.studyset_dense)

    def time_kda(self):
        """
        Time the KDA estimator.

        Fits the KDA estimator to the dataset and measures the time taken.
        """
        meta = KDA()
        meta.fit(self.dataset)

    def time_kda_studyset(self):
        """
        Time the KDA estimator on a Studyset converted from the benchmark Dataset.
        """
        meta = KDA()
        meta.fit(self.studyset)

    def time_mkdachi2(self):
        """
        Time the MKDAChi2 estimator.

        Fits the MKDAChi2 estimator to the dataset and measures the time taken.
        """
        meta = MKDAChi2()
        meta.fit(self.dataset, self.dataset)

    def time_mkdachi2_studyset(self):
        """
        Time the MKDAChi2 estimator on Studyset inputs converted from the benchmark Dataset.
        """
        meta = MKDAChi2()
        meta.fit(self.studyset, self.studyset)

    def time_mkdachi2_dense(self):
        """
        Time the MKDAChi2 estimator on a denser simulated dataset.

        Fits the MKDAChi2 estimator to a dataset with >=50 foci/study and
        >=40 studies to showcase compute_kda_ma improvements.
        """
        meta = MKDAChi2()
        meta.fit(self.dataset_dense, self.dataset_dense)

    def time_mkdachi2_dense_studyset(self):
        """
        Time the MKDAChi2 estimator on dense Studyset inputs.
        """
        meta = MKDAChi2()
        meta.fit(self.studyset_dense, self.studyset_dense)

    def time_ale_studyset(self):
        """
        Time the ALE estimator on a Studyset converted from the benchmark Dataset.
        """
        meta = ALE()
        meta.fit(self.studyset)
