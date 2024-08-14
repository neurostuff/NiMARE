import os

import nimare
from nimare.meta.cbma import ALE, MKDADensity, KDA, MKDAChi2
from nimare.tests.utils import get_test_data_path


class TimeCBMA:
    def setup(self):
        self.dataset = nimare.dataset.Dataset(os.path.join(get_test_data_path(), "test_pain_dataset.json"))
    def time_ale(self):
        meta = ALE()
        meta.fit(self.dataset)

    def time_mkdadensity(self):
        meta = MKDADensity()
        meta.fit(self.dataset)

    def time_kda(self):
        meta = KDA()
        meta.fit(self.dataset)

    def time_mkdachi2(self):
        meta = MKDAChi2()
        meta.fit(self.dataset, self.dataset)
