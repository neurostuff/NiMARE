import nimare
from nimare.meta.cbma import ALE, MKDADensity, KDA, MKDAChi2
from nimare.tests.utils import get_test_data_path


class TimeCBMA:
    def setup(self):
        self.dataset = nimare.dataset.Dataset(get_test_data_path())

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
        meta.fit(self.dataset)
