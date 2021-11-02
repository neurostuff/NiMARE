"""Methods for diagnosing problems in meta-analytic datasets or analyses."""
from abc import ABCMeta

import pandas as pd
from nilearn.reporting import get_clusters_table


class Jackknife(metaclass=ABCMeta):
    """Run a jackknife analysis on a meta-analysis result."""
    def __init__(self):
        ...

    def transform(self, result):
        dset = result.estimator.dataset
        estimator = result.estimator
        stat_values = result.get_map("stat", return_type="array")
        cfwe_img = result.get_map("logp_level-cluster_method-montecarlo", return_type="image")

        output = pd.DataFrame(index=dset.ids, columns=cluster_ids)

        for expid in dset.ids:
            other_ids = sorted(list(set(dset.ids) - expid))
            temp_dset = dset.slice(other_ids)
            temp_result = estimator.fit(temp_dset)
            temp_stat_values = temp_result.get_map("stat", return_type="array")
            stat_prop_arr = temp_stat_values / stat_values

