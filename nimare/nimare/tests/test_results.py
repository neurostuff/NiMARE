"""Test nimare.results."""

import os

import numpy as np
import pytest

from nimare.correct import FDRCorrector, FWECorrector
from nimare.diagnostics import Jackknife
from nimare.meta.cbma.mkda import MKDADensity
from nimare.meta.ibma import PermutedOLS, Stouffers
from nimare.reports.base import run_reports
from nimare.results import DroppedInput, MetaResult


class _SubResult(MetaResult):
    """Stand-in for the MetaResult subclasses estimators may return."""


@pytest.fixture(scope="module")
def ibma_result(testdata_ibma):
    """Return a corrected image-based result to save and reload."""
    return FDRCorrector(method="indep").transform(Stouffers().fit(testdata_ibma))


def test_save_with_inputs_round_trips_the_input_arrays(tmp_path, ibma_result):
    """The default save keeps the inputs it always kept."""
    filename = os.path.join(tmp_path, "full.pkl.gz")
    ibma_result.save(filename)
    inputs = MetaResult.load(filename).estimator.inputs_

    assert set(inputs) == set(ibma_result.estimator.inputs_)
    assert np.array_equal(inputs["z_maps"], ibma_result.estimator.inputs_["z_maps"])


def test_save_without_inputs_drops_only_the_image_arrays(tmp_path, ibma_result):
    """The flag empties the image keys, shrinks the file, and leaves the caller's copy."""
    subclassed = ibma_result.copy()
    subclassed.__class__ = _SubResult

    full = os.path.join(tmp_path, "full.pkl.gz")
    lean = os.path.join(tmp_path, "lean.pkl.gz")
    subclassed.save(full)
    subclassed.save(lean, with_inputs=False)
    loaded = _SubResult.load(lean)

    assert os.path.getsize(lean) < 0.5 * os.path.getsize(full)
    assert isinstance(loaded, _SubResult)
    assert isinstance(loaded.estimator.inputs_["z_maps"], DroppedInput)
    assert isinstance(loaded.estimator.inputs_["data_bags"], DroppedInput)
    assert set(loaded.estimator.inputs_) == set(ibma_result.estimator.inputs_)
    assert list(loaded.estimator.inputs_["id"]) == list(ibma_result.estimator.inputs_["id"])
    assert isinstance(subclassed.estimator.inputs_["z_maps"], np.ndarray)


def test_result_without_inputs_still_serves_maps_and_diagnostics(tmp_path, ibma_result):
    """Everything that does not read the input arrays survives the drop."""
    filename = os.path.join(tmp_path, "lean.pkl.gz")
    ibma_result.save(filename, with_inputs=False)
    loaded = MetaResult.load(filename)

    assert loaded.get_map("z") is not None
    loaded.save_maps(output_dir=os.path.join(tmp_path, "maps"))
    loaded.copy()
    FDRCorrector(method="indep").transform(loaded)

    # The jackknife refits from the images on disk, so it needs the ids but not the arrays.
    diagnosed = Jackknife(target_image="z_corr-FDR_method-indep", target_threshold=0.5).transform(
        loaded
    )
    assert any("Jackknife" in name for name in diagnosed.tables)


def test_report_of_a_result_without_inputs_reports_the_drop(tmp_path, ibma_result):
    """Reporting reads the input maps, so it works on a whole file and says why on a lean one."""
    full = os.path.join(tmp_path, "full.pkl.gz")
    lean = os.path.join(tmp_path, "lean.pkl.gz")
    ibma_result.save(full)
    ibma_result.save(lean, with_inputs=False)

    run_reports(MetaResult.load(full), os.path.join(tmp_path, "report"))
    with pytest.raises(RuntimeError, match="z_maps.*with_inputs=False"):
        run_reports(MetaResult.load(lean), os.path.join(tmp_path, "lean_report"))


def test_montecarlo_correction_without_inputs_reports_the_drop(tmp_path, testdata_ibma):
    """Monte Carlo FWE refits from the input arrays, so it reports the drop too."""
    result = PermutedOLS().fit(testdata_ibma)
    full = os.path.join(tmp_path, "full.pkl.gz")
    lean = os.path.join(tmp_path, "lean.pkl.gz")
    result.save(full)
    result.save(lean, with_inputs=False)

    corrector = FWECorrector(method="montecarlo", n_iters=5)
    assert corrector.transform(MetaResult.load(full)).maps
    with pytest.raises(RuntimeError, match="beta_maps.*with_inputs=False"):
        corrector.transform(MetaResult.load(lean))


def test_save_without_inputs_leaves_coordinate_results_alone(tmp_path, testdata_cbma):
    """A coordinate-based estimator declares no image inputs, so nothing is dropped."""
    result = MKDADensity(null_method="approximate").fit(testdata_cbma)
    filename = os.path.join(tmp_path, "lean.pkl.gz")
    result.save(filename, with_inputs=False)
    inputs = MetaResult.load(filename).estimator.inputs_

    assert not any(isinstance(value, DroppedInput) for value in inputs.values())
    assert list(inputs["id"]) == list(result.estimator.inputs_["id"])
