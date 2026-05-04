"""Tests for cross-implementation ALE compatibility functionality."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nimare.correct import FWECorrector
from nimare.diagnostics import ResampledStability
from nimare.generate import create_coordinate_dataset
from nimare.meta.cbma import (
    ALE,
    ALESubtraction,
    BalancedALESubstraction,
)
from nimare.meta.cbma.predictive import feature_extraction
from nimare.workflows import ContrastWorkflow

JALE_ROOT = Path(__file__).resolve().parents[2] / "JALE"


@pytest.mark.skipif(not JALE_ROOT.exists(), reason="Local JALE checkout is unavailable.")
def test_predictive_feature_extraction_matches_jale():
    """The packaged ALE predictive features should match the reference implementation."""
    sys.path.insert(0, str(JALE_ROOT))
    try:
        from jale.core.utils.cutoff_prediction import (
            feature_extraction as jale_feature_extraction,
        )
    finally:
        sys.path.pop(0)

    nsub = np.array([12, 20, 33, 18], dtype=float)
    nfoci = np.array([5, 7, 9, 6], dtype=float)
    ours = feature_extraction(4, nsub, nfoci)
    theirs = jale_feature_extraction(4, nsub, nfoci)
    np.testing.assert_allclose(ours, theirs)


def test_resampled_stability_smoke(testdata_cbma_full):
    """A ResampledStability run should produce a bounded voxelwise stability map."""
    dset = testdata_cbma_full.slice(testdata_cbma_full.ids[:5])
    result = ALE().fit(dset)
    result = FWECorrector(method="montecarlo", n_iters=2, n_cores=1).transform(result)
    diagnostic = ResampledStability(
        target_image="z_level-voxel_corr-FWE_method-montecarlo",
        resampling_policy="leave_1_out",
        n_cores=1,
    )
    result = diagnostic.transform(result)
    map_name = "z_level-voxel_corr-FWE_method-montecarlo_diag-ResampledStability"
    values = result.get_map(map_name, return_type="array")
    assert np.all(values >= 0)
    assert np.all(values <= 1)


def test_balanced_ale_substraction_smoke(testdata_cbma_full):
    """A BalancedALESubstraction run should emit balanced subtraction outputs."""
    dset1 = testdata_cbma_full.slice(testdata_cbma_full.ids[:5])
    dset2 = testdata_cbma_full.slice(testdata_cbma_full.ids[5:10])
    result = BalancedALESubstraction(
        target_n=3,
        n_subsamples=4,
        difference_iterations=2,
        n_iters=4,
        n_cores=1,
    ).fit(dset1, dset2)
    assert "stat_desc-balancedGroup1MinusGroup2" in result.maps
    assert "z_desc-balancedGroup1MinusGroup2" in result.maps
    assert "stat_desc-conjunction" in result.maps


def test_predictive_corrector_smoke_if_xgboost_available(testdata_cbma_full):
    """Predictive ALE correction should expose thresholded vFWE and cFWE maps."""
    pytest.importorskip("xgboost")
    dset = testdata_cbma_full.slice(testdata_cbma_full.ids[:8])
    result = ALE().fit(dset)
    result = FWECorrector(method="predictive").transform(result)
    assert "z_level-voxel_corr-FWE_method-predictive" in result.maps
    assert "z_desc-size_level-cluster_corr-FWE_method-predictive" in result.maps


def _prefix_dataset_ids(dataset, prefix):
    dataset = dataset.copy()
    old_ids = list(dataset.ids)
    new_ids = np.array([f"{prefix}{id_}" for id_ in old_ids])
    id_map = dict(zip(old_ids, new_ids))
    dataset._ids = new_ids
    for attr in ("annotations", "coordinates", "images", "metadata", "texts"):
        df = getattr(dataset, attr)
        if df is None or not hasattr(df, "columns"):
            continue
        df = df.copy()
        if "id" in df.columns:
            df["id"] = df["id"].map(lambda x: id_map.get(x, x))
        if "study_id" in df.columns:
            df["study_id"] = prefix + df["study_id"].astype(str)
        setattr(dataset, attr, df)
    return dataset


def _jale_mm2vox(coords_mm, affine):
    coords_mm = np.asarray(coords_mm, dtype=float)
    padded = np.pad(coords_mm, ((0, 0), (0, 1)), constant_values=1)
    vox = np.ceil(np.dot(np.linalg.inv(affine), padded.T))[:3].T.astype(int)
    return np.minimum(vox, np.array([90, 108, 90]))


def _build_jale_experiment_table(dataset, affine):
    metadata = dataset.metadata.set_index("id")
    rows = []
    for exp_id, coords_df in dataset.coordinates.groupby("id"):
        coords_mm = coords_df[["x", "y", "z"]].to_numpy(dtype=float)
        sample_sizes = metadata.loc[exp_id, "sample_sizes"]
        if isinstance(sample_sizes, (list, tuple, np.ndarray)):
            subjects = int(round(float(np.mean(sample_sizes))))
        else:
            subjects = int(sample_sizes)
        rows.append(
            {
                "Articles": exp_id,
                "Subjects": subjects,
                "NumberOfFoci": coords_mm.shape[0],
                "Coordinates_mm": coords_mm,
                "Coordinates": _jale_mm2vox(coords_mm, affine),
            }
        )
    return pd.DataFrame(rows)


def _nimare_to_jale_volume(masked_values, masker):
    return masker.inverse_transform(np.asarray(masked_values)).get_fdata(dtype=np.float32)[
        ::-1, :, :
    ]


@pytest.mark.skipif(not JALE_ROOT.exists(), reason="Local JALE checkout is unavailable.")
def test_contrast_workflow_ale_matches_jale_contrast():
    """ALE ContrastWorkflow should approximate the reference masked contrast workflow."""
    sys.path.insert(0, str(JALE_ROOT))
    try:
        from jale.core.utils.compute import (
            compute_ale,
            compute_hx,
            compute_hx_conv,
            compute_ma,
            compute_monte_carlo_null,
            compute_permuted_ale_diff,
            compute_sig_diff,
            compute_z,
        )
        from jale.core.utils.kernel import create_kernel_array
        from jale.core.utils.template import GM_PRIOR
        from jale.core.utils.template import MNI_AFFINE as JALE_AFFINE
    finally:
        sys.path.pop(0)

    def _main_effect_state(dataset):
        exp_df = _build_jale_experiment_table(dataset, JALE_AFFINE)
        kernels = create_kernel_array(exp_df)
        ma = compute_ma(exp_df.Coordinates.values, kernels)
        ale = compute_ale(ma)
        ale[GM_PRIOR == 0] = 0
        max_ma = np.prod([1 - np.max(kernel) for kernel in kernels])
        bin_edges = np.arange(0.00005, 1 - max_ma + 0.001, 0.0001)
        bin_centers = np.arange(0, 1 - max_ma + 0.001, 0.0001)
        step = int(round(1 / 0.0001))
        hx = compute_hx(ma, bin_edges)
        hx_conv = compute_hx_conv(hx, bin_centers, step)
        z = compute_z(ale, hx_conv, step)
        z[GM_PRIOR == 0] = 0
        return {
            "exp_df": exp_df,
            "kernels": kernels,
            "ma": ma,
            "ale": ale,
            "bin_edges": bin_edges,
            "bin_centers": bin_centers,
            "hx_conv": hx_conv,
            "step": step,
            "z": z,
        }

    def _jale_cfwe_map(state, n_iters, seed):
        rng = np.random.RandomState(seed)
        cfwe_null = np.zeros(n_iters, dtype=float)
        for i_iter in range(n_iters):
            np.random.set_state(rng.get_state())
            _, cfwe_null[i_iter], _ = compute_monte_carlo_null(
                num_foci=state["exp_df"].NumberOfFoci.to_numpy(),
                kernels=state["kernels"],
                bin_edges=state["bin_edges"],
                bin_centers=state["bin_centers"],
                step=state["step"],
                cluster_forming_threshold=0.001,
                target_n=None,
                hx_conv=state["hx_conv"],
                tfce_enabled=False,
            )
            rng.seed(seed + i_iter + 1)
        cluster_threshold = float(np.percentile(cfwe_null, 95))

        from jale.core.utils.compute import compute_clusters

        cfwe_map, _ = compute_clusters(state["z"].copy(), 0.001, cfwe_threshold=cluster_threshold)
        return cfwe_map

    def _dice(a, b):
        a = np.asarray(a, dtype=bool)
        b = np.asarray(b, dtype=bool)
        denom = a.sum() + b.sum()
        return 1.0 if denom == 0 else 2 * np.sum(a & b) / denom

    _, group1 = create_coordinate_dataset(
        foci=[[0, -20, 10], [-36, 18, 18]],
        foci_percentage="80%",
        sample_size=[20, 40],
        n_studies=12,
        n_noise_foci=2,
        seed=201,
    )
    _, group2 = create_coordinate_dataset(
        foci=[[0, -20, 10], [36, 18, 18]],
        foci_percentage="80%",
        sample_size=[20, 40],
        n_studies=12,
        n_noise_foci=2,
        seed=202,
    )
    group1 = _prefix_dataset_ids(group1, "g1-")
    group2 = _prefix_dataset_ids(group2, "g2-")

    state1 = _main_effect_state(group1)
    state2 = _main_effect_state(group2)
    main_effect1 = _jale_cfwe_map(state1, n_iters=16, seed=13)
    main_effect2 = _jale_cfwe_map(state2, n_iters=16, seed=14)

    jale_contrast = np.zeros_like(main_effect1, dtype=np.float32)
    sig_mask1 = main_effect1 > 0
    if sig_mask1.any():
        null_difference1 = np.asarray(
            [
                compute_permuted_ale_diff(
                    np.vstack((state1["ma"][:, sig_mask1], state2["ma"][:, sig_mask1])),
                    state1["ma"].shape[0],
                )
                for _ in range(32)
            ]
        )
        z1, idx1 = compute_sig_diff(
            state1["ale"][sig_mask1] - state2["ale"][sig_mask1],
            null_difference1,
            0.05,
        )
        idx1 = idx1[0] if isinstance(idx1, tuple) else np.array([], dtype=int)
        if idx1.size:
            flat1 = np.where(sig_mask1)
            jale_contrast[flat1[0][idx1], flat1[1][idx1], flat1[2][idx1]] = z1

    sig_mask2 = main_effect2 > 0
    if sig_mask2.any():
        null_difference2 = np.asarray(
            [
                compute_permuted_ale_diff(
                    np.vstack((state1["ma"][:, sig_mask2], state2["ma"][:, sig_mask2])),
                    state2["ma"].shape[0],
                )
                for _ in range(32)
            ]
        )
        z2, idx2 = compute_sig_diff(
            state2["ale"][sig_mask2] - state1["ale"][sig_mask2],
            null_difference2,
            0.05,
        )
        idx2 = idx2[0] if isinstance(idx2, tuple) else np.array([], dtype=int)
        if idx2.size:
            flat2 = np.where(sig_mask2)
            jale_contrast[flat2[0][idx2], flat2[1][idx2], flat2[2][idx2]] = -z2

    workflow = ContrastWorkflow(
        corrector=FWECorrector(method="montecarlo", n_iters=16, n_cores=1),
        pairwise_estimator=ALESubtraction(n_iters=32, n_cores=1, generate_description=False),
        alpha=0.05,
        n_cores=1,
    )
    result = workflow.fit(group1, group2)
    nimare_contrast = _nimare_to_jale_volume(
        result.get_map("z_desc-contrast", return_type="array"), group1.masker
    )

    support_dice = _dice(jale_contrast != 0, nimare_contrast != 0)
    assert support_dice >= 0.8
