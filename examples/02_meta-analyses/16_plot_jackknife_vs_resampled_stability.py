"""Jackknife vs. ResampledStability: Choosing the Right Diagnostic.

This notebook runs an ALE meta-analysis on the NiMARE pain dataset, applies Jackknife
and ResampledStability diagnostics, and plots the results.
"""

import copy
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from nilearn.plotting import plot_stat_map

from nimare.correct import FWECorrector
from nimare.diagnostics import Jackknife, ResampledStability
from nimare.meta.cbma.ale import ALE
from nimare.nimads import Studyset
from nimare.utils import get_resource_path


N_ITERS = 50
N_RESAMPLES = 20
RANDOM_STATE = 42


def main():
    warnings.filterwarnings("ignore")

    studyset_file = os.path.join(get_resource_path(), "nidm_pain_studyset.json")
    studyset = Studyset(studyset_file, target="mni152_2mm")
    print(f"Number of studies: {len(studyset.studies)}")

    ale = ALE()
    result = ale.fit(studyset)

    corrector = FWECorrector(method="montecarlo", n_iters=N_ITERS, n_cores=1)
    result = corrector.transform(result)

    TARGET_IMAGE = "z_desc-size_level-cluster_corr-FWE_method-montecarlo"

    print("Available maps:")
    for k in result.maps:
        print(" ", k)

    plot_stat_map(
        result.get_map(TARGET_IMAGE),
        cut_coords=5,
        display_mode="z",
        title="ALE (cluster-level FWE corrected)",
        threshold=1.65,
        cmap="RdBu_r",
        symmetric_cbar=True,
        vmax=5,
    )
    plt.show()

    jackknife = Jackknife(target_image=TARGET_IMAGE, n_cores=1)
    result_jk = jackknife.transform(copy.deepcopy(result))

    print("Tables added by Jackknife:")
    for k in result_jk.tables:
        print(" ", k)

    clust_key = f"{TARGET_IMAGE}_tab-clust"
    print("\nJackknife clusters table:")
    print(result_jk.tables[clust_key])

    contrib_key = f"{TARGET_IMAGE}_diag-Jackknife_tab-counts_tail-positive"
    contrib_df = result_jk.tables.get(contrib_key)
    print("\nJackknife study contribution table:")
    print(contrib_df)

    if contrib_df is not None and not contrib_df.empty:
        fig, ax = plt.subplots(
            figsize=(max(6, len(contrib_df.columns) * 1.2), max(5, len(contrib_df) * 0.35))
        )
        im = ax.imshow(contrib_df.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(contrib_df.columns)))
        ax.set_xticklabels(contrib_df.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(contrib_df)))
        ax.set_yticklabels(contrib_df.index, fontsize=7)
        ax.set_xlabel("Cluster", fontsize=11)
        ax.set_ylabel("Study", fontsize=11)
        ax.set_title("Jackknife: proportional contribution per study per cluster", fontsize=12)
        plt.colorbar(im, ax=ax, label="Contribution (0 = none, 1 = complete)")
        plt.tight_layout()
        plt.show()

        mean_contrib = contrib_df.mean(axis=1).sort_values(ascending=False)
        print("\nMean contribution across all clusters (top 10):")
        print(mean_contrib.head(10).to_string())
    else:
        print("No clusters found — try lowering the cluster_threshold or increasing N_ITERS.")

    rs_loo = ResampledStability(target_image=TARGET_IMAGE, resampling_policy="leave_1_out", n_cores=1)
    result_loo = rs_loo.transform(copy.deepcopy(result))
    print("\nLeave-one-out ResampledStability summary:")
    print(result_loo.tables[f"{TARGET_IMAGE}_diag-ResampledStability_tab-summary"])

    rs_lko = ResampledStability(
        target_image=TARGET_IMAGE,
        resampling_policy="leave_k_out",
        k=3,
        n_resamples=N_RESAMPLES,
        random_state=RANDOM_STATE,
        n_cores=1,
    )
    result_lko = rs_lko.transform(copy.deepcopy(result))
    print("\nLeave-k-out ResampledStability summary:")
    print(result_lko.tables[f"{TARGET_IMAGE}_diag-ResampledStability_tab-summary"])

    n_studies = len(studyset.studies)
    target_n = max(3, int(n_studies * 0.75))
    rs_sub = ResampledStability(
        target_image=TARGET_IMAGE,
        resampling_policy="subsample",
        target_n=target_n,
        n_resamples=N_RESAMPLES,
        random_state=RANDOM_STATE,
        n_cores=1,
    )
    result_sub = rs_sub.transform(copy.deepcopy(result))
    print("\nSubsample ResampledStability summary:")
    print(result_sub.tables[f"{TARGET_IMAGE}_diag-ResampledStability_tab-summary"])

    stability_key = f"{TARGET_IMAGE}_diag-ResampledStability"
    configs = [
        (result_loo, "Leave-one-out"),
        (result_lko, f"Leave-{rs_lko.k}-out ({N_RESAMPLES} resamples)"),
        (result_sub, f"Subsample n={target_n} ({N_RESAMPLES} resamples)"),
    ]

    fig, axes = plt.subplots(len(configs), 1, figsize=(14, 4 * len(configs)))
    for ax, (res, title) in zip(axes, configs):
        plot_stat_map(
            res.get_map(stability_key),
            cut_coords=5,
            display_mode="z",
            title=f"Stability ({title})",
            threshold=0.1,
            vmin=0,
            vmax=1,
            cmap="hot",
            symmetric_cbar=False,
            axes=ax,
            figure=fig,
        )
    fig.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, len(configs), figsize=(14, 4), sharey=True)
    for ax, (res, title) in zip(axes, configs):
        stab = res.get_map(stability_key, return_type="array")
        nonzero = stab[stab > 0]
        ax.hist(nonzero, bins=20, range=(0, 1), color="steelblue", edgecolor="white")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Stability")
        ax.set_xlim(0, 1)
        mean_val = nonzero.mean() if len(nonzero) > 0 else 0
        ax.axvline(mean_val, color="red", linestyle="--", label=f"mean = {mean_val:.2f}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Voxel count")
    fig.suptitle("Distribution of non-zero stability values", fontsize=13)
    fig.tight_layout()
    plt.show()

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    plot_stat_map(
        result.get_map(TARGET_IMAGE),
        cut_coords=5,
        display_mode="z",
        title="ALE: cluster-level FWE corrected z-map (baseline)",
        threshold=1.65,
        cmap="RdBu_r",
        symmetric_cbar=True,
        vmax=5,
        axes=axes[0],
        figure=fig,
    )

    if contrib_df is not None and not contrib_df.empty:
        label_key = f"label_{TARGET_IMAGE}_tail-positive"
        if label_key in result_jk.maps:
            plot_stat_map(
                result_jk.get_map(label_key),
                cut_coords=5,
                display_mode="z",
                title="Jackknife: cluster label map (each integer = one cluster)",
                threshold=0.5,
                cmap="Set1",
                symmetric_cbar=False,
                axes=axes[1],
                figure=fig,
            )
        else:
            axes[1].set_title("Jackknife label map not available")
    else:
        axes[1].set_title("No clusters found for Jackknife")

    plot_stat_map(
        result_loo.get_map(stability_key),
        cut_coords=5,
        display_mode="z",
        title="ResampledStability (leave-one-out): voxelwise stability (0–1)",
        threshold=0.1,
        vmin=0,
        vmax=1,
        cmap="hot",
        symmetric_cbar=False,
        axes=axes[2],
        figure=fig,
    )

    fig.tight_layout()
    plt.show()

    rows = []
    for res, label in configs:
        stab = res.get_map(stability_key, return_type="array")
        nonzero = stab[stab > 0]
        rows.append(
            {
                "Policy": label,
                "N replicates": int(
                    res.tables[f"{TARGET_IMAGE}_diag-ResampledStability_tab-summary"]["n_resamples"].iloc[0]
                ),
                "Stable voxels (>0)": int(len(nonzero)),
                "Stable voxels (≥0.5)": int((stab >= 0.5).sum()),
                "Stable voxels (≥0.8)": int((stab >= 0.8).sum()),
                "Mean stability (nonzero)": round(float(nonzero.mean()), 3) if len(nonzero) > 0 else 0.0,
            }
        )

    print("\nCombined numerical summary:")
    print(pd.DataFrame(rows).set_index("Policy"))


if __name__ == "__main__":
    main()
