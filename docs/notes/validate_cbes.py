"""Validation runs behind the numbers quoted in ``effect_size_cbma.md``.

Not library code and not part of the test suite: these take minutes to hours, and the image
validation downloads the 21-study NIDM pain collection. Run as::

    python docs/notes/validate_cbes.py images
    python docs/notes/validate_cbes.py fpr [n_sims] [n_iters]

``images``
    Pools per-study Hedges' g across the full t images of the NIDM pain studies to build a
    reference map, then thresholds those same images, keeps only the peaks, and asks how well
    CBES recovers the reference from them. ALE and MKDA are included on the identical
    coordinates as a baseline.
``fpr``
    Global null: every focus is noise. Reports the uncorrected false positive rate and whether
    each corrector rejects anywhere, per selection model and null method.
"""

import sys
import warnings

import nibabel as nib
import numpy as np
from scipy import stats

import nimare
from nimare.correct import FDRCorrector, FWECorrector
from nimare.extract import download_nidm_pain
from nimare.generate import create_effect_size_coordinate_studyset
from nimare.meta.cbma import ALE, CBES, MKDADensity
from nimare.meta.cbma.effectsize import peak_stat_to_hedges_g
from nimare.tests.utils import get_test_data_path
from nimare.transforms import ImagesToCoordinates, ImageTransformer

warnings.simplefilter("ignore")

THRESHOLD_Z = 3.2905267314919255

#: Cluster-forming threshold used by the false positive rate run.
CLUSTER_FORMING_P = 0.01


def _pain_studyset():
    import os

    path = download_nidm_pain()
    dset = nimare.dataset.Dataset(os.path.join(get_test_data_path(), "nidm_pain_dset.json"))
    dset.update_path(path)
    return ImageTransformer(target="z").transform(nimare.studyset.normalize_collection(dset))


def _reference_map(studyset):
    """Random-effects pooling of per-study Hedges' g, computed from the full t images."""
    import os

    masker = studyset.masker
    sample_sizes = np.asarray(studyset.sample_sizes(), dtype=float)

    g_stack, var_stack = [], []
    for i, path in enumerate(studyset.images["t"].values):
        if path is None or not os.path.isfile(str(path)):
            continue
        t_data = masker.transform(str(path)).ravel()
        g, var_g = peak_stat_to_hedges_g(
            t_data, np.full(t_data.shape, sample_sizes[i]), stat_type="t"
        )
        g_stack.append(g)
        var_stack.append(var_g)

    g_stack, var_stack = np.vstack(g_stack), np.vstack(var_stack)
    weights = 1.0 / var_stack
    fixed = (weights * g_stack).sum(0) / weights.sum(0)
    q_stat = (weights * (g_stack - fixed) ** 2).sum(0)
    scale = weights.sum(0) - (weights**2).sum(0) / weights.sum(0)
    tau2 = np.nan_to_num(
        np.clip((q_stat - (g_stack.shape[0] - 1)) / np.where(scale > 0, scale, np.nan), 0, None)
    )
    re_weights = 1.0 / (var_stack + tau2)
    return (re_weights * g_stack).sum(0) / re_weights.sum(0), g_stack.shape[0]


def run_images():
    studyset = _pain_studyset()
    reference, n_images = _reference_map(studyset)
    print(f"{n_images} t images, {reference.size} voxels, reference g max {reference.max():.3f}")

    coords = ImagesToCoordinates(
        merge_strategy="demolish",
        z_threshold=THRESHOLD_Z,
        two_sided=True,
        remove_subpeaks=True,
    ).transform(studyset)
    n_foci = len(coords.coordinates)
    print(f"{n_foci} peaks survive thresholding ({n_foci / reference.size * 100:.2f}% of voxels)")

    estimators = [
        ("CBES none", CBES(fwhm=10.0, selection_model="none", null_method="parametric"), "g"),
        ("CBES tobit", CBES(fwhm=10.0, selection_model="tobit", null_method="parametric"), "g"),
        (
            "CBES zero-inflated",
            CBES(fwhm=10.0, selection_model="zero-inflated", null_method="parametric"),
            "g",
        ),
        (
            "CBES zero-inf (marginal)",
            CBES(fwhm=10.0, selection_model="zero-inflated", null_method="parametric"),
            "g_marginal",
        ),
        ("ALE", ALE(), "z"),
        ("MKDADensity", MKDADensity(), "z"),
    ]

    big = reference > 0.2
    print(f"\n{'estimator':26s} {'r':>7s} {'rho':>7s} {'slope':>7s} {'mean|ref>.2':>12s}")
    for name, estimator, map_name in estimators:
        values = estimator.fit(coords).get_map(map_name, return_type="array").ravel()
        r = stats.pearsonr(values, reference)[0]
        rho = stats.spearmanr(values, reference)[0]
        slope = np.polyfit(values, reference, 1)[0]
        print(f"{name:26s} {r:7.3f} {rho:7.3f} {slope:7.3f} {values[big].mean():12.3f}")
    print(f"{'(reference)':26s} {'':7s} {'':7s} {'':7s} {reference[big].mean():12.3f}")


def run_fpr(n_sims=10, n_iters=100):
    affine = np.array([[4.0, 0, 0, -40.0], [0, 4.0, 0, -40.0], [0, 0, 4.0, -40.0], [0, 0, 0, 1.0]])
    mask = nib.Nifti1Image(np.ones((21, 21, 21), dtype=np.int32), affine)

    def rejects(maps, name):
        return bool(np.any(10.0 ** -maps[name] < 0.05))

    print(f"global null: 30 studies x 8 noise foci, {n_sims} sims, {n_iters} null iterations")
    header = (
        f"{'model':16s} {'null':12s} {'p<.05':>7s} {'p<.01':>7s} {'bonf':>6s} {'FDR':>6s} "
        f"{'vFWE':>6s} {'cFWE-size':>10s} {'cFWE-mass':>10s}"
    )
    print(header)

    for model in ("none", "zero-inflated"):
        for null_method in ("parametric", "montecarlo"):
            rates01, rates05 = [], []
            bonferroni, fdr, voxel, size, mass = [], [], [], [], []
            for seed in range(n_sims):
                studyset = create_effect_size_coordinate_studyset(
                    [(0, 0, 0)],
                    effect_sizes=0.0,
                    n_studies=30,
                    sample_size=(20, 40),
                    prevalence=0.0,
                    n_noise_foci=8,
                    noise_extent=36.0,
                    seed=seed,
                )
                estimator = CBES(
                    fwhm=12.0,
                    mask=mask,
                    selection_model=model,
                    null_method=null_method,
                    n_iters=n_iters,
                    # Match the forming threshold the correction below asks for, so fit()'s
                    # permutations record the cluster nulls and they are not rebuilt.
                    cluster_threshold=CLUSTER_FORMING_P,
                    seed=1000 * seed,
                )
                result = estimator.fit(studyset)
                p_map = result.get_map("p", return_type="array")
                rates05.append(np.mean(p_map < 0.05))
                rates01.append(np.mean(p_map < 0.01))
                bonferroni.append(
                    np.any(
                        FWECorrector(method="bonferroni")
                        .transform(result)
                        .maps["p_corr-FWE_method-bonferroni"]
                        < 0.05
                    )
                )
                fdr.append(
                    np.any(
                        FDRCorrector(method="indep", alpha=0.05)
                        .transform(result)
                        .maps["p_corr-FDR_method-indep"]
                        < 0.05
                    )
                )
                if null_method == "montecarlo":
                    maps, _, _ = estimator.correct_fwe_montecarlo(
                        result, voxel_thresh=CLUSTER_FORMING_P, n_iters=n_iters
                    )
                    voxel.append(rejects(maps, "logp_level-voxel"))
                    size.append(rejects(maps, "logp_desc-size_level-cluster"))
                    mass.append(rejects(maps, "logp_desc-mass_level-cluster"))

            def mean_or_nan(values):
                return np.mean(values) if values else float("nan")

            print(
                f"{model:16s} {null_method:12s} {np.mean(rates05):7.3f} {np.mean(rates01):7.3f} "
                f"{np.mean(bonferroni):6.2f} {np.mean(fdr):6.2f} {mean_or_nan(voxel):6.2f} "
                f"{mean_or_nan(size):10.2f} {mean_or_nan(mass):10.2f}"
            )


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "images"
    if what == "images":
        run_images()
    elif what == "fpr":
        run_fpr(*(int(a) for a in sys.argv[2:]))
    else:
        raise SystemExit(f"unknown validation {what!r}; expected 'images' or 'fpr'")
