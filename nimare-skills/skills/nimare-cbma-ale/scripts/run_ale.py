#!/usr/bin/env python3
"""Run a defensible ALE meta-analysis with NiMARE.

Applies the settings the ALE literature validates rather than NiMARE's constructor
defaults: cluster-level FWE by Monte Carlo (not the ``bonferroni`` default), a p<0.001
cluster-forming threshold, >=5000 iterations, and mandatory contribution diagnostics.
Refuses to run silently on an underpowered study set.

Usage
-----
    python run_ale.py foci.txt -o results/
    python run_ale.py studyset.json -o results/ --n-iters 10000 --n-cores -1
    python run_ale.py foci.txt -o results/ --pool          # pool contrasts per study first
    python run_ale.py foci.txt -o results/ --fwhm 15       # no sample sizes available
"""

import argparse
import json
import sys
from pathlib import Path

MIN_EXPERIMENTS = 17  # Müller et al. (2018)


def load_studyset(path, pool):
    from nimare.studyset import Studyset

    path = Path(path)
    if path.suffix.lower() in {".txt", ".text"}:
        studyset = Studyset.from_sleuth(str(path))
    elif path.is_dir():
        studyset = Studyset.from_parquet(str(path))
    else:
        studyset = Studyset.from_nimads(str(path))
    if pool:
        before = len(studyset.ids)
        studyset = studyset.combine_analyses()
        print(f"Pooled contrasts within studies: {before} -> {len(studyset.ids)} analyses.")
        print("  (correct only if every analysis in a study shares one sample)")
    return studyset


def preflight(studyset, force):
    n = len(studyset.ids)
    n_studies = len(set(str(s) for s in studyset.study_ids))
    print(f"Experiments NiMARE will fit : {n}")
    print(f"Distinct study ids          : {n_studies}")
    if n != n_studies:
        print(f"  [!] {n - n_studies} analyses come from a study that contributes more than")
        print("      one. NiMARE counts each as an independent experiment. Run")
        print("      audit_independence.py, and use --pool if they share a sample.")
    if n < MIN_EXPERIMENTS:
        print(f"  [!!] Fewer than {MIN_EXPERIMENTS} experiments: ALE is underpowered here")
        print("       (Müller et al., 2018). NiMARE itself enforces no minimum.")
        if not force:
            print("       Refusing to run. Pass --force to proceed and report the limitation.")
            return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="Sleuth .txt, NiMADS .json, or parquet directory")
    ap.add_argument("-o", "--output-dir", default="ale_results")
    ap.add_argument("--n-iters", type=int, default=5000,
                    help="Monte Carlo iterations for FWE correction (default 5000)")
    ap.add_argument("--voxel-thresh", type=float, default=0.001,
                    help="cluster-forming threshold, uncorrected p (default 0.001)")
    ap.add_argument("--fwhm", type=float, default=None,
                    help="fixed ALE kernel FWHM in mm; use when sample sizes are missing "
                         "(15 mm is the usual choice for database-derived sets)")
    ap.add_argument("--n-cores", type=int, default=1)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--pool", action="store_true",
                    help="Studyset.combine_analyses() before fitting")
    ap.add_argument("--force", action="store_true",
                    help="run even with too few experiments")
    args = ap.parse_args(argv)

    from nimare.correct import FWECorrector
    from nimare.diagnostics import FocusCounter, FocusFilter, Jackknife
    from nimare.meta.cbma import ALE
    from nimare.meta.kernel import ALEKernel

    studyset = load_studyset(args.path, args.pool)

    try:
        before = len(studyset.coordinates)
        studyset = FocusFilter(mask=studyset.masker).transform(studyset)
        dropped = before - len(studyset.coordinates)
        if dropped:
            print(f"FocusFilter dropped {dropped} out-of-mask foci.")
    except Exception as exc:  # pragma: no cover - mask may be undefined
        print(f"FocusFilter skipped ({exc}).")

    if not preflight(studyset, args.force):
        return 2

    kernel = ALEKernel(fwhm=args.fwhm) if args.fwhm else ALEKernel()
    est = ALE(kernel_transformer=kernel, null_method="approximate",
              random_state=args.random_state)
    print("Fitting ALE ...")
    result = est.fit(studyset)

    print(f"Correcting: cluster-level FWE, Monte Carlo, {args.n_iters} iterations ...")
    corr = FWECorrector(method="montecarlo", voxel_thresh=args.voxel_thresh,
                        n_iters=args.n_iters, n_cores=args.n_cores)
    cres = corr.transform(result)

    target = "z_desc-mass_level-cluster_corr-FWE_method-montecarlo"
    if target not in cres.maps:
        target = next(k for k in cres.maps if k.startswith("z_desc-mass"))
    print(f"Diagnostics on {target} ...")
    cres = Jackknife(target_image=target, voxel_thresh=None).transform(cres)
    cres = FocusCounter(target_image=target, voxel_thresh=None).transform(cres)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cres.save_maps(output_dir=str(out))
    cres.save_tables(output_dir=str(out))

    provenance = {
        "nimare_version": __import__("nimare").__version__,
        "input": str(args.path),
        "pooled_within_study": bool(args.pool),
        "n_experiments": len(studyset.ids),
        "n_studies": len(set(str(s) for s in studyset.study_ids)),
        "estimator": "ALE",
        "kernel": f"ALEKernel(fwhm={args.fwhm})" if args.fwhm
                  else "ALEKernel(sample-size-derived FWHM)",
        "null_method": "approximate",
        "correction": "FWE, montecarlo, cluster-level",
        "cluster_forming_threshold_p": args.voxel_thresh,
        "n_iters": args.n_iters,
        "random_state": args.random_state,
        "reported_map": target,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))

    clust_key = f"{target}_tab-clust"
    if clust_key in cres.tables:
        table = cres.tables[clust_key]
        print(f"\n{len(table)} cluster(s) survived (p_FWE < .05, "
              f"cluster-forming p < {args.voxel_thresh}):")
        print(table.to_string(index=False))
    else:
        print("\nNo cluster table produced; inspect sorted(cres.tables).")

    print(f"\nWritten to {out}/ (maps, tables, provenance.json).")
    print("Report the jackknife/focus-counter tables alongside the clusters: a cluster")
    print("carried by one or two experiments is a finding about those experiments.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
