#!/usr/bin/env python3
"""Audit a NiMARE study set for contrasts that are not independent experiments.

NiMARE's coordinate-based estimators treat every *analysis* (contrast) as an independent
experiment: ids are ``"<study_id>-<analysis_id>"`` and nothing groups them by subject
sample. Two contrasts from the same participants are therefore counted twice, with no
warning. This script finds the cases and can pool them.

Usage
-----
    python audit_independence.py FOCI.txt                  # Sleuth/BrainMap text file
    python audit_independence.py STUDYSET.json             # NiMADS studyset
    python audit_independence.py DATASET.json --kind dataset
    python audit_independence.py FOCI.txt --pool pooled.json

Exit status is 1 when a likely double-count is found, so it can gate a pipeline.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

MIN_EXPERIMENTS = 17  # Müller et al. (2018), adequate power for ALE


def load_studyset(path, kind):
    from nimare.studyset import Studyset

    path = Path(path)
    if kind == "auto":
        if path.suffix.lower() in {".txt", ".text"}:
            kind = "sleuth"
        elif path.is_dir():
            kind = "parquet"
        else:
            kind = "nimads"

    if kind == "sleuth":
        return Studyset.from_sleuth(str(path))
    if kind == "parquet":
        return Studyset.from_parquet(str(path))
    if kind == "dataset":
        from nimare.dataset import Dataset

        return Studyset.from_dataset(Dataset(str(path)))
    return Studyset.from_nimads(str(path))


def sample_sizes_by_analysis(studyset):
    """Best-effort ``{full analysis id: sample size}``.

    ``Studyset.sample_sizes()`` returns a plain array aligned with ``Studyset.ids``;
    the metadata table carries ``sample_sizes`` as a per-analysis list. Try both.
    """
    ids = [str(i) for i in studyset.ids]
    out = {}
    try:
        values = studyset.sample_sizes(reduce="mean")
        if values is not None and len(values) == len(ids):
            for key, value in zip(ids, values):
                if value is not None and value == value:  # not NaN
                    out[key] = float(value)
    except Exception:
        pass
    if out:
        return out
    try:
        meta = studyset.metadata
        if "sample_sizes" in meta.columns:
            for _, row in meta.iterrows():
                raw = row["sample_sizes"]
                if raw is None:
                    continue
                values = raw if isinstance(raw, (list, tuple)) else [raw]
                values = [float(v) for v in values if v is not None and v == v]
                if values:
                    out[str(row["id"])] = sum(values) / len(values)
    except Exception:
        pass
    return out


def analyses_by_study(studyset):
    """``{study_id: [full analysis id, ...]}`` using the declared study ids."""
    per_study = defaultdict(list)
    seen = set()
    for frame_name in ("metadata", "coordinates"):
        try:
            frame = getattr(studyset, frame_name)
        except Exception:
            continue
        if frame is None or "study_id" not in getattr(frame, "columns", []):
            continue
        for _, row in frame.iterrows():
            full_id = str(row["id"])
            if full_id in seen:
                continue
            seen.add(full_id)
            per_study[str(row["study_id"])].append(full_id)
        if per_study:
            break
    if not per_study:  # last resort: split the id
        for full_id in (str(i) for i in studyset.ids):
            study, _, _ = full_id.rpartition("-")
            per_study[study or full_id].append(full_id)
    return per_study


_AUTHOR_YEAR = re.compile(r"^(?P<author>[A-Za-z]+).*?(?P<year>(?:19|20)\d{2})")


def author_year_key(study_id):
    """``smith2010faces`` / ``Smith, 2010`` -> ``('smith', '2010')``; else None."""
    m = _AUTHOR_YEAR.match(str(study_id).strip().lower().replace(" ", ""))
    if not m:
        return None
    return (m.group("author"), m.group("year"))


def audit(studyset):
    per_study = analyses_by_study(studyset)
    sizes = sample_sizes_by_analysis(studyset)
    findings = {"multi_analysis": [], "same_n_within_study": [], "split_studies": []}

    for study, analyses in sorted(per_study.items()):
        if len(analyses) < 2:
            continue
        ns = [sizes.get(a) for a in analyses]
        findings["multi_analysis"].append((study, analyses, ns))
        known = [n for n in ns if n is not None]
        if len(known) > 1 and len({round(float(n), 3) for n in known}) == 1:
            findings["same_n_within_study"].append((study, analyses, known[0]))

    # Studies that look like one paper split across study ids (the Sleuth header trap).
    by_key = defaultdict(list)
    for study in sorted(per_study):
        key = author_year_key(study)
        if key:
            by_key[key].append(study)
    for key, studies in sorted(by_key.items()):
        if len(studies) < 2:
            continue
        ns = []
        for study in studies:
            values = [sizes[a] for a in per_study[study] if a in sizes]
            ns.append(values[0] if values else None)
        known = [n for n in ns if n is not None]
        same_n = len(known) == len(studies) and len({round(float(n), 3) for n in known}) == 1
        findings["split_studies"].append((key, studies, ns, same_n))

    return per_study, sizes, findings


def report(studyset, per_study, sizes, findings):
    n_analyses = len(studyset.ids)
    n_studies = len(set(str(s) for s in studyset.study_ids))
    print(f"analyses (what NiMARE will fit) : {n_analyses}")
    print(f"studies                         : {n_studies}")
    print(f"with sample_size recorded       : {len(sizes)}/{n_analyses}")
    print()

    problems = 0

    multi = findings["multi_analysis"]
    if multi:
        print(f"[!] {len(multi)} study/studies contribute more than one analysis.")
        print("    NiMARE will fit each as an independent experiment.")
        for study, analyses, ns in multi[:20]:
            shown = ", ".join(
                f"{a}(N={int(n)})" if n is not None else f"{a}(N=?)"
                for a, n in zip(analyses, ns)
            )
            print(f"      {study}: {shown}")
        if len(multi) > 20:
            print(f"      ... and {len(multi) - 20} more")
        print()

    same_n = findings["same_n_within_study"]
    if same_n:
        problems += len(same_n)
        print(f"[!!] {len(same_n)} study/studies have multiple analyses with an IDENTICAL")
        print("     sample size -- the usual signature of one sample reported twice.")
        for study, analyses, n in same_n[:20]:
            print(f"      {study}: {len(analyses)} analyses, all N={int(n)}")
        print("     -> pool these into one analysis unless the paper states the samples")
        print("        are independent. See --pool.")
        print()

    split = [f for f in findings["split_studies"] if f[3]]
    if split:
        problems += len(split)
        print(f"[!!] {len(split)} group(s) of study ids look like the same paper split apart")
        print("     (same first author + year + sample size, different study ids).")
        print("     Sleuth headers without a ':' or ';' after the year do this.")
        for key, studies, ns, _ in split[:20]:
            n = next((x for x in ns if x is not None), None)
            print(f"      {key[0]} {key[1]} (N={int(n) if n else '?'}): {', '.join(studies)}")
        print("     -> these are CANDIDATES, not verdicts: two cohorts of the same size")
        print("        (e.g. six- and nine-year-olds, N=20 each) are genuinely independent.")
        print("        Check the papers. Where they are one sample, rename the headers to")
        print("        '<Author>, <Year>: <contrast>' and re-import -- combine_analyses()")
        print("        cannot fix this after the ids have been split.")
        print()

    # Lower bound: collapse every study to one sample, and merge the split-study groups.
    effective = len(per_study) - sum(len(studies) - 1 for _, studies, _, flag in
                                     findings["split_studies"] if flag)
    print(f"Independent samples: between {effective} (if every within-study contrast and "
          f"every flagged split shares a sample)")
    print(f"                     and {n_analyses} (if every contrast is its own sample).")
    print("NiMARE will fit %d experiments regardless." % n_analyses)
    if effective < MIN_EXPERIMENTS:
        problems += 1
        print(f"[!!] Fewer than {MIN_EXPERIMENTS} independent experiments. ALE is")
        print("     underpowered below ~17-20 (Müller et al., 2018), and NiMARE enforces")
        print("     no minimum -- it will return a map regardless.")
    print()

    if not problems:
        print("No double-counting signature detected. Still confirm against the papers:")
        print("sample sizes can coincide, and NiMARE records no subject-group identity.")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="Sleuth .txt, NiMADS .json, Dataset .json, or parquet dir")
    ap.add_argument("--kind", default="auto",
                    choices=["auto", "sleuth", "nimads", "dataset", "parquet"])
    ap.add_argument("--pool", metavar="OUT.json",
                    help="write a NiMADS studyset with one analysis per study "
                         "(Studyset.combine_analyses())")
    args = ap.parse_args(argv)

    studyset = load_studyset(args.path, args.kind)
    per_study, sizes, findings = audit(studyset)
    problems = report(studyset, per_study, sizes, findings)

    if args.pool:
        pooled = studyset.combine_analyses()
        with open(args.pool, "w") as fh:
            json.dump(pooled.to_dict(), fh, indent=1)
        print(f"Pooled study set written to {args.pool}: "
              f"{len(pooled.ids)} analyses from {len(set(str(s) for s in pooled.study_ids))} studies.")
        print("WARNING: combine_analyses() pools by STUDY. If a study contributed two")
        print("independent samples (e.g. patients and controls), this wrongly merges them.")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
