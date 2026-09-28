"""Assemble the modelling cohort: QC'd rows joined to normalized labels."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))

import nsnorm  # noqa: E402

# Where the feature cache and result tables live. These run to a couple of
# gigabytes, so they belong outside the repository.
WORK = os.environ.get("NEUROSTORE_EXPLORE_DIR", str(_ROOT.parent / "work"))
W = WORK
os.makedirs(WORK, exist_ok=True)


# Plausibility bounds for the demographic fields; outside them the extractor
# has read the wrong number off the page.
AGE_RANGE = (1.0, 100.0)
COUNT_RANGE = (1, 2000)


def load_cohort():
    """Return the per-row label frame, aligned to the cached feature matrices."""
    rows = pd.read_parquet(f"{W}/feat/rows.parquet")
    labels = pd.read_parquet(f"{W}/work_labels.parquet")
    # The annotations table keys on the full "<study>-<analysis>" id, which
    # rows.parquet carries as ``full_id``.
    out = rows.merge(labels, on="full_id", how="left", suffixes=("", "_lab"))
    assert len(out) == len(rows), "join changed the row count"
    return out


def build_labels():
    """Normalize the annotation table into the label frame the cohort joins to."""
    from nimare.extract import fetch_neurostore

    ss = fetch_neurostore(version="nightly")
    ann = ss.annotations_df
    G = "ParticipantDemographicsExtractor.groups[0]."

    df = pd.DataFrame({
        "full_id": ann["id"].values,
        "group": ann[G + "group_name"].values,
        "dx": nsnorm.canonical_diagnosis(ann[G + "diagnosis"]).values,
        "modality": nsnorm.canonical_modality(ann["TaskExtractor.Modality"]).values,
        "resting": ann["TaskExtractor.fMRITasks[0].RestingState"].values,
        "design": ann["TaskExtractor.fMRITasks[0].TaskDesign"].values,
        "n_subj": ann[G + "count"].values,
        "age": ann[G + "age_mean"].values,
        "female": ann[G + "female_count"].values,
        "male": ann[G + "male_count"].values,
    })
    df["domains"] = nsnorm.canonical_domains(
        ann["TaskExtractor.fMRITasks[0].Domain"]).values

    # Screen the demographics; an out-of-range value is a misread, not a datum.
    df.loc[~df.age.between(*AGE_RANGE), "age"] = np.nan
    df.loc[~df.n_subj.between(*COUNT_RANGE), "n_subj"] = np.nan
    total = df.female + df.male
    df["female_frac"] = np.where(total > 0, df.female / total, np.nan)

    for domain in nsnorm.DOMAINS:
        df["dom_" + domain] = df.domains.map(lambda ls, d=domain: d in ls)
    df["n_domains"] = df.domains.map(len)
    return df.drop(columns=["domains"])


def healthy_task_mask(df):
    """Healthy participants, task fMRI-BOLD: the cleanest cognitive sample."""
    return (
        (df.group == "healthy")
        & (df.modality == "fMRI-BOLD")
        & (df.resting == 0.0)
        & (df.n_domains > 0)
    )


def clinical_task_mask(df):
    """Task fMRI-BOLD analyses carrying a healthy/patients label."""
    return (
        df.group.isin(["healthy", "patients"])
        & (df.modality == "fMRI-BOLD")
        & (df.resting == 0.0)
    )


def load_features(name, rows_index=None):
    """Load a cached atlas feature matrix, optionally restricted to rows."""
    X = np.load(f"{W}/feat/{name}.npy", mmap_mode="r")
    return np.asarray(X if rows_index is None else X[rows_index])
