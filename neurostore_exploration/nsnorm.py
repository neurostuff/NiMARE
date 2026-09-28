"""Normalization helpers for the NeuroStore nightly release.

The release carries LLM-extracted annotations, so its controlled vocabularies
have drifted and its free-text fields are unnormalized. This module folds them
back to canonical values and screens the coordinates.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------- coordinates
# A generous box around MNI152; anything outside is a transcription error, not
# a brain location.
MNI_BOX = dict(x=(-90, 90), y=(-130, 90), z=(-80, 100))


def flag_coordinates(coordinates):
    """Label each focus and each analysis with the quality problems it has.

    Returns ``(foci, analyses)``: the coordinate frame with an ``out_of_box``
    column, and a per-analysis frame with ``n_foci``, ``frac_bad`` and
    ``voxel_indices`` (every focus non-negative and under 200, i.e. array
    indices reported as millimetres).
    """
    co = coordinates.copy()
    co["out_of_box"] = (
        (co.x < MNI_BOX["x"][0]) | (co.x > MNI_BOX["x"][1])
        | (co.y < MNI_BOX["y"][0]) | (co.y > MNI_BOX["y"][1])
        | (co.z < MNI_BOX["z"][0]) | (co.z > MNI_BOX["z"][1])
    )
    xyz = co[["x", "y", "z"]]
    co["index_like"] = ((xyz >= 0) & (xyz <= 200)).all(axis=1)

    per = co.groupby("contrast_id").agg(
        n_foci=("x", "size"),
        frac_bad=("out_of_box", "mean"),
        voxel_indices=("index_like", "all"),
    )
    # Index-like only matters when the analysis is otherwise implausible: a
    # real analysis can legitimately report only right-hemisphere positives.
    per["voxel_indices"] &= per.frac_bad > 0.5
    return co, per


def clean_coordinates(coordinates, drop_duplicates=True):
    """Drop out-of-box foci, analyses reporting voxel indices, and repeats."""
    co, per = flag_coordinates(coordinates)
    suspect = set(per.index[per.voxel_indices])
    clean = co[~co.out_of_box & ~co.contrast_id.isin(suspect)]
    if drop_duplicates:
        clean = clean.drop_duplicates(subset=["contrast_id", "x", "y", "z"])
    return clean.drop(columns=["out_of_box", "index_like"])


# ------------------------------------------------------------------- domains
# The extractor was given a fixed domain list but drifted off it; these are the
# off-vocabulary values and the canonical label each belongs to.
DOMAIN_CANON = {
    "Emotion": "Emotion",
    "Affective": "Emotion",
    "Learning and memory": "Learning and memory",
    "Memory": "Learning and memory",
    "Executive cognitive control": "Executive cognitive control",
    "Cognitive control": "Executive cognitive control",
    "Cognition": "Executive cognitive control",
    "Cognitive": "Executive cognitive control",
    "Perception": "Perception",
    "Somatosensory": "Perception",
    "Somatic sensation": "Perception",
    "Pain": "Perception",
    "Interoception": "Perception",
    "Music": "Perception",
    "Attention": "Attention",
    "Spatial orientation": "Attention",
    "Social function": "Social function",
    "Language": "Language",
    "Communication": "Language",
    "Reasoning and decision making": "Reasoning and decision making",
    "Decision making": "Reasoning and decision making",
    "Creativity": "Reasoning and decision making",
    "Action": "Action",
    "Motor control": "Action",
    "Motor": "Action",
    "Motor function": "Action",
    "Repetitive behaviors": "Action",
    "Motivation": "Motivation",
}

DOMAINS = [
    "Emotion", "Learning and memory", "Executive cognitive control", "Perception",
    "Attention", "Social function", "Language", "Reasoning and decision making",
    "Action", "Motivation",
]

MODALITY_CANON = {
    "fMRI-BOLD": "fMRI-BOLD", "fMRI": "fMRI-BOLD", "rsfMRI": "fMRI-BOLD",
    "StructuralMRI": "StructuralMRI", "MRI": "StructuralMRI",
    "DiffusionMRI": "DiffusionMRI", "DTI": "DiffusionMRI",
    "fMRI-CBF": "Perfusion", "ASL": "Perfusion", "fMRI-CBV": "Perfusion",
    "EEG": "Electrophysiology", "ERP": "Electrophysiology", "MEG": "Electrophysiology",
    "iEEG": "Electrophysiology", "ECoG": "Electrophysiology", "LFP": "Electrophysiology",
    "EROS": "Electrophysiology",
    "fNIRS": "Optical", "NIRS": "Optical",
    "MRS": "MRS", "H-MRS": "MRS", "P MRS": "MRS",
    "TMS": "Stimulation",
    "SPECT": "SPECT",
}


def parse_labels(value):
    """Return the atomic labels in a JSON-list-valued annotation cell."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value)
    if text.startswith("["):
        try:
            return [str(item) for item in json.loads(text)]
        except (ValueError, TypeError):
            pass
    return [text]


def canonical_domains(series):
    """Map a Domain column to sorted lists of canonical domain labels."""
    def one(value):
        seen = {DOMAIN_CANON.get(label) for label in parse_labels(value)}
        return sorted(label for label in seen if label)
    return series.map(one)


def canonical_modality(series):
    """Map a Modality column to its primary canonical modality."""
    def one(value):
        for label in parse_labels(value):
            canon = MODALITY_CANON.get(label)
            if canon:
                return canon
        labels = parse_labels(value)
        return labels[0] if labels else None
    return series.map(one)


# ----------------------------------------------------------------- diagnosis
# Ordered: the first pattern that matches wins, so specific entities are listed
# before the broader families that would also match them.
DIAGNOSIS_PATTERNS = [
    ("Healthy", r"\b(healthy|normal|control|non-?clinical|typically\s+developing|"
                r"neurotypical)\b|\bno\s+(known\s+|current\s+|history\s+of\s+|"
                r"significant\s+)*(psychiatric|neurologic|medical|history)"),
    ("Alzheimer's disease", r"alzheimer|\bad\b(?!hd)|dementia\s+of\s+the\s+alzheimer"),
    ("Mild cognitive impairment", r"mild\s+cognitive\s+impairment|\bmci\b|\bamci\b"),
    ("Parkinson's disease", r"parkinson|\bpd\b"),
    ("Schizophrenia", r"schizophren|\bsz\b|first-?episode\s+psychosis|psychosis|psychotic"),
    ("Bipolar disorder", r"bipolar|\bbd\b|manic"),
    ("Major depression", r"depress|\bmdd\b|dysthymi"),
    ("Anxiety disorders", r"anxiet|\bgad\b|panic\s+disorder|phobi|social\s+anxiety"),
    ("PTSD", r"post-?traumatic|\bptsd\b"),
    ("OCD", r"obsessive|\bocd\b"),
    ("ADHD", r"attention[-\s]?deficit|\badhd\b|\baddh\b"),
    ("Autism spectrum", r"autis|\basd\b|asperger"),
    ("Substance use", r"depend|addict|abuse|alcohol|cocaine|nicotine|smok|cannabis|"
                      r"opioid|heroin|methamphetamine|gambling|gaming\s+disorder|"
                      r"internet\s+addiction"),
    ("Eating disorders", r"anorexi|bulimi|binge|eating\s+disorder"),
    ("Obesity / metabolic", r"obes|overweight|diabet|metabolic\s+syndrome"),
    ("Personality disorders", r"personality\s+disorder|\bbpd\b|psychopath"),
    ("Epilepsy", r"epileps|seizure|temporal\s+lobe\s+epilep"),
    ("Multiple sclerosis", r"multiple\s+sclerosis|\bms\b(?!\s*=)"),
    ("Stroke / vascular", r"stroke|infarct|aphasi|cerebrovascular|moyamoya|hemorrhag"),
    ("Traumatic brain injury", r"traumatic\s+brain|\btbi\b|concussion"),
    ("Chronic pain", r"pain|fibromyalgi|migrain|headache"),
    ("Sleep disorders", r"sleep|insomni|apnea|narcoleps"),
    ("Tinnitus / sensory", r"tinnitus|\bdeaf|blind|amblyopia|hearing\s+loss"),
    ("Dyslexia / learning", r"dyslex|learning\s+disab|dyscalculi|developmental\s+language"),
    ("Tourette / tic", r"tourette|\btic\b"),
    ("Huntington's disease", r"huntington"),
    ("ALS / motor neuron", r"amyotrophic|\bals\b|motor\s+neuron"),
    ("Frontotemporal dementia", r"frontotemporal|\bftd\b|semantic\s+dementia"),
    ("Tumour", r"tumou?r|glioma|glioblastoma|meningioma"),
    ("Brain injury / lesion", r"lesion|amnesi"),
]

_COMPILED = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in DIAGNOSIS_PATTERNS]


def canonical_diagnosis(series):
    """Fold free-text diagnosis strings onto a coarse diagnostic vocabulary."""
    cache = {}

    def one(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        text = str(value).strip()
        if text not in cache:
            match = next((name for name, pat in _COMPILED if pat.search(text)), "Other")
            cache[text] = match
        return cache[text]

    return series.map(one)
