"""Perform neurosynth-compose meta-analyses."""
from importlib import import_module

import requests

from ..io import convert_nimads_to_dataset
from ..nimads import Annotation, Studyset

COMPOSE_URL = "https://compose.neurosynth.org"
STORE_URL = "https://neurostore.org"


def compose_workflow(meta_id):
    """Run meta-analysis.

    Parameters
    ----------
    meta_id: str
        id corresponding to neurosynth
    """
    data = requests.get(f"{COMPOSE_URL}/api/meta-analyses/{meta_id}?nested=true").json()
    dset = load_meta_analysis(data["studyset"], data.get("annotation"))
    workflow = load_specification(data["specification"])

    return workflow(dset)


def load_meta_analysis(studyset, annotation=None):
    """Download requisite data and load it into nimare."""
    if not studyset["snapshot"]:
        ss = Studyset(
            requests.get(
                f"{STORE_URL}/api/studysets/{studyset['neurostore_id']}?nested=true"
            ).json()
        )
    else:
        ss = Studyset(studyset["snapshot"])

    if annotation:
        if not annotation["snapshot"]:
            annot = Annotation(
                requests.get(f"{STORE_URL}/api/annotations/{annotation['neurostore_id']}").json()
            )
        else:
            annot = Annotation(annotation["snapshot"])
    else:
        annot = None

    return convert_nimads_to_dataset(ss, annotation=annot)


def load_specification(spec):
    """Returns function to run analysis on dataset."""
    est_mod = import_module(".".join(["nimare", "meta", spec["type"].lower()]))
    estimator = getattr(est_mod, spec["estimator"]["type"])
    if spec["estimator"].get("args"):
        if spec["estimator"]["args"].get("**kwargs") is not None:
            for k, v in spec["estimator"]["args"]["**kwargs"].items():
                spec["estimator"]["args"][k] = v
            spec["estimator"]["args"].pop("**kwargs")
        estimator_init = estimator(**spec["estimator"]["args"])
    else:
        estimator_init = estimator()

    if spec.get("corrector"):
        cor_mod = import_module(".".join(["nimare", "correct"]))
        corrector = getattr(cor_mod, spec["corrector"]["type"])
        corrector_args = spec["corrector"].get("args")
        if corrector_args:
            if corrector_args.get("**kwargs") is not None:
                for k, v in corrector_args["**kwargs"].items():
                    corrector_args[k] = v
                corrector_args.pop("**kwargs")
            corrector_init = corrector(**corrector_args)
        else:
            corrector_init = corrector()
    else:
        corrector_init = None

    if corrector_init:
        return lambda dset: corrector_init.transform(estimator_init.fit(dset))
    else:
        return lambda dset: estimator_init.fit(dset)


def filter_analyses(specification, annotation):
    """Filter which analyses should be run in this meta-analysis."""
    column = specification["filter"]
    keep_ids = []
    for annot in annotation["notes"]:
        if annot["note"].get(column):
            keep_ids.append(f"{annot['study']}-{annot['analysis']}")
    return keep_ids
