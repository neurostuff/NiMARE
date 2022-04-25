from importlib import import_module

import requests

from ..io import convert_nimads_to_dataset
from ..nimads import Studyset, Annotation

COMPOSE_URL = "https://synth.neurostore.xyz"
STORE_URL = "https://neurostore.xyz"


def run(meta_id):
    """run meta-analysis

    Parameters
    ----------
    meta_id: str
        id corresponding to neurosynth
    """
    data = requests.get(f"{COMPOSE_URL}/api/meta-analyses/{meta_id}?nested=true").json()
    dset = load_meta_analysis(data['studyset'], data.get('annotation'))
    workflow = load_specification(data['specification'])

    return workflow(dset)


def load_meta_analysis(studyset, annotation=None):
    """download requisite data and load it into nimare"""
    if not studyset['snapshot']:
        ss = Studyset(
            requests.get(
                f"{STORE_URL}/api/studysets/{studyset['neurostore_id']}?nested=true"
            ).json()
        )
    else:
        ss = Studyset(studyset['snapshot'])

    if annotation:
        if not annotation['snapshot']:
            annot = Annotation(
                requests.get(
                    f"{STORE_URL}/api/annotations/{annotation['neurostore_id']}"
                ).json()
            )
        else:
            annot = Annotation(annotation['snapshot'])
    else:
        annot = None

    return convert_nimads_to_dataset(ss, annotation=annot)


def load_specification(spec):
    """returns function to run analysis on dataset"""
    est_mod = import_module('.'.join(['nimare', 'meta', spec['type'].lower()]))
    estimator = getattr(est_mod, spec['estimator']['type'])
    if spec['estimator'].get('args'):
        estimator_init = estimator(**spec['estimator']['args'])
    else:
        estimator_init = estimator()

    if spec.get('corrector'):
        cor_mod = import_module('.'.join(['nimare', 'correct']))
        corrector = getattr(cor_mod, spec['corrector']['type'])
        corrector_args = spec['corrector'].get('args')
        if corrector_args:
            corrector_init = corrector(**corrector_args)
        else:
            corrector_init = None

    if corrector_init:
        return lambda dset: corrector_init.transform(estimator_init.fit(dset))
    else:
        return lambda dset: estimator_init.fit(dset)


def filter_analyses(specification, annotation):
    column = specification['filter']
    keep_ids = []
    for annot in annotation['notes']:
        if annot['note'].get(column):
            keep_ids.append(f"{annot['study']}-{annot['analysis']}")
    return keep_ids