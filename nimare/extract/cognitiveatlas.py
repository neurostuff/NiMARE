"""The following module is partially derived from cognitiveatlas' code."""
import json
from urllib.request import urlopen

import pandas as pd

API_VERSION = "v-alpha"


def _get_json(url):
    return urlopen(url).read().decode("utf-8")


def _get_df(myjson):
    """Convert json to pandas data frame."""
    try:
        df = pd.DataFrame(myjson)
    except:
        df = []
    return df


# Load a json object
def _parse_json(myjson):
    return json.loads(myjson)


# Data Json Object (from URL)
class _DataJson:
    """DataJson: internal class for storing json, accessed by NeuroVault Object."""

    def __init__(self, url, silent=False):
        if silent is False:
            print(url)
        self.url = url
        self.txt = _get_json(url)
        self.json = _parse_json(self.txt)
        self.pandas = _get_df(self.json)

    """Print json data fields"""

    def __str__(self):
        return "Result Includes:<pandas:data frame><json:dict><txt:str><url:str>"


def _get_concept(id=None, name=None, contrast_id=None, silent=False):
    """get_concept return one or more concepts.

    :param id: Return the specified Concept.
    :param name name: Return the specified Concept.
    :param contrast_id: Return all Concepts related to the specified Contrast.

    [no parameters] - Return all Concepts.

    :Example:

        http://cognitiveatlas.org/api/v-alpha/concept?id=trm_4a3fd79d096be
    """
    base_url = "http://cognitiveatlas.org/api/%s/concept" % (API_VERSION)
    parameters = {"id": id, "name": name, "contrast_id": contrast_id}
    url = _generate_url(base_url, parameters)
    result = _DataJson(url, silent=silent)
    if not silent:
        print(result)
    return result


def _get_task(id=None, name=None, silent=False):
    """get_task return one or more tasks.

    :param id: Return the specified Task.
    :param name name: Return the specified Task.

    [no parameters] - Return all Tasks with basic information only.

    :Example:

        http://cognitiveatlas.org/api/v-alpha/task?id=trm_4f244f46ebf58
    """
    base_url = "http://cognitiveatlas.org/api/%s/task" % (API_VERSION)
    parameters = {"id": id, "name": name}
    url = _generate_url(base_url, parameters)
    result = _DataJson(url, silent=silent)
    if not silent:
        print(result)
    return result


def _get_disorder(id=None, name=None, silent=False):
    """get_disorder return one or more disorders.

    :param id: Return the specified Disorder.
    :param name name: Return the specified Disorder.

    [no parameters] - Return all Tasks with basic information only.
    """
    base_url = "http://cognitiveatlas.org/api/%s/disorder" % (API_VERSION)
    parameters = {"id": id, "name": name}
    url = _generate_url(base_url, parameters)
    result = _DataJson(url, silent=silent)
    if not silent:
        print(result)
    return result


def _generate_url(base_url, parameters):
    """Generate a complete url from a base and list of parameters.

    :param base_url: the base url (string)
    :param parameters: a dictionary with the keys being the parameter, values being the values of
        the parameter. Any values of None will not be added to the url.
    """
    values = [x.replace(" ", "%20") for x in list(parameters.values()) if x]
    keys = [key for (key, value) in list(parameters.items()) if value]
    arguments = ["%s=%s" % (keys[i], values[i]) for i in range(len(values))]
    arguments = "&".join(arguments)
    return "%s?%s" % (base_url, arguments)
