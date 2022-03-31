import requests

from nimare import nimads


def test_load_nimads():
    nimads_data = requests.get("https://neurostore.xyz/api/datasets/5e6u9J4poc38?nested=true").json()
    studyset = nimads.Studyset(nimads_data)
