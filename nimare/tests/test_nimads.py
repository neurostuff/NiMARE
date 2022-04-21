import requests

from nimare import nimads


def test_load_nimads():
    nimads_data = requests.get(
        "https://neurostore.xyz/api/studysets/78rWEjjjuC65?nested=true"
    ).json()
    studyset = nimads.Studyset(nimads_data)
    assert isinstance(studyset, nimads.Studyset)

    annotation_data = requests.get("https://neurostore.xyz/api/annotations/4aLPSznu6jJa").json()

    annotation = nimads.Annotation(annotation_data)

    studyset.annotations = annotation
