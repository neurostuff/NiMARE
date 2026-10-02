"""Test nimare.decode.discrete.

Tests for nimare.decode.discrete.gclda_decode_roi are in test_annotate_gclda.
"""

import numpy as np
import pandas as pd
import pytest

from nimare.decode import discrete
from nimare.transforms import chi2_to_nlogp, nlogp_to_z


def test_neurosynth_decode(testdata_laird):
    """Smoke test for discrete.neurosynth_decode."""
    ids = testdata_laird.ids[:5]
    features = testdata_laird.annotations.columns.tolist()[5:10]
    decoded_df = discrete.neurosynth_decode(
        testdata_laird.coordinates,
        testdata_laird.annotations,
        ids=ids,
        features=features,
        correction=None,
    )
    assert isinstance(decoded_df, pd.DataFrame)


def test_brainmap_decode(testdata_laird):
    """Smoke test for discrete.brainmap_decode."""
    ids = testdata_laird.ids[:5]
    features = testdata_laird.annotations.columns.tolist()[5:10]
    decoded_df = discrete.brainmap_decode(
        testdata_laird.coordinates,
        testdata_laird.annotations,
        ids=ids,
        features=features,
        correction=None,
    )
    assert isinstance(decoded_df, pd.DataFrame)


def test_NeurosynthDecoder(testdata_laird):
    """Smoke test for discrete.NeurosynthDecoder."""
    ids = testdata_laird.ids[:5]
    labels = testdata_laird.get_labels(ids=testdata_laird.ids)
    decoder = discrete.NeurosynthDecoder(features=labels)
    decoder.fit(testdata_laird)
    decoded_df = decoder.transform(ids=ids)
    assert isinstance(decoded_df, pd.DataFrame)
    assert decoded_df.shape == (len(labels), 6)


def test_NeurosynthDecoder_featuregroup(testdata_laird):
    """Smoke test for discrete.NeurosynthDecoder with feature group selection."""
    ids = testdata_laird.ids[:5]
    decoder = discrete.NeurosynthDecoder(feature_group="Neurosynth_TFIDF")
    decoder.fit(testdata_laird)
    decoded_df = decoder.transform(ids=ids)
    assert isinstance(decoded_df, pd.DataFrame)


def test_NeurosynthDecoder_featuregroup_failure(testdata_laird):
    """Smoke test for NeurosynthDecoder with feature group selection and no detected features."""
    decoder = discrete.NeurosynthDecoder(feature_group="Neurosynth_TFIDF", features=["01", "05"])
    with pytest.raises(Exception):
        decoder.fit(testdata_laird)


def test_BrainMapDecoder(testdata_laird):
    """Smoke test for discrete.BrainMapDecoder."""
    ids = testdata_laird.ids[:5]
    labels = testdata_laird.get_labels(ids=testdata_laird.ids)
    decoder = discrete.BrainMapDecoder(features=labels)
    decoder.fit(testdata_laird)
    decoded_df = decoder.transform(ids=ids)
    assert isinstance(decoded_df, pd.DataFrame)
    assert decoded_df.shape == (len(labels), 6)


def test_BrainMapDecoder_failure(testdata_laird):
    """Smoke test for discrete.BrainMapDecoder where there are no features left."""
    decoder = discrete.BrainMapDecoder(features=["doggy"])
    with pytest.raises(Exception):
        decoder.fit(testdata_laird)


def test_ROIAssociationDecoder(testdata_laird, roi_img):
    """Smoke test for discrete.ROIAssociationDecoder."""
    labels = testdata_laird.get_labels(ids=testdata_laird.ids)
    decoder = discrete.ROIAssociationDecoder(masker=roi_img, features=labels)
    decoder.fit(testdata_laird)
    decoded_df = decoder.transform()
    assert isinstance(decoded_df, pd.DataFrame)
    assert decoded_df.shape == (len(labels), 1)


def test_brainmap_decode_forward_z_is_one_tailed_and_unsigned(testdata_laird):
    """Forward inference is ``binom.logsf``, an upper tail, so its z has no lower side.

    Regression test: the one-sided p-value was converted with a two-tailed rule and then
    multiplied by a sign taken from the mean label count, which could report a
    significant *depletion* that the test never tested for.
    """
    ids = testdata_laird.ids[:5]
    features = testdata_laird.annotations.columns.tolist()[5:10]
    decoded_df = discrete.brainmap_decode(
        testdata_laird.coordinates,
        testdata_laird.annotations,
        ids=ids,
        features=features,
        correction=None,
    )

    finite = decoded_df["zForward"].dropna()
    assert (finite >= 0).all()

    expected = nlogp_to_z(np.log(decoded_df["pForward"].values), "one")
    np.testing.assert_allclose(decoded_df["zForward"].values, expected, rtol=1e-6)


def _saturated_label_inputs():
    """One label whose every focus falls inside the selection, and one carried by nobody."""
    ids = [f"s{i:02d}" for i in range(40)]
    coordinates = pd.DataFrame({"id": ids, "x": 0.0, "y": 0.0, "z": 0.0, "space": "MNI"})
    annotations = pd.DataFrame(
        {
            "id": ids,
            "common": [1] * 30 + [0] * 10,
            "absent": [0] * 39 + [1],  # carried only by an unselected study
            "saturated": [1] * 8 + [0] * 32,  # every carrier is selected
        }
    )
    return coordinates, annotations, ids[:20]


def test_brainmap_decode_forward_p_is_the_inclusive_upper_tail():
    """The one-sided p-value for observing k is P(X >= k), i.e. ``logsf(k - 1)``.

    Regression test: ``logsf(k)`` is P(X > k), which excludes the observation. A label whose
    every focus falls inside the selection then gets P(X > n) == 0 and an infinite z.
    """
    coordinates, annotations, selected = _saturated_label_inputs()
    decoded_df = discrete.brainmap_decode(
        coordinates,
        annotations,
        ids=selected,
        features=["common", "absent", "saturated"],
        correction=None,
    )

    assert np.isfinite(decoded_df["zForward"]).all()
    # All 8 carriers of 'saturated' are selected, and p_selected is 0.5, so P(X >= 8) = 0.5 ** 8.
    assert decoded_df.loc["saturated", "pForward"] == pytest.approx(0.5**8)
    # A label no selected study carries is never evidence of enrichment.
    assert decoded_df.loc["absent", "pForward"] == pytest.approx(1.0)
    assert decoded_df.loc["absent", "zForward"] == pytest.approx(0.0)


def test_brainmap_decode_zero_count_is_never_enrichment():
    """``logsf(-1)`` is log(1), so an unobserved label needs no special case."""
    from scipy.stats import binom

    assert binom.logsf(k=-1, n=1, p=0.001) == 0.0
    assert binom.logsf(k=-1, n=50, p=0.3) == 0.0


def _rare_label_inputs():
    """Build inputs where one label is far rarer than the average selected-label count.

    The rare label is what used to drive the one-way chi-squared statistic negative: its
    database-wide count sat below the mean number of selected studies per label.
    """
    n_studies = 60
    ids = [f"study-{i:02d}" for i in range(n_studies)]
    coordinates = pd.DataFrame(
        {"id": ids, "x": 0.0, "y": 0.0, "z": 0.0, "space": "MNI"},
    )
    # Two common labels carried by most studies, and one label carried by a single study.
    annotations = pd.DataFrame(
        {
            "id": ids,
            "common1": [1] * 50 + [0] * 10,
            "common2": [1] * 40 + [0] * 20,
            "rare": [1] + [0] * (n_studies - 1),
        }
    )
    return coordinates, annotations, ids


def test_neurosynth_decode_rare_label_is_finite():
    """A label rarer than the mean selected count must not yield a negative chi-squared.

    Regression test: ``one_way`` was called with the per-label database count as ``n``
    rather than the number of selected studies, so ``n - expected`` went negative for rare
    labels and the statistic came back NaN.
    """
    coordinates, annotations, ids = _rare_label_inputs()
    decoded_df = discrete.neurosynth_decode(
        coordinates,
        annotations,
        ids=ids[:30],
        features=["common1", "common2", "rare"],
        correction=None,
        min_studies=None,
    )
    assert decoded_df["pForward"].notna().all()
    assert decoded_df["zForward"].notna().all()


def test_neurosynth_decode_one_nan_does_not_erase_the_correction():
    """With BH correction on, every label still gets a finite corrected p-value."""
    coordinates, annotations, ids = _rare_label_inputs()
    decoded_df = discrete.neurosynth_decode(
        coordinates,
        annotations,
        ids=ids[:30],
        features=["common1", "common2", "rare"],
        correction="bh",
        min_studies=None,
    )
    assert decoded_df[["pForward", "zForward", "pReverse", "zReverse"]].notna().all().all()


def test_neurosynth_decode_forward_matches_the_reference_formula():
    """The forward statistic is a one-sample chi-squared on ``n_selected`` trials."""
    coordinates, annotations, ids = _rare_label_inputs()
    features = ["common1", "common2", "rare"]
    selected = ids[:30]
    decoded_df = discrete.neurosynth_decode(
        coordinates,
        annotations,
        ids=selected,
        features=features,
        correction=None,
        min_studies=None,
    )

    n_selected = len(selected)
    observed = annotations.set_index("id").loc[selected, features].ge(0.001).sum(axis=0).values
    expected = observed.mean()
    # The two-cell chi-squared written out longhand, as Neurosynth's stats.one_way has it.
    chi2 = (observed - expected) ** 2 / expected + (
        (n_selected - observed) - (n_selected - expected)
    ) ** 2 / (n_selected - expected)
    np.testing.assert_allclose(
        decoded_df["pForward"].values, np.exp(chi2_to_nlogp(chi2, 1)), rtol=1e-10
    )


def test_neurosynth_decode_min_studies_drops_rare_labels():
    """``min_studies`` removes labels below the floor, as a count or as a proportion."""
    coordinates, annotations, ids = _rare_label_inputs()
    features = ["common1", "common2", "rare"]
    kwargs = dict(ids=ids[:30], features=features, correction=None)

    kept_all = discrete.neurosynth_decode(coordinates, annotations, min_studies=1, **kwargs)
    assert sorted(kept_all.index) == sorted(features)

    by_count = discrete.neurosynth_decode(coordinates, annotations, min_studies=5, **kwargs)
    assert "rare" not in by_count.index
    assert sorted(by_count.index) == ["common1", "common2"]

    # 0.03 of 60 studies is 1.8, so the single-study label falls below the floor.
    by_proportion = discrete.neurosynth_decode(
        coordinates, annotations, min_studies=0.03, **kwargs
    )
    assert "rare" not in by_proportion.index


def test_neurosynth_decode_min_studies_excluding_everything_raises():
    """A floor no label can reach is an error rather than an empty table."""
    coordinates, annotations, ids = _rare_label_inputs()
    with pytest.raises(ValueError, match="No labels reach min_studies"):
        discrete.neurosynth_decode(
            coordinates,
            annotations,
            ids=ids[:30],
            features=["common1", "common2", "rare"],
            correction=None,
            min_studies=1000,
        )


@pytest.mark.parametrize("carried_by_all", [False, True])
def test_neurosynth_decode_degenerate_selection_is_finite(carried_by_all):
    """Every label absent from, or present in, every selected study still decodes.

    Both put the expected count at a bound, which used to make the uniformity statistic
    0/0 and hand back NaN for every label.
    """
    n_studies = 40
    ids = [f"s{i:02d}" for i in range(n_studies)]
    coordinates = pd.DataFrame({"id": ids, "x": 0.0, "y": 0.0, "z": 0.0, "space": "MNI"})
    selected, unselected = ids[:20], ids[20:]
    if carried_by_all:
        values = [1] * len(selected) + [0] * len(unselected)
    else:
        values = [0] * len(selected) + [1] * len(unselected)
    annotations = pd.DataFrame({"id": ids, "a": values, "b": values})

    decoded_df = discrete.neurosynth_decode(
        coordinates, annotations, ids=selected, features=["a", "b"], correction=None
    )

    assert decoded_df["pForward"].notna().all()
    assert decoded_df["zForward"].notna().all()
    # No label deviates from the average label, so the uniformity test has nothing to report.
    np.testing.assert_allclose(decoded_df["pForward"].values, 1.0)
    np.testing.assert_allclose(decoded_df["zForward"].values, 0.0)


def test_neurosynth_decode_min_studies_is_relative_to_the_analysis_universe():
    """``ids2`` narrows the universe, and a proportional floor narrows with it.

    Every other quantity in the function -- ``n_term``, ``p_term``, both chi-squared
    tests -- ignores studies outside ``ids`` plus ``ids2``, and Neurosynth's
    ``MetaAnalysis`` scales its own ``min_studies`` by the same union.
    """
    ids_all = [f"s{i:03d}" for i in range(100)]
    coordinates = pd.DataFrame({"id": ids_all, "x": 0.0, "y": 0.0, "z": 0.0, "space": "MNI"})
    # 'edge' appears in 2 studies: one selected, one in ids2. That is 5% of the 40-study
    # universe but only 2% of the 100-study database.
    values = [0] * 100
    values[0] = values[20] = 1
    annotations = pd.DataFrame({"id": ids_all, "edge": values, "filler": [1] * 100})
    selected, ids2 = ids_all[:20], ids_all[20:40]
    kwargs = dict(ids=selected, ids2=ids2, features=["edge", "filler"], correction=None)

    # 3% of the universe is 1.2 studies, so 'edge' clears it on 2.
    kept = discrete.neurosynth_decode(coordinates, annotations, min_studies=0.03, **kwargs)
    assert "edge" in kept.index
    # 6% of the universe is 2.4 studies, so it does not.
    dropped = discrete.neurosynth_decode(coordinates, annotations, min_studies=0.06, **kwargs)
    assert "edge" not in dropped.index
