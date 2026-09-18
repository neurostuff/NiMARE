"""Test that NiMARE's stochastic procedures can be seeded and reproduced."""

import numpy as np
import pytest

from nimare.annotate.gclda import GCLDAModel
from nimare.annotate.lda import LDAModel
from nimare.annotate.text import generate_counts
from nimare.correct import FWECorrector
from nimare.meta.cbma.ale import ALE, SCALE, ALESubtraction
from nimare.meta.cbma.mkda import MKDAChi2, MKDADensity
from nimare.utils import _check_random_state, _seed_sequence, vox2mm

UNCORRECTED_NULL = "histweights_corr-none_method-montecarlo"
VOXEL_FWE_NULL = "values_level-voxel_corr-fwe_method-montecarlo"


def test_check_random_state_repeats_a_seeded_sequence():
    """The same seed and stream must always give the same draws."""
    first = _check_random_state(42, stream="null").integers(0, 1000, size=20)
    second = _check_random_state(42, stream="null").integers(0, 1000, size=20)

    np.testing.assert_array_equal(first, second)


def test_check_random_state_streams_are_independent():
    """Two steps of the same seeded object must not reuse one sequence."""
    null = _check_random_state(42, stream="null").integers(0, 1000, size=20)
    fwe = _check_random_state(42, stream="fwe").integers(0, 1000, size=20)

    assert not np.array_equal(null, fwe)


def test_check_random_state_accepts_negative_seeds_and_generators():
    """Negative integers are valid seeds, and a Generator is passed straight through."""
    np.testing.assert_array_equal(
        _check_random_state(-7).integers(0, 1000, size=5),
        _check_random_state(-7).integers(0, 1000, size=5),
    )

    generator = np.random.default_rng(3)
    assert _check_random_state(generator, stream="ignored") is generator


def test_check_random_state_rejects_unusable_seeds():
    """A seed NiMARE cannot interpret should fail loudly rather than silently vary."""
    with pytest.raises(TypeError, match="random_state must be"):
        _check_random_state("not-a-seed")


def test_unseeded_random_state_varies_between_calls():
    """Without a seed, the default behaviour is still to draw fresh randomness."""
    first = _check_random_state(None).integers(0, 2**32, size=20)
    second = _check_random_state(None).integers(0, 2**32, size=20)

    assert not np.array_equal(first, second)


def test_seed_sequence_spawns_distinct_children():
    """Per-iteration seeds must be independent of one another but fixed by the seed."""
    children = [child.entropy for child in _seed_sequence(11, stream="perm").spawn(4)]
    repeat = [child.entropy for child in _seed_sequence(11, stream="perm").spawn(4)]
    keys = [child.spawn_key for child in _seed_sequence(11, stream="perm").spawn(4)]

    assert children == repeat
    assert len(set(keys)) == 4


def test_estimator_streams_do_not_collide(testdata_cbma):
    """An Estimator's uncorrected null and FWE null must draw different permutations."""
    meta = ALE(null_method="montecarlo", n_iters=5, random_state=99)

    null_draws = meta._get_rng("null_montecarlo").integers(0, 1000, size=20)
    fwe_draws = meta._get_rng("correct_fwe_montecarlo").integers(0, 1000, size=20)

    assert not np.array_equal(null_draws, fwe_draws)


def test_iteration_seeds_default_to_iteration_indices():
    """With no seed, per-iteration seeding stays as it was before ``random_state``."""
    meta = ALESubtraction(n_iters=4)

    assert meta._iteration_seeds(4, stream="alesubtraction_permutations") == [0, 1, 2, 3]


def test_iteration_seeds_follow_the_seed():
    """With a seed, per-iteration seeds are reproducible but no longer the indices."""
    meta = ALESubtraction(n_iters=4, random_state=5)

    seeds = meta._iteration_seeds(4, stream="alesubtraction_permutations")
    repeat = meta._iteration_seeds(4, stream="alesubtraction_permutations")

    assert [seed.entropy for seed in seeds] == [seed.entropy for seed in repeat]
    assert [seed.spawn_key for seed in seeds] != [0, 1, 2, 3]


def test_ale_montecarlo_null_is_reproducible(testdata_cbma):
    """A seeded ALE builds the same Monte Carlo null every time."""
    first = ALE(null_method="montecarlo", n_iters=5, random_state=8).fit(testdata_cbma)
    second = ALE(null_method="montecarlo", n_iters=5, random_state=8).fit(testdata_cbma)
    other = ALE(null_method="montecarlo", n_iters=5, random_state=9).fit(testdata_cbma)

    np.testing.assert_array_equal(
        first.estimator.null_distributions_[UNCORRECTED_NULL],
        second.estimator.null_distributions_[UNCORRECTED_NULL],
    )
    np.testing.assert_array_equal(
        first.get_map("z", return_type="array"), second.get_map("z", return_type="array")
    )
    assert not np.array_equal(
        first.estimator.null_distributions_[UNCORRECTED_NULL],
        other.estimator.null_distributions_[UNCORRECTED_NULL],
    )


def test_ale_montecarlo_null_varies_without_a_seed(testdata_cbma):
    """Leaving ``random_state`` unset keeps the historical, unseeded behaviour."""
    first = ALE(null_method="montecarlo", n_iters=5).fit(testdata_cbma)
    second = ALE(null_method="montecarlo", n_iters=5).fit(testdata_cbma)

    assert not np.array_equal(
        first.estimator.null_distributions_[UNCORRECTED_NULL],
        second.estimator.null_distributions_[UNCORRECTED_NULL],
    )


def test_correct_fwe_montecarlo_is_reproducible(testdata_cbma):
    """A seeded Estimator's Monte Carlo FWE correction repeats exactly."""
    corrector = FWECorrector(method="montecarlo", voxel_thresh=0.01, n_iters=5, n_cores=1)

    first = corrector.transform(MKDADensity(random_state=3).fit(testdata_cbma))
    second = corrector.transform(MKDADensity(random_state=3).fit(testdata_cbma))
    other = corrector.transform(MKDADensity(random_state=4).fit(testdata_cbma))

    np.testing.assert_array_equal(
        first.estimator.null_distributions_[VOXEL_FWE_NULL],
        second.estimator.null_distributions_[VOXEL_FWE_NULL],
    )
    assert not np.array_equal(
        first.estimator.null_distributions_[VOXEL_FWE_NULL],
        other.estimator.null_distributions_[VOXEL_FWE_NULL],
    )


def test_scale_is_reproducible(testdata_cbma):
    """A seeded SCALE analysis gives the same map twice."""
    mask_img = testdata_cbma.masker.mask_img
    # Every 500th in-mask voxel: enough of a sampling space for the permutations to differ.
    xyz = vox2mm(np.vstack(np.where(mask_img.get_fdata())).T, mask_img.affine)[::500, :]
    dset = testdata_cbma.slice(testdata_cbma.ids[:5])

    first = SCALE(xyz, n_iters=5, n_cores=1, random_state=1).fit(dset)
    second = SCALE(xyz, n_iters=5, n_cores=1, random_state=1).fit(dset)
    other = SCALE(xyz, n_iters=5, n_cores=1, random_state=2).fit(dset)

    np.testing.assert_array_equal(
        first.get_map("z", return_type="array"),
        second.get_map("z", return_type="array"),
    )
    assert not np.array_equal(
        first.get_map("z", return_type="array"),
        other.get_map("z", return_type="array"),
    )


def test_alesubtraction_random_state_selects_the_permutations(testdata_cbma):
    """Check that ALESubtraction stays reproducible with and without a seed."""
    dset1 = testdata_cbma.slice(testdata_cbma.ids[:4])
    dset2 = testdata_cbma.slice(testdata_cbma.ids[4:8])

    unseeded = ALESubtraction(n_iters=5, n_cores=1).fit(dset1, dset2)
    unseeded_again = ALESubtraction(n_iters=5, n_cores=1).fit(dset1, dset2)
    seeded = ALESubtraction(n_iters=5, n_cores=1, random_state=6).fit(dset1, dset2)
    seeded_again = ALESubtraction(n_iters=5, n_cores=1, random_state=6).fit(dset1, dset2)

    stat = "p_desc-group1MinusGroup2"
    # Unseeded ALESubtraction was already deterministic, and stays that way.
    np.testing.assert_array_equal(
        unseeded.get_map(stat, return_type="array"),
        unseeded_again.get_map(stat, return_type="array"),
    )
    np.testing.assert_array_equal(
        seeded.get_map(stat, return_type="array"),
        seeded_again.get_map(stat, return_type="array"),
    )
    # A seed selects a different set of group assignments than the iteration-index default.
    assert not np.array_equal(
        unseeded.get_map(stat, return_type="array"),
        seeded.get_map(stat, return_type="array"),
    )


def test_mkdachi2_label_permutation_fwe_is_reproducible(testdata_cbma):
    """A seeded MKDAChi2 repeats its label-permutation FWE null."""
    dset1 = testdata_cbma.slice(testdata_cbma.ids[:4])
    dset2 = testdata_cbma.slice(testdata_cbma.ids[4:8])
    corrector = FWECorrector(method="montecarlo", n_iters=5, n_cores=1)
    null_key = "values_desc-pAgF_level-voxel_corr-fwe_method-montecarlo"

    first = corrector.transform(MKDAChi2(random_state=2).fit(dset1, dset2))
    second = corrector.transform(MKDAChi2(random_state=2).fit(dset1, dset2))
    other = corrector.transform(MKDAChi2(random_state=12).fit(dset1, dset2))

    np.testing.assert_array_equal(
        first.estimator.null_distributions_[null_key],
        second.estimator.null_distributions_[null_key],
    )
    assert not np.array_equal(
        first.estimator.null_distributions_[null_key],
        other.estimator.null_distributions_[null_key],
    )


def test_mkdachi2_random_foci_fwe_is_reproducible(testdata_cbma, monkeypatch):
    """The coordinate-permutation FWE null draws the same foci for the same seed.

    The permuted coordinates are inspected directly: on a dataset this small the maximum
    chi-squared value saturates, so the null distribution itself cannot tell two sets of
    permuted foci apart.
    """
    dset1 = testdata_cbma.slice(testdata_cbma.ids[:4])
    dset2 = testdata_cbma.slice(testdata_cbma.ids[4:8])
    corrector = FWECorrector(method="montecarlo", n_iters=3, n_cores=1)

    def _permuted_foci(random_state):
        drawn = []
        meta = MKDAChi2(fwe_null_method="random-foci", random_state=random_state)
        original = MKDAChi2._run_fwe_permutation

        def _record(self, iter_xyz1, iter_xyz2, *args, **kwargs):
            drawn.append(np.asarray(iter_xyz1).copy())
            return original(self, iter_xyz1, iter_xyz2, *args, **kwargs)

        monkeypatch.setattr(MKDAChi2, "_run_fwe_permutation", _record)
        corrector.transform(meta.fit(dset1, dset2))
        return np.stack(drawn)

    np.testing.assert_array_equal(_permuted_foci(2), _permuted_foci(2))
    assert not np.array_equal(_permuted_foci(2), _permuted_foci(12))


def test_lda_random_state_is_reproducible(testdata_laird):
    """A seeded LDA model produces the same topics twice."""

    def _topics(random_state):
        model = LDAModel(n_topics=3, max_iter=5, text_column="abstract", random_state=random_state)
        model.fit(testdata_laird)
        return model.distributions_["p_topic_g_word"]

    np.testing.assert_array_equal(_topics(0), _topics(0))
    assert not np.allclose(_topics(0), _topics(1))


def test_gclda_leaves_the_global_random_state_alone(testdata_laird):
    """Building a GCLDA model must not reseed NumPy for the rest of the session."""
    counts_df = generate_counts(
        testdata_laird.texts, text_column="abstract", tfidf=False, min_df=1, max_df=1.0
    )

    np.random.seed(7)
    expected = np.random.rand(5)

    np.random.seed(7)
    model = GCLDAModel(
        counts_df,
        testdata_laird.coordinates,
        mask=testdata_laird.masker.mask_img,
        n_topics=2,
        n_regions=2,
        symmetric=True,
    )
    model.fit(n_iters=1, loglikely_freq=1)

    np.testing.assert_array_equal(np.random.rand(5), expected)


def test_gclda_seed_init_fixes_the_model(testdata_laird):
    """Two GCLDA models with the same ``seed_init`` agree, and different seeds do not."""
    counts_df = generate_counts(
        testdata_laird.texts, text_column="abstract", tfidf=False, min_df=1, max_df=1.0
    )

    def _fit(seed_init):
        model = GCLDAModel(
            counts_df,
            testdata_laird.coordinates,
            mask=testdata_laird.masker.mask_img,
            n_topics=2,
            n_regions=2,
            symmetric=True,
            seed_init=seed_init,
        )
        model.fit(n_iters=2, loglikely_freq=2)
        return model.p_word_g_topic_

    np.testing.assert_array_equal(_fit(1), _fit(1))
    assert not np.array_equal(_fit(1), _fit(2))
