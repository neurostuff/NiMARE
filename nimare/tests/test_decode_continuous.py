"""Test nimare.decode.continuous.

Tests for nimare.decode.continuous.gclda_decode_map are in test_annotate_gclda.
"""

import os

import numpy as np
import pandas as pd
import pytest

from nimare.decode import continuous
from nimare.meta import kernel, mkda


def test_CorrelationDecoder_smoke(testdata_laird, tmp_path_factory):
    """Smoke test for continuous.CorrelationDecoder."""
    testdata_laird = testdata_laird.copy()
    features = testdata_laird.get_labels(ids=testdata_laird.ids[0])[:5]

    # Make an image to decode
    meta = mkda.KDA(null_method="approximate")
    res = meta.fit(testdata_laird)
    img = res.get_map("stat")

    # Test: train decoder with Dataset object
    decoder = continuous.CorrelationDecoder(features=features)
    decoder.fit(testdata_laird)
    decoded_df = decoder.transform(img)

    assert isinstance(decoded_df, pd.DataFrame)

    # Get features and images to compare with other methods
    features = list(decoder.results_.maps.keys())
    images = np.array(list(decoder.results_.maps.values()))

    # Save images to disk to test load_imgs method
    tmpdir = tmp_path_factory.mktemp("test_CorrelationDecoder")
    decoder.results_.save_maps(tmpdir)

    # Test: load pregenerated maps using a dictionary of feature names and paths
    img_dict = {feature: os.path.join(tmpdir, f"{feature}.nii.gz") for feature in features}
    decoder2 = continuous.CorrelationDecoder()
    decoder2.load_imgs(img_dict, mask=testdata_laird.masker)
    decoded2_df = decoder2.transform(img)

    features2 = list(decoder2.results_.maps.keys())
    images2 = np.array(list(decoder2.results_.maps.values()))

    assert isinstance(decoded2_df, pd.DataFrame)
    assert np.array_equal(features, features2)
    assert np.array_equal(images, images2)
    assert decoded_df.equals(decoded2_df)

    # Test: load pregenerated maps from a directory
    decoder3 = continuous.CorrelationDecoder()
    decoder3.load_imgs(tmpdir.as_posix(), mask=testdata_laird.masker)
    decoded3_df = decoder3.transform(img)

    features3 = list(decoder3.results_.maps.keys())
    images3 = np.array(list(decoder3.results_.maps.values()))

    assert isinstance(decoded3_df, pd.DataFrame)
    assert np.array_equal(features, features3)
    assert np.array_equal(images, images3)
    assert decoded_df.equals(decoded3_df)

    # Test: passing a dataset to load_imgs
    decoder4 = continuous.CorrelationDecoder()
    with pytest.raises(ValueError):
        decoder4.load_imgs(testdata_laird, mask=testdata_laird.masker)

    # Test: try loading pregenerated maps without a masker
    decoder5 = continuous.CorrelationDecoder()
    with pytest.raises(ValueError):
        decoder5.load_imgs(img_dict)

    # Test: try transforming an image without fitting the decoder
    decoder6 = continuous.CorrelationDecoder()
    with pytest.raises(AttributeError):
        decoder6.transform(img)


def test_CorrelationDistributionDecoder_smoke(testdata_laird, tmp_path_factory):
    """Smoke test for continuous.CorrelationDistributionDecoder."""
    tmpdir = tmp_path_factory.mktemp("test_CorrelationDistributionDecoder")

    testdata_laird = testdata_laird.copy()
    dset = testdata_laird.copy()
    features = testdata_laird.get_labels(ids=testdata_laird.ids[0])[:5]

    decoder = continuous.CorrelationDistributionDecoder(features=features)

    # No images of the requested type
    with pytest.raises(ValueError):
        decoder.fit(testdata_laird)

    # Let's add the path
    testdata_laird.update_path(tmpdir)

    # Then let's make some images to decode
    kern = kernel.MKDAKernel(r=10, value=1)
    kern._infer_names()  # Determine MA map filenames

    imgs = kern.transform(testdata_laird, return_type="image")
    for i_img, img in enumerate(imgs):
        id_ = testdata_laird.ids[i_img]
        out_file = os.path.join(testdata_laird.basepath, kern.filename_pattern.format(id=id_))

        # Add file names to dset.images DataFrame
        img.to_filename(out_file)
        dset.images.loc[testdata_laird.images["id"] == id_, kern.image_type] = out_file

    # And now we have images we can use for decoding!
    decoder = continuous.CorrelationDistributionDecoder(
        features=features,
        target_image=kern.image_type,
    )
    decoder.fit(dset)

    # Make an image to decode
    meta = mkda.KDA(null_method="approximate")
    res = meta.fit(testdata_laird)
    img = res.get_map("stat")
    decoded_df = decoder.transform(img)

    assert isinstance(decoded_df, pd.DataFrame)

    # Test: try transforming an image without fitting the decoder
    decoder2 = decoder = continuous.CorrelationDistributionDecoder()
    with pytest.raises(AttributeError):
        decoder2.transform(img)
