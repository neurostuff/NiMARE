"""Utilities for coordinate-based meta-analysis estimators."""
import logging
import os

import nibabel as nib
import numpy as np
import numpy.linalg as npl
import sparse
from scipy import ndimage

from nimare import references
from nimare.due import due
from nimare.extract import download_peaks2maps_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
LGR = logging.getLogger(__name__)


def model_fn(features, labels, mode, params):
    """Run model function used internally by peaks2maps.

    .. deprecated:: 0.0.11
        `model_fn` will be removed in NiMARE 0.0.13.

    .. versionadded:: 0.0.4

    """
    import tensorflow as tf
    from tensorflow.python.estimator.export.export_output import PredictOutput

    ngf = 64
    layers = []

    training_flag = mode == tf.estimator.ModeKeys.TRAIN

    input_images_placeholder = tf.expand_dims(features, -1)

    conv_args = {
        "strides": 2,
        "kernel_size": 4,
        "padding": "valid",
        "activation": tf.nn.leaky_relu,
        "kernel_initializer": tf.random_normal_initializer(0, 0.02),
        "name": "conv",
        "use_bias": False,
    }

    deconv_args = conv_args.copy()
    deconv_args["padding"] = "same"
    deconv_args["name"] = "deconv"

    batchnorm_args = {
        "scale": True,
        "gamma_initializer": tf.random_normal_initializer(1.0, 0.02),
        "center": True,
        "beta_initializer": tf.zeros_initializer(),
        "name": "batchnorm",
        "training": training_flag,
    }

    def pad_and_conv(input, out_channels, conv_args):
        padded_input = tf.pad(input, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        convolved = tf.compat.v1.layers.conv3d(padded_input, out_channels, **conv_args)
        return convolved

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.compat.v1.variable_scope("encoder_1"):
        this_args = conv_args.copy()
        output = pad_and_conv(input_images_placeholder, ngf, this_args)
        layers.append(output)

    layer_specs = [
        (ngf * 2, 0.2),
        # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        (ngf * 2, 0.2),
        # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        (ngf * 4, 0.2),
        # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        (ngf * 8, 0.2),
        # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        # ngf * 8,
        # # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
    ]

    for out_channels, dropout in layer_specs:
        with tf.compat.v1.variable_scope("encoder_%d" % (len(layers) + 1)):
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2,
            # out_channels]
            convolved = pad_and_conv(layers[-1], out_channels, conv_args)
            output = tf.compat.v1.layers.batch_normalization(convolved, **batchnorm_args)
            if dropout > 0.0:
                output = tf.compat.v1.layers.dropout(output, rate=dropout, training=training_flag)
            layers.append(output)

    layer_specs = [
        # (ngf * 8, 0.5),
        # # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (ngf * 8, 0.2),
        # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (ngf * 4, 0.2),
        # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (ngf * 2, 0.2),
        # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (ngf * 2, 0.2),
        # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.compat.v1.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=4)

            output = tf.compat.v1.layers.conv3d_transpose(input, out_channels, **deconv_args)
            output = tf.compat.v1.layers.batch_normalization(output, **batchnorm_args)

            if dropout > 0.0:
                output = tf.compat.v1.layers.dropout(output, rate=dropout, training=training_flag)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.compat.v1.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=4)
        this_args = deconv_args.copy()
        this_args["activation"] = None
        output = tf.compat.v1.layers.conv3d_transpose(input, 1, **this_args)
        layers.append(output)

    predictions = tf.squeeze(layers[-1], -1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        temp = tf.compat.v1.saved_model.signature_constants
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={temp.DEFAULT_SERVING_SIGNATURE_DEF_KEY: PredictOutput(predictions)},
        )
    else:
        labels, filenames = labels
        loss = tf.losses.mean_squared_error(labels, predictions)

        # Add a scalar summary for the snapshot loss.
        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            export_outputs={"output": predictions},
        )


def _get_resize_arg(target_shape):
    """Get resizing arguments, as used by peaks2maps.

    .. deprecated:: 0.0.11
        `_get_resize_arg` will be removed in NiMARE 0.0.13.

    .. versionadded:: 0.0.1

    """
    mni_shape_mm = np.array([148.0, 184.0, 156.0])
    target_resolution_mm = np.ceil(mni_shape_mm / np.array(target_shape)).astype(np.int32)
    target_affine = np.array(
        [
            [4.0, 0.0, 0.0, -75.0],
            [0.0, 4.0, 0.0, -105.0],
            [0.0, 0.0, 4.0, -70.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    target_affine[0, 0] = target_resolution_mm[0]
    target_affine[1, 1] = target_resolution_mm[1]
    target_affine[2, 2] = target_resolution_mm[2]
    return target_affine, list(target_shape)


def _get_generator(contrasts_coordinates, target_shape, affine, skip_out_of_bounds=False):
    """Get generator, as used by peaks2maps."""

    def generator():
        for contrast in contrasts_coordinates:
            encoded_coords = np.zeros(list(target_shape))
            for real_pt in contrast:
                vox_pt = np.rint(nib.affines.apply_affine(npl.inv(affine), real_pt)).astype(int)
                if skip_out_of_bounds and (vox_pt[0] >= 32 or vox_pt[1] >= 32 or vox_pt[2] >= 32):
                    continue
                encoded_coords[vox_pt[0], vox_pt[1], vox_pt[2]] = 1
            yield (encoded_coords, encoded_coords)

    return generator


@due.dcite(
    references.PEAKS2MAPS,
    description="Transforms coordinates of peaks to unthresholded maps using a deep "
    "convolutional neural net.",
)
def compute_p2m_ma(
    contrasts_coordinates, skip_out_of_bounds=True, tf_verbosity_level=None, model_dir="auto"
):
    """Generate modeled activation (MA) maps using deep ConvNet model peaks2maps.

    .. deprecated:: 0.0.11
        `compute_p2m_ma` will be removed in NiMARE 0.0.13.

    Parameters
    ----------
    contrasts_coordinates : list of lists that are len == 3
        List of contrasts and their coordinates
    skip_out_of_bounds : bool, optional
        Remove coordinates outside of the bounding box of the peaks2maps model
    tf_verbosity_level : int
        Tensorflow verbosity logging level
    model_dir : str, optional
        Location of peaks2maps model. Default is "auto".

    Returns
    -------
    ma_values : array-like
        1d array of modeled activation values.
    """
    try:
        import tensorflow as tf
    except ImportError as e:
        if "No module named 'tensorflow'" in str(e):
            raise Exception(
                "tensorflow not installed - see https://www.tensorflow.org/install/ "
                "for instructions"
            )
        else:
            raise

    if tf_verbosity_level is None:
        tf_verbosity_level = tf.compat.v1.logging.FATAL
    target_shape = (32, 32, 32)
    affine, _ = _get_resize_arg(target_shape)
    tf.compat.v1.logging.set_verbosity(tf_verbosity_level)

    def generate_input_fn():
        dataset = tf.compat.v1.data.Dataset.from_generator(
            _get_generator(
                contrasts_coordinates, target_shape, affine, skip_out_of_bounds=skip_out_of_bounds
            ),
            (tf.float32, tf.float32),
            (tf.TensorShape(target_shape), tf.TensorShape(target_shape)),
        )
        dataset = dataset.batch(1)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    # download_peaks2maps_model expects None for "auto"
    model_dir = None if model_dir == "auto" else model_dir
    model_dir = download_peaks2maps_model(data_dir=model_dir)
    model = tf.estimator.Estimator(model_fn, model_dir=model_dir)

    results = model.predict(generate_input_fn)
    results = [result for result in results]
    assert len(results) == len(contrasts_coordinates), "returned %d" % len(results)

    niis = [nib.Nifti1Image(np.squeeze(result), affine) for result in results]
    return niis


def compute_kda_ma(
    shape,
    vox_dims,
    ijks,
    r,
    value=1.0,
    exp_idx=None,
    sum_overlap=False,
):
    """Compute (M)KDA modeled activation (MA) map.

    .. versionchanged:: 0.0.12

        * Remove low-memory option in favor of sparse arrays.
        * Return 4D sparse array.

    .. versionadded:: 0.0.4

    Replaces the values around each focus in ijk with binary sphere.

    Parameters
    ----------
    shape : :obj:`tuple`
        Shape of brain image + buffer. Typically (91, 109, 91).
    vox_dims : array_like
        Size (in mm) of each dimension of a voxel.
    ijks : array-like
        Indices of foci. Each row is a coordinate, with the three columns
        corresponding to index in each of three dimensions.
    r : :obj:`int`
        Sphere radius, in mm.
    value : :obj:`int`
        Value for sphere.
    exp_idx : array_like
        Optional indices of experiments. If passed, must be of same length as
        ijks. Each unique value identifies all coordinates in ijk that come from
        the same experiment. If None passed, it is assumed that all coordinates
        come from the same experiment.
    sum_overlap : :obj:`bool`
        Whether to sum voxel values in overlapping spheres.

    Returns
    -------
    kernel_data : :obj:`sparse._coo.core.COO`
        4D sparse array. If `exp_idx` is none, a 3d array in the same
        shape as the `shape` argument is returned. If `exp_idx` is passed, a 4d array
        is returned, where the first dimension has size equal to the number of
        unique experiments, and the remaining 3 dimensions are equal to `shape`.
    """
    if exp_idx is None:
        exp_idx = np.ones(len(ijks))

    uniq, exp_idx = np.unique(exp_idx, return_inverse=True)
    n_studies = len(uniq)

    kernel_shape = (n_studies,) + shape

    n_dim = ijks.shape[1]
    xx, yy, zz = [slice(-r // vox_dims[i], r // vox_dims[i] + 0.01, 1) for i in range(n_dim)]
    cube = np.vstack([row.ravel() for row in np.mgrid[xx, yy, zz]])
    kernel = cube[:, np.sum(np.dot(np.diag(vox_dims), cube) ** 2, 0) ** 0.5 <= r]

    # Preallocate coords array, np.concatenate is too slow
    n_coords = ijks.shape[0] * kernel.shape[1]  # = n_peaks * n_voxels
    coords = np.zeros((4, n_coords), dtype=int)
    temp_idx = 0
    for i, peak in enumerate(ijks):
        sphere = np.round(kernel.T + peak)
        idx = (np.min(sphere, 1) >= 0) & (np.max(np.subtract(sphere, shape), 1) <= -1)
        sphere = sphere[idx, :].astype(int)
        exp = exp_idx[i]

        n_brain_voxels = sphere.shape[0]
        chunk_idx = np.arange(temp_idx, temp_idx + n_brain_voxels, 1, dtype=int)

        coords[0, chunk_idx] = np.full((1, n_brain_voxels), exp)
        coords[1:, chunk_idx] = sphere.T

        temp_idx += n_brain_voxels

    # Usually coords.shape[1] < n_coords, since n_brain_voxels < n_voxels sometimes
    coords = coords[:, :temp_idx]

    if not sum_overlap:
        coords = np.unique(coords, axis=1)

    data = np.full(coords.shape[1], value)
    kernel_data = sparse.COO(coords, data, shape=kernel_shape)

    return kernel_data


def compute_ale_ma(shape, ijk, kernel):
    """Generate ALE modeled activation (MA) maps.

    Replaces the values around each focus in ijk with the contrast-specific
    kernel. Takes the element-wise maximum when looping through foci, which
    accounts for foci which are near to one another and may have overlapping
    kernels.

    Parameters
    ----------
    shape : tuple
        Shape of brain image + buffer. Typically (91, 109, 91) + (30, 30, 30).
    ijk : array-like
        Indices of foci. Each row is a coordinate, with the three columns
        corresponding to index in each of three dimensions.
    kernel : array-like
        3D array of smoothing kernel. Typically of shape (30, 30, 30).

    Returns
    -------
    ma_values : array-like
        1d array of modeled activation values.
    """
    ma_values = np.zeros(shape)
    mid = int(np.floor(kernel.shape[0] / 2.0))
    mid1 = mid + 1
    for j_peak in range(ijk.shape[0]):
        i, j, k = ijk[j_peak, :]
        xl = max(i - mid, 0)
        xh = min(i + mid1, ma_values.shape[0])
        yl = max(j - mid, 0)
        yh = min(j + mid1, ma_values.shape[1])
        zl = max(k - mid, 0)
        zh = min(k + mid1, ma_values.shape[2])
        xlk = mid - (i - xl)
        xhk = mid - (i - xh)
        ylk = mid - (j - yl)
        yhk = mid - (j - yh)
        zlk = mid - (k - zl)
        zhk = mid - (k - zh)

        if (
            (xl >= 0)
            & (xh >= 0)
            & (yl >= 0)
            & (yh >= 0)
            & (zl >= 0)
            & (zh >= 0)
            & (xlk >= 0)
            & (xhk >= 0)
            & (ylk >= 0)
            & (yhk >= 0)
            & (zlk >= 0)
            & (zhk >= 0)
        ):
            ma_values[xl:xh, yl:yh, zl:zh] = np.maximum(
                ma_values[xl:xh, yl:yh, zl:zh], kernel[xlk:xhk, ylk:yhk, zlk:zhk]
            )
    return ma_values


@due.dcite(references.ALE_KERNEL, description="Introduces sample size-dependent kernels to ALE.")
def get_ale_kernel(img, sample_size=None, fwhm=None):
    """Estimate 3D Gaussian and sigma (in voxels) for ALE kernel given sample size or fwhm."""
    if sample_size is not None and fwhm is not None:
        raise ValueError('Only one of "sample_size" and "fwhm" may be specified')
    elif sample_size is None and fwhm is None:
        raise ValueError('Either "sample_size" or "fwhm" must be provided')
    elif sample_size is not None:
        uncertain_templates = (
            5.7 / (2.0 * np.sqrt(2.0 / np.pi)) * np.sqrt(8.0 * np.log(2.0))
        )  # pylint: disable=no-member
        # Assuming 11.6 mm ED between matching points
        uncertain_subjects = (11.6 / (2 * np.sqrt(2 / np.pi)) * np.sqrt(8 * np.log(2))) / np.sqrt(
            sample_size
        )  # pylint: disable=no-member
        fwhm = np.sqrt(uncertain_subjects**2 + uncertain_templates**2)

    fwhm_vox = fwhm / np.sqrt(np.prod(img.header.get_zooms()))
    sigma_vox = (
        fwhm_vox * np.sqrt(2.0) / (np.sqrt(2.0 * np.log(2.0)) * 2.0)
    )  # pylint: disable=no-member

    data = np.zeros((31, 31, 31))
    mid = int(np.floor(data.shape[0] / 2.0))
    data[mid, mid, mid] = 1.0
    kernel = ndimage.filters.gaussian_filter(data, sigma_vox, mode="constant")

    # Crop kernel to drop surrounding zeros
    mn = np.min(np.where(kernel > np.spacing(1))[0])
    mx = np.max(np.where(kernel > np.spacing(1))[0])
    kernel = kernel[mn : mx + 1, mn : mx + 1, mn : mx + 1]
    mid = int(np.floor(data.shape[0] / 2.0))
    return sigma_vox, kernel


def _get_last_bin(arr1d):
    """Index the last location in a 1D array with a non-zero value."""
    if np.any(arr1d):
        last_bin = np.where(arr1d)[0][-1]

    else:
        last_bin = 0

    return last_bin


def _calculate_cluster_measures(arr3d, threshold, conn, tail="upper"):
    """Calculate maximum cluster mass and size for an array.

    This method assesses both positive and negative clusters.

    Parameters
    ----------
    arr3d : :obj:`numpy.ndarray`
        Unthresholded 3D summary-statistic matrix. This matrix will end up changed in place.
    threshold : :obj:`float`
        Uncorrected summary-statistic thresholded for defining clusters.
    conn : :obj:`numpy.ndarray` of shape (3, 3, 3)
        Connectivity matrix for defining clusters.

    Returns
    -------
    max_size, max_mass : :obj:`float`
        Maximum cluster size and mass from the matrix.
    """
    if tail == "upper":
        arr3d[arr3d <= threshold] = 0
    else:
        arr3d[np.abs(arr3d) <= threshold] = 0

    labeled_arr3d = np.empty(arr3d.shape, int)
    labeled_arr3d, _ = ndimage.measurements.label(arr3d > 0, conn)

    if tail == "two":
        # Label positive and negative clusters separately
        n_positive_clusters = np.max(labeled_arr3d)
        temp_labeled_arr3d, _ = ndimage.measurements.label(arr3d < 0, conn)
        temp_labeled_arr3d[temp_labeled_arr3d > 0] += n_positive_clusters
        labeled_arr3d = labeled_arr3d + temp_labeled_arr3d
        del temp_labeled_arr3d

    clust_vals, clust_sizes = np.unique(labeled_arr3d, return_counts=True)
    assert clust_vals[0] == 0

    # Cluster mass-based inference
    max_mass = 0
    for unique_val in clust_vals[1:]:
        ss_vals = np.abs(arr3d[labeled_arr3d == unique_val]) - threshold
        max_mass = np.maximum(max_mass, np.sum(ss_vals))

    # Cluster size-based inference
    clust_sizes = clust_sizes[1:]  # First cluster is zeros in matrix
    if clust_sizes.size:
        max_size = np.max(clust_sizes)
    else:
        max_size = 0

    return max_size, max_mass
