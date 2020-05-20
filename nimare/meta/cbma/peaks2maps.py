"""
Functions for the Peaks2Maps transformer.
"""


def model_fn(features, labels, mode, params):
    """
    Used internally by utils.peaks2maps
    """
    import tensorflow as tf
    from tensorflow.python.estimator.export.export_output import PredictOutput
    ngf = 64
    layers = []

    training_flag = mode == tf.estimator.ModeKeys.TRAIN

    input_images_placeholder = tf.expand_dims(features, -1)

    conv_args = {"strides": 2,
                 "kernel_size": 4,
                 "padding": "valid",
                 "activation": tf.nn.leaky_relu,
                 "kernel_initializer": tf.random_normal_initializer(0, 0.02),
                 "name": "conv",
                 "use_bias": False}

    deconv_args = conv_args.copy()
    deconv_args["padding"] = "same"
    deconv_args["name"] = "deconv"

    batchnorm_args = {"scale": True,
                      "gamma_initializer": tf.random_normal_initializer(1.0, 0.02),
                      "center": True,
                      "beta_initializer": tf.zeros_initializer(),
                      "name": "batchnorm",
                      "training": training_flag}

    def pad_and_conv(input, out_channels, conv_args):
        padded_input = tf.pad(input,
                              [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]],
                              mode="CONSTANT")
        convolved = tf.layers.conv3d(padded_input, out_channels,
                                     **conv_args)
        return convolved

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
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
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = pad_and_conv(layers[-1], out_channels, conv_args)
            output = tf.layers.batch_normalization(convolved, **batchnorm_args)
            if dropout > 0.0:
                output = tf.layers.dropout(output, rate=dropout,
                                           training=training_flag)
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
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=4)

            output = tf.layers.conv3d_transpose(input, out_channels,
                                                **deconv_args)
            output = tf.layers.batch_normalization(output,
                                                   **batchnorm_args)

            if dropout > 0.0:
                output = tf.layers.dropout(output, rate=dropout,
                                           training=training_flag)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=4)
        this_args = deconv_args.copy()
        this_args['activation'] = None
        output = tf.layers.conv3d_transpose(input, 1, **this_args)
        layers.append(output)

    predictions = tf.squeeze(layers[-1], -1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: PredictOutput(predictions)}
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
            train_op = optimizer.minimize(loss,
                                          global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            export_outputs={"output": predictions}
        )
