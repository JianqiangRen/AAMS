# coding=utf-8
# summary: Code of CVPR 2019 accepted paper Attention-aware Multi-stroke Style Transfer

import tensorflow as tf
import tensorflow.contrib.slim as slim


def vgg_arg_scope(weight_decay=0.0005):
    """Defines the VGG arg scope.

    Args:
        weight_decay: The l2 regularization coefficient.

    Returns:
        An arg_scope.
      """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc


def vgg_19(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           reuse=True,
           scope='vgg_19'):
    """Oxford Net VGG 19-Layers version E Example.

    Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

    Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        num_classes: number of predicted classes.
        is_training: whether or not the models is being trained.
        dropout_keep_prob: the probability that activations are kept in the dropout
            layers during training.
        spatial_squeeze: whether or not should squeeze the spatial dimensions of the
            outputs. Useful to remove unnecessary dimensions for classification.
        reuse: whether to reuse the network parameters
        scope: Optional scope for the variables.

    Returns:
        the last op containing the log predictions and end_points dict.
    """
    with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            # net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.avg_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            # net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.avg_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
            # net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.avg_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
            # net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.avg_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
            # net = slim.max_pool2d(net, [2, 2], scope='pool5')
            net = slim.avg_pool2d(net, [2, 2], scope='pool5')
            # Use conv2d instead of fully_connected layers.
            net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout7')
            net = slim.conv2d(net, num_classes, [1, 1],
                              activation_fn=None,
                              normalizer_fn=None,
                              scope='fc8')
            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if spatial_squeeze:
                net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                end_points[sc.name + '/fc8'] = net
            return net, end_points


vgg_19.default_image_size = 224