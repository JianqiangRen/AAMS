# coding=utf-8
# summary: Code of CVPR 2019 accepted paper Attention-aware Multi-stroke Style Transfer

import tensorflow as tf
import vgg
from tensorflow.python.ops import control_flow_ops
import tensorflow.contrib.slim as slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512


def zca_normalization(features):
    shape = tf.shape(features)

    # reshape the features to orderless feature vectors
    mean_features = tf.reduce_mean(features, axis=[1, 2], keep_dims=True)
    unbiased_features = tf.reshape(features - mean_features, shape=(shape[0], -1, shape[3]))

    # get the covariance matrix
    gram = tf.matmul(unbiased_features, unbiased_features, transpose_a=True)
    gram /= tf.reduce_prod(tf.cast(shape[1:3], tf.float32))

    # converting the feature spaces
    s, u, v = tf.svd(gram, compute_uv=True)
    s = tf.expand_dims(s, axis=1)  # let it be active in the last dimension

    # get the effective singular values
    valid_index = tf.cast(s > 0.00001, dtype=tf.float32)
    s_effective = tf.maximum(s, 0.00001)
    sqrt_s_effective = tf.sqrt(s_effective) * valid_index
    sqrt_inv_s_effective = tf.sqrt(1.0/s_effective) * valid_index

    # colorization functions
    colorization_kernel = tf.matmul(tf.multiply(u, sqrt_s_effective), v, transpose_b=True)

    # normalized features
    normalized_features = tf.matmul(unbiased_features, u)
    normalized_features = tf.multiply(normalized_features, sqrt_inv_s_effective)
    normalized_features = tf.matmul(normalized_features, v, transpose_b=True)
    normalized_features = tf.reshape(normalized_features, shape=shape)

    return normalized_features, colorization_kernel, mean_features


def zca_colorization(normalized_features, colorization_kernel, mean_features):
    # broadcasting the tensors for matrix multiplication
    shape = tf.shape(normalized_features)
    normalized_features = tf.reshape(
        normalized_features, shape=(shape[0], -1, shape[3]))
    colorized_features = tf.matmul(normalized_features, colorization_kernel)
    colorized_features = tf.reshape(colorized_features, shape=shape) + mean_features
    return colorized_features


def adain_normalization(features):
    epsilon = 1e-7
    mean_features, colorization_kernels = tf.nn.moments(features, [1, 2], keep_dims=True)
    normalized_features = tf.div(
        tf.subtract(features, mean_features), tf.sqrt(tf.add(colorization_kernels, epsilon)))
    return normalized_features, colorization_kernels, mean_features


def adain_colorization(normalized_features, colorization_kernels, mean_features):
    return tf.sqrt(colorization_kernels) * normalized_features + mean_features


def project_features(features, projection_module='ZCA'):
    if projection_module == 'ZCA':
        return zca_normalization(features)
    elif projection_module == 'AdaIN':
        return adain_normalization(features)
    else:
        return features, None, None


def reconstruct_features(projected_features, feature_kernels, mean_features, reconstruction_module='ZCA'):
    if reconstruction_module == 'ZCA':
        return zca_colorization(projected_features, feature_kernels, mean_features)
    elif reconstruction_module == 'AdaIN':
        return adain_colorization(projected_features, feature_kernels, mean_features)
    else:
        return projected_features


def instance_norm(inputs, epsilon=1e-10):
    inst_mean, inst_var = tf.nn.moments(inputs, [1, 2], keep_dims=True)
    normalized_inputs = tf.div( tf.subtract(inputs, inst_mean), tf.sqrt(tf.add(inst_var, epsilon)))
    return normalized_inputs


def adaptive_instance_normalization(content_feature, style_feature):
    normalized_content_feature = instance_norm(content_feature)
    inst_mean, inst_var = tf.nn.moments(style_feature, [1, 2], keep_dims=True)
    return tf.sqrt(inst_var) * normalized_content_feature + inst_mean


def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])
 
 
def conv(x, channels, kernel=3, stride=1, pad=1, pad_type='zero', scope='conv_0'):
    with tf.variable_scope(scope):
        if pad_type == 'zero':
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect':
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')
        
        x = tf.layers.conv2d(x, channels, kernel, stride, kernel_initializer=tf.contrib.layers.xavier_initializer())
        return x


def upsampling(x, stride, scope='upsample_0'):
    with tf.variable_scope(scope):
        stride_larger_than_one = tf.greater(stride, 1)
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        new_height, new_width = tf.cond(
            stride_larger_than_one,
            lambda: (height * stride, width * stride),
            lambda: (height, width))
        x = tf.image.resize_nearest_neighbor(x, [new_height, new_width])
        return x


def avg_pooling(x, size, stride, scope='pool_0'):
    with tf.variable_scope(scope):
        x = tf.layers.average_pooling2d(x, size, stride, 'same')
        return x

def mean_image_subtraction(images, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
    num_channels = 3
    channels = tf.split(images, num_channels, axis=2)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(channels, axis=2)


def mean_image_summation(image, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
    num_channels = 3
    channels = tf.split(image, num_channels, axis=2)
    for i in range(num_channels):
        channels[i] += means[i]
    return tf.concat(channels, axis=2)


def batch_mean_image_subtraction(images, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
    if images.get_shape().ndims != 4:
        raise ValueError('Input must be of size [batch, height, width, C>0')
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(images, num_channels, axis=3)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(channels, axis=3)


def batch_mean_image_summation(images, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
    if images.get_shape().ndims != 4:
        raise ValueError('Input must be of size [batch, height, width, C>0')
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(images, num_channels, axis=3)
    for i in range(num_channels):
        channels[i] += means[i]
    return tf.concat(channels, axis=3)


def extract_image_features(inputs,  reuse=True):
    with slim.arg_scope(vgg.vgg_arg_scope()):
        _, end_points = vgg.vgg_19(inputs, spatial_squeeze=False, is_training=False, reuse=reuse)
    return end_points


def compute_total_variation_loss_l1(inputs, weights=1, scope=None):
    inputs_shape = tf.shape(inputs)
    height = inputs_shape[1]
    width = inputs_shape[2]

    with tf.variable_scope(scope, 'total_variation_loss', [inputs]):
        loss_y = tf.losses.absolute_difference(
            tf.slice(inputs, [0, 0, 0, 0], [-1, height-1, -1, -1]),
            tf.slice(inputs, [0, 1, 0, 0], [-1, -1, -1, -1]),
            weights=weights,
            scope='loss_y')
        loss_x = tf.losses.absolute_difference(
            tf.slice(inputs, [0, 0, 0, 0], [-1, -1, width-1, -1]),
            tf.slice(inputs, [0, 0, 1, 0], [-1, -1, -1, -1]),
            weights=weights,
            scope='loss_x')
        loss = loss_y + loss_x
        return loss
    
    
def _smallest_size_at_least(height, width, smallest_side):
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)
    return new_height, new_width


def _aspect_preserving_resize(image, smallest_side):
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                             align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image


def _mean_image_subtraction(image, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)

def preprocessing_for_train(image, output_height, output_width, resize_side):
    image = _aspect_preserving_resize(image, resize_side)
    image = _random_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


def preprocessing_for_eval(image, output_height, output_width, resize_side):
    image = _aspect_preserving_resize(image, resize_side)
    image = _central_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


def preprocessing_image(image, output_height, output_width,
                        resize_side=_RESIZE_SIDE_MIN, is_training=False):
    if is_training:
        return preprocessing_for_train(image, output_height, output_width, resize_side)
    else:
        return preprocessing_for_eval(image, output_height, output_width, resize_side)


def _crop(image, offset_height, offset_width, crop_height, crop_width):
    original_shape = tf.shape(image)

    rank_assertion = tf.Assert(
        tf.equal(tf.rank(image), 3),
        ['Rank of image must be equal to 3.'])
    cropped_shape = control_flow_ops.with_dependencies(
        [rank_assertion],
        tf.stack([crop_height, crop_width, original_shape[2]]))

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)),
        ['Crop size greater than the image size.'])

    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    image = control_flow_ops.with_dependencies(
        [size_assertion],
        tf.slice(image, offsets, cropped_shape))
    return tf.reshape(image, cropped_shape)


def _random_crop(image_list, crop_height, crop_width):
    if not image_list:
        raise ValueError('Empty image_list.')

    # Compute the rank assertions.
    rank_assertions = []
    for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(
            tf.equal(image_rank, 3),
            ['Wrong rank for tensor  %s [expected] [actual]',
             image_list[i].name, 3, image_rank])
        rank_assertions.append(rank_assert)

    image_shape = control_flow_ops.with_dependencies(
        [rank_assertions[0]],
        tf.shape(image_list[0]))
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(image_height, crop_height),
            tf.greater_equal(image_width, crop_width)),
        ['Crop size greater than the image size.'])

    asserts = [rank_assertions[0], crop_size_assert]

    for i in range(1, len(image_list)):
        image = image_list[i]
        asserts.append(rank_assertions[i])
        shape = control_flow_ops.with_dependencies([rank_assertions[i]],
                                                   tf.shape(image))
        height = shape[0]
        width = shape[1]

        height_assert = tf.Assert(
            tf.equal(height, image_height),
            ['Wrong height for tensor %s [expected][actual]',
             image.name, height, image_height])
        width_assert = tf.Assert(
            tf.equal(width, image_width),
            ['Wrong width for tensor %s [expected][actual]',
             image.name, width, image_width])
        asserts.extend([height_assert, width_assert])

    # Create a random bounding box.
    #
    # Use tf.random_uniform and not numpy.random.rand as doing the former would
    # generate random numbers at graph eval time, unlike the latter which
    # generates random numbers at graph definition time.
    max_offset_height = control_flow_ops.with_dependencies(
        asserts, tf.reshape(image_height - crop_height + 1, []))
    max_offset_width = control_flow_ops.with_dependencies(
        asserts, tf.reshape(image_width - crop_width + 1, []))
    offset_height = tf.random_uniform(
        [], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform(
        [], maxval=max_offset_width, dtype=tf.int32)

    return [_crop(image, offset_height, offset_width,
                  crop_height, crop_width) for image in image_list]


def _central_crop(image_list, crop_height, crop_width):
    outputs = []
    for image in image_list:
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        offset_height = (image_height - crop_height) / 2
        offset_width = (image_width - crop_width) / 2

        outputs.append(_crop(image, offset_height, offset_width,
                             crop_height, crop_width))
    return outputs


def k_means(image, clusters_num):
    image = tf.squeeze(image)
    print("k_means", image.shape)
    _points = tf.reshape(image, (-1, 1))
    centroids = tf.slice(tf.random_shuffle(_points), [0, 0], [clusters_num, -1])
    points_expanded = tf.expand_dims(_points, 0)

    for i in xrange(80):
        centroids_expanded = tf.expand_dims(centroids, 1)
        distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
        assignments = tf.argmin(distances, 0)
        centroids = tf.concat(
            [tf.reduce_mean(tf.gather(_points, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])), axis=1) for c
             in
             xrange(clusters_num)], 0)

    centroids = tf.squeeze(centroids)
    centroids = -tf.nn.top_k(-centroids, clusters_num)[0]  # sort
    return centroids


if __name__ == "__main__":
    import cv2
    img = cv2.imread('lenna_cropped.jpg', cv2.IMREAD_GRAYSCALE)
    points = tf.cast(tf.convert_to_tensor(img), tf.float32)
    print(points.shape)
    centroids = k_means(points, 4)
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    print(sess.run(centroids))