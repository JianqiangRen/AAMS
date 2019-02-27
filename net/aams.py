# coding=utf-8
# summary: Code of CVPR 2019 accepted paper Attention-aware Multi-stroke Style Transfer

import utils
import tensorflow as tf

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)


class AAMS():
    def __init__(self):
        self.total_loss = 0.0
        self.recons_loss = None
        self.tv_loss = None
        self.attention_l1_loss = None
        self.perceptual_loss = None
        self.perceptual_loss_layers = ['conv1/conv1_1', 'conv2/conv2_1', 'conv3/conv3_1', 'conv4/conv4_1']
        self.summaries = None
        self.recons_weight = 10
        self.perceptual_weight = 1
        self.tv_weight = 10
        self.attention_weight = 6
        self.train_op = None
        self.learning_rate = 0.0001

    @staticmethod
    def decode(x, encode_features, reuse=False):
        with tf.variable_scope("decoder", reuse=reuse):
            x = utils.conv(x, 256, 3, 1, scope='inv_conv4_1')
            x = tf.nn.relu(x)
            x = utils.upsampling(x, 2, scope='upsample_1')
            x = utils.conv(x, 256, 3, 1, scope='inv_conv3_4')
            x = tf.nn.relu(x)
            x = utils.conv(x, 256, 3, 1, scope='inv_conv3_3')
            x = tf.nn.relu(x)
            x = utils.conv(x, 256, 3, 1, scope='inv_conv3_2')
            x = tf.nn.relu(x)
            x = utils.adaptive_instance_normalization(x, encode_features['vgg_19/conv3/conv3_1'])
            x = utils.conv(x, 128, 3, 1, scope='inv_conv3_1')
            x = tf.nn.relu(x)
            x = utils.upsampling(x, 2, scope='upsample_2')
            x = utils.conv(x, 128, 3, 1, scope='inv_conv2_2')
            x = tf.nn.relu(x)
            x = utils.adaptive_instance_normalization(x, encode_features['vgg_19/conv2/conv2_1'])
            x = utils.conv(x, 64, 3, 1, scope='inv_conv2_1')
            x = tf.nn.relu(x)
            x = utils.upsampling(x, 2, scope='upsample_3')
            x = utils.conv(x, 64, 3, 1, scope='inv_conv1_2')
            x = tf.nn.relu(x)
            x = utils.adaptive_instance_normalization(x, encode_features['vgg_19/conv1/conv1_1'])
            x = utils.conv(x, 3, 3, 1, scope='inv_conv1_1')
            return x + 127.5

    @staticmethod
    def self_attention_autoencoder(x):
        input_features = utils.extract_image_features(x, False)
        tf.logging.debug(input_features['vgg_19/conv1/conv1_1'])
        tf.logging.debug(input_features['vgg_19/conv2/conv2_1'])
        tf.logging.debug(input_features['vgg_19/conv3/conv3_1'])
        tf.logging.debug(input_features['vgg_19/conv4/conv4_1'])

        projected_hidden_feature, colorization_kernels, mean_features = utils.adain_normalization(input_features['vgg_19/conv4/conv4_1'])

        attention_feature_map = AAMS.self_attention(projected_hidden_feature, tf.shape(projected_hidden_feature))
        hidden_feature = tf.multiply(projected_hidden_feature, attention_feature_map) + projected_hidden_feature
       
        hidden_feature = utils.adain_colorization(hidden_feature, colorization_kernels, mean_features)
        
        output = AAMS.decode(hidden_feature, input_features)
        return output,  attention_feature_map

    @staticmethod
    def self_attention(x, size, scope='self_attention', reuse=False):
        tf.logging.debug(x)
    
        with tf.variable_scope(scope, reuse=reuse):
            C = x.shape[3]
            f = utils.conv(x, C // 2, kernel=1, stride=1, pad=0, scope='f_conv')  # [bs, h, w, c']
            g = utils.conv(x, C // 2, kernel=1, stride=1, pad=0, scope='g_conv')  # [bs, h, w, c']
            h = utils.conv(x, C, kernel=1, stride=1, pad=0, scope='h_conv')  # [bs, h, w, c]
        
            s = tf.matmul(utils.hw_flatten(g), utils.hw_flatten(f), transpose_b=True)  ## [bs, N, N]
            beta = tf.nn.softmax(s)  # self_attention map
        
            o = tf.matmul(beta, utils.hw_flatten(h))  # [bs, N, C]
            o = tf.reshape(o, shape=size)  # [bs, h, w, C]
        return o

    @staticmethod
    def multi_scale_style_swap(content_features, style_features, patch_size=5):
        # channels for both the content and style, must be the same
        c_shape = tf.shape(content_features)
        s_shape = tf.shape(style_features)
        channel_assertion = tf.Assert(
            tf.equal(c_shape[3], s_shape[3]), ['number of channels  must be the same'])
    
        with tf.control_dependencies([channel_assertion]):
            # spatial shapes for style and content features
            c_height, c_width, c_channel = c_shape[1], c_shape[2], c_shape[3]
            proposed_outputs = []
  
            for beta in [1.0/2, 1.0/(2**0.5), 1.0]:
                # convert the style features into convolutional kernels
                new_height = tf.cast(tf.multiply(tf.cast(s_shape[1], tf.float32), beta), tf.int32)
                new_width = tf.cast(tf.multiply(tf.cast(s_shape[2], tf.float32), beta), tf.int32)
                
                tmp_style_features = tf.image.resize_images(style_features, [new_height, new_width],
                                                            method=tf.image.ResizeMethod.BILINEAR)  # 修改图片的尺寸
                # tmp_style_features = style_features
                style_kernels = tf.extract_image_patches(
                    tmp_style_features, ksizes=[1, patch_size, patch_size, 1],
                    strides=[1, 1, 1, 1], rates=[1, 1, 1, 1],
                    padding='SAME')  # [batch, H, W, patch_size * patch_size * channel]
                style_kernels = tf.squeeze(style_kernels, axis=0)  # [H, W, patch_size * patch_size * channel]
                style_kernels = tf.transpose(style_kernels, perm=[2, 0, 1])  # [patch_size * patch_size * channel, H, W]
            
                # gather the conv and deconv kernels
                deconv_kernels = tf.reshape(
                    style_kernels, shape=(patch_size, patch_size, c_channel, -1))
            
                kernels_norm = tf.norm(style_kernels, axis=0, keep_dims=True)
                kernels_norm = tf.reshape(kernels_norm, shape=(1, 1, 1, -1))
            
                # calculate the normalization factor
                mask = tf.ones((c_height, c_width), tf.float32)
                fullmask = tf.zeros((c_height + patch_size - 1, c_width + patch_size - 1), tf.float32)
                for x in range(patch_size):
                    for y in range(patch_size):
                        paddings = [[x, patch_size - x - 1], [y, patch_size - y - 1]]
                        padded_mask = tf.pad(mask, paddings=paddings, mode="CONSTANT")
                        fullmask += padded_mask
                pad_width = int((patch_size - 1) / 2)
                deconv_norm = tf.slice(fullmask, [pad_width, pad_width], [c_height, c_width])
                deconv_norm = tf.reshape(deconv_norm, shape=(1, c_height, c_width, 1))
            
                ########################
                # starting convolution #
                ########################
                # padding operation
                pad_total = patch_size - 1
                pad_beg = pad_total // 2
                pad_end = pad_total - pad_beg
                paddings = [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]
            
                # convolutional operations
                net = tf.pad(content_features, paddings=paddings, mode="REFLECT")
                net = tf.nn.conv2d(
                    net,
                    tf.div(deconv_kernels, kernels_norm + 1e-7),
                    strides=[1, 1, 1, 1],
                    padding='VALID')
                # find the maximum locations
                best_match_ids = tf.argmax(net, axis=3)
                best_match_ids = tf.cast(
                    tf.one_hot(best_match_ids, depth=tf.shape(net)[3]), dtype=tf.float32)
            
                # find the patches and warping the output
                unnormalized_output = tf.nn.conv2d_transpose(
                    value=best_match_ids,
                    filter=deconv_kernels,
                    output_shape=(c_shape[0], c_height + pad_total, c_width + pad_total, c_channel),
                    strides=[1, 1, 1, 1],
                    padding='VALID')
                unnormalized_output = tf.slice(unnormalized_output, [0, pad_beg, pad_beg, 0], c_shape)
                output = tf.div(unnormalized_output, deconv_norm)
                output = tf.reshape(output, shape=c_shape)
                proposed_outputs.append(output)
        
            proposed_outputs.append(content_features)
            return proposed_outputs

    @staticmethod
    def multi_stroke_fusion(stylized_maps, attention_map, theta=50.0, mode='softmax'):
        # stylized_maps: [1,w,h,512]
        stroke_num = len(stylized_maps)
        if stroke_num == 1:
            return stylized_maps[0]
        
        one_channel_attention = tf.expand_dims(tf.reduce_mean(attention_map, axis=-1), 3)
  
        centroids = utils.k_means(one_channel_attention, stroke_num)
        saliency_distances = []

        for i in range(stroke_num):
            saliency_distances.append(tf.abs(one_channel_attention - centroids[i]))
        
        multi_channel_saliency = tf.concat(saliency_distances, -1)  # [1, h, w, stroke_num]
    
        if mode == 'softmax':
            multi_channel_saliency = tf.nn.softmax(theta * (1.0 - multi_channel_saliency), -1)
        elif mode == 'linear':
            multi_channel_saliency = tf.div(multi_channel_saliency,
                                            tf.expand_dims(tf.reduce_sum(multi_channel_saliency, -1), 3))
        else:
            pass
    
        finial_stylized_map = 0
        for i in range(stroke_num):
            temp = tf.expand_dims(multi_channel_saliency[0, :, :, i], 0)
            temp = tf.expand_dims(temp, 3)
            finial_stylized_map += tf.multiply(stylized_maps[i], temp)
    
        return finial_stylized_map, centroids
    
    def build_graph(self, x):
        output, attention_feature_map = self.self_attention_autoencoder(x)
        output = utils.batch_mean_image_subtraction(output)
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        
        if self.recons_weight > 0.0:
            recons_loss = tf.losses.mean_squared_error(
                x, output, weights=self.recons_weight, scope='recons_loss')
            self.recons_loss = recons_loss
            self.total_loss += recons_loss
            summaries.add(tf.summary.scalar('losses/recons_loss', recons_loss))
            
        if self.perceptual_weight > 0.0:
            input_features = utils.extract_image_features(x, True)
            output_features = utils.extract_image_features(output, True)
            
            perceptual_loss = 0.0
            
            for layer in self.perceptual_loss_layers:
                input_perceptual_features = input_features['vgg_19/' + layer]
                output_perceptual_features = output_features['vgg_19/' + layer]

                perceptual_loss += tf.losses.mean_squared_error(
                    input_perceptual_features, output_perceptual_features, weights=self.perceptual_weight, scope=layer)
                
            self.perceptual_loss = perceptual_loss
            self.total_loss += perceptual_loss
            summaries.add(tf.summary.scalar('losses/perceptual_loss', perceptual_loss))
        
        if self.tv_weight > 0.0:
            tv_loss = utils.compute_total_variation_loss_l1(output, self.tv_weight)
            self.tv_loss = tv_loss
            self.total_loss += tv_loss
            summaries.add(tf.summary.scalar('losses/tv_loss', tv_loss))
            
        if self.attention_weight > 0.0:
            atten_l1_loss = self.attention_weight * tf.norm(attention_feature_map, 1)
            self.attention_l1_loss = atten_l1_loss
            self.total_loss += atten_l1_loss
            summaries.add(tf.summary.scalar('losses/attention_l1_loss', atten_l1_loss))

        summaries.add(tf.summary.scalar('losses/total_loss', self.total_loss))

        image_tiles = tf.concat([x, output], axis=2)
        image_tiles = utils.batch_mean_image_summation(image_tiles)
        image_tiles = tf.cast(tf.clip_by_value(image_tiles, 0.0, 255.0), tf.uint8)
        summaries.add(tf.summary.image('image_comparison', image_tiles, max_outputs=8))

        self.summaries = summaries
        return self.total_loss
    
    def _set_optimizer(self):
        optimizer = tf.train.AdamOptimizer(
            self.learning_rate,
            beta1=0.9,
            beta2=0.99,
            epsilon=1.0)
        return optimizer
        
    def get_training_op(self,  global_step, variables_to_train=tf.trainable_variables()):
        # gather the variable summaries
        optimizer = self._set_optimizer()
        variables_summaries = []
        for var in variables_to_train:
            variables_summaries.append(tf.summary.histogram(var.op.name, var))
        variables_summaries = set(variables_summaries)

        # add the training operations
        train_ops = []
        grads_and_vars = optimizer.compute_gradients(
            self.total_loss, var_list=variables_to_train)
        train_op = optimizer.apply_gradients(
            grads_and_vars=grads_and_vars,
            global_step=global_step)
        train_ops.append(train_op)

        self.summaries |= variables_summaries
        self.train_op = tf.group(*train_ops)
        return self.train_op
    
    @staticmethod
    def attention_filter(attention_feature_map, kernel_size=3, mean=6, stddev=5):
        attention_map = tf.abs(attention_feature_map)
    
        attention_mask = attention_map > 2 * tf.reduce_mean(attention_map)
        attention_mask = tf.cast(attention_mask, tf.float32)
    
        w = tf.truncated_normal(shape=[kernel_size, kernel_size], mean=mean, stddev=stddev)
        w = tf.div(w, tf.reduce_sum(w))
        
        # [filter_height, filter_width, in_channels, out_channels]
        w = tf.expand_dims(w, 2)
        w = tf.tile(w, [1, 1, attention_mask.shape[3]])
        w = tf.expand_dims(w, 3)
        w = tf.tile(w, [1, 1, 1, attention_mask.shape[3]])
        attention_map = tf.nn.conv2d(attention_mask, w, strides=[1, 1, 1, 1], padding='SAME')
        attention_map = attention_map - tf.reduce_min(attention_map)
        attention_map = attention_map / tf.reduce_max(attention_map)
        return attention_map
    
    def transfer(self, contents, styles, inter_weight=1.0):
        content_features = utils.extract_image_features(contents, False)
        style_features = utils.extract_image_features(styles, True)
       
        content_hidden_feature = content_features['vgg_19/' + self.perceptual_loss_layers[-1]]
        style_hidden_feature = style_features['vgg_19/' + self.perceptual_loss_layers[-1]]
        
        projected_content_features, _, _ = utils.project_features(content_hidden_feature, 'ZCA')
        projected_style_features, style_kernels, mean_style_features = utils.project_features(style_hidden_feature, 'ZCA')

        attention_feature_map = AAMS.self_attention(projected_content_features, tf.shape(projected_content_features))
        projected_content_features = tf.multiply(projected_content_features, attention_feature_map) + projected_content_features

        attention_map = AAMS.attention_filter(attention_feature_map)

        multi_swapped_features = AAMS.multi_scale_style_swap(projected_content_features, projected_style_features)

        fused_features, centroids = AAMS.multi_stroke_fusion(multi_swapped_features, attention_map, theta=50.0)

        fused_features = inter_weight * fused_features + \
                              (1 - inter_weight) * projected_content_features
        
        reconstructed_features = utils.reconstruct_features(
            fused_features,
            style_kernels,
            mean_style_features,
            'ZCA'
        )
        output = AAMS.decode(reconstructed_features, style_features)
        return output, attention_map, centroids
    
 


