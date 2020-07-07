# coding=utf-8
# summary: Code of CVPR 2019 accepted paper Attention-aware Multi-stroke Style Transfer

import tensorflow as tf
from net import aams, utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", dest='ckpt_path', type=str)
parser.add_argument("--name", dest='name', type=str)
args = parser.parse_args()


def freeze():
    model = aams.AAMS()
    # predict the stylized image
    inp_content_image = tf.placeholder(tf.float32, shape=(None, None, 3), name='content')
    inp_style_image = tf.placeholder(tf.float32, shape=(None, None, 3), name='style')
    inter_weight = tf.placeholder(tf.float32, shape=(), name='inter_weight')
    # preprocess the content and style images
    content_image = utils.mean_image_subtraction(inp_content_image)
    content_image = tf.expand_dims(content_image, axis=0)
    # style resizing and cropping
    style_image = utils.preprocessing_image(
        inp_style_image,
        448,
        448,
        512)
    style_image = tf.expand_dims(style_image, axis=0)
    
    # style transfer
    stylized_image, attention_map, centroids = model.transfer(
        content_image,
        style_image,
        inter_weight=inter_weight)
    stylized_image = tf.identity(stylized_image, name='stylized_output')
    attention_map = tf.identity(attention_map, name='attention_map')
    centroids = tf.identity(centroids, name="centroids")

    init_op = tf.global_variables_initializer()
    
    restore_saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(init_op)
        restore_saver.restore(sess, args.ckpt_path)
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, \
                                                                        output_node_names=['stylized_output',
                                                                                           'attention_map','centroids'])
        
        with open(args.name, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())
    
    print('freeze done')


'''
usage:
python freeze_model.py  --ckpt_path tfmodel/model.ckpt-80000 \
                --name tfmodel/aams.pb
'''
if __name__ == '__main__':
    freeze()

