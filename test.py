# coding=utf-8
# summary: Code of CVPR 2019 accepted paper Attention-aware Multi-stroke Style Transfer

import tensorflow as tf
import scipy.misc
import numpy as np
from PIL import Image
import argparse
import os
import errno
import shutil
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--model", dest='model', type=str)
parser.add_argument("--content", dest='content', type=str)
parser.add_argument("--style", dest='style', type=str)
parser.add_argument("--get_sal", dest='get_sal', type=bool, default=False)
parser.add_argument("--inter_weight", dest='inter_weight', type=float, default=1.0)
args = parser.parse_args()

max_length = 800


def single_img_test(model_path, content_path, style_path, inter_weight_value=1.0):
    f = tf.gfile.FastGFile(model_path, 'rb')
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    persisted_graph = tf.import_graph_def(graph_def, name='')
    
    sess = tf.InteractiveSession(graph=persisted_graph)
    
    content = tf.get_default_graph().get_tensor_by_name("content:0")
    style = tf.get_default_graph().get_tensor_by_name("style:0")
    output = tf.get_default_graph().get_tensor_by_name("stylized_output:0")
    attention = tf.get_default_graph().get_tensor_by_name('attention_map:0')
    inter_weight = tf.get_default_graph().get_tensor_by_name("inter_weight:0")
    centroids = tf.get_default_graph().get_tensor_by_name("centroids:0")
    content_feed = image_reader(content_path)
    
    style_feed = image_reader(style_path)
    
    if np.shape(content_feed)[0] >max_length or np.shape(content_feed)[1]>max_length:
        h = np.shape(content_feed)[0]
        w = np.shape(content_feed)[1]
        if h > w:
            content_feed = cv2.resize(content_feed, (max_length, max_length *h /w))
        else:
            content_feed = cv2.resize(content_feed, (max_length*w/h, max_length) )
    
    output_value, attention_value, centroids_value = sess.run([output, attention, centroids], feed_dict={content: content_feed,
                                                                        style: style_feed,
                                                                        inter_weight: inter_weight_value
                                                                        })
    
    print('content size:', np.shape(content_feed))
    print('style size:', np.shape(style_feed))
    print('output size:', np.shape(output_value))
    print('attention size:', np.shape(attention_value))
    print('centroids',centroids_value)

    prepare_dir('images/test_result')
    filename = 'images/test_result/{}_stylized_{}.{}'.format(
        content_path.split('/')[-1].split('.')[0],
        style_path.split('/')[-1].split('.')[0],
        content_path.split('.')[-1]
    )
    output_image = output_value[0]
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
 
    imsave(filename, output_image.astype(np.uint8))
    print('saving {}'.format(filename))
    
    ''' save attention map'''
    mean_sal = 0
    for i in xrange(attention_value.shape[3]):
        mean_sal += attention_value[0, :, :, i]
    
    mean_sal = mean_sal * 1.0 / attention_value.shape[3]
    
    from matplotlib import pyplot as plt
    from matplotlib import cm
    plt.switch_backend('agg')
    
    mean_sal = mean_sal - np.min(mean_sal)
    mean_sal = mean_sal * 1.0 / np.max(mean_sal)
    
    plt.imshow(mean_sal, cmap=cm.get_cmap('rainbow', 1000))
    plt.colorbar()
    plt.axis('off')
 
    print('mean_sal size:', np.shape(mean_sal))
    filename = 'images/test_result/{}_mean_atten.png'.format(
        content_path.split('/')[-1].split('.')[0])
    
    plt.savefig(filename, bbox_inches="tight")
    print('attention mean:{}, min:{}, max:{}'.format(np.mean(mean_sal), np.min(mean_sal), np.max(mean_sal)))
 
    sess.close()
    
    print('single image test done')


def imsave(filename, img):
    Image.fromarray(img).save(filename, quality=95)


def empty_dir(path):
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print 'Warning: {}'.format(e)


def create_dir(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise


def prepare_dir(path, empty=False):
    if not os.path.exists(path):
        create_dir(path)
    
    if empty:
        empty_dir(path)


def image_reader(filename):
    img = scipy.misc.imread(filename).astype(np.float)
    if len(img.shape) == 2:
        img = np.dstack((img, img, img))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


'''
usage:
python single_img_test.py --model models/author/avatar.pb
                        --content data/contents/images/woman_side_portrait.jpg
                        --style data/styles/brushstrokers.jpg
                        --inter_weight 1.0
'''
if __name__ == "__main__":
    single_img_test(args.model, args.content, args.style, args.inter_weight)
