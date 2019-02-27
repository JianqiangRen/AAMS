# coding=utf-8
# summary: Code of CVPR 2019 accepted paper Attention-aware Multi-stroke Style Transfer

import tensorflow as tf
import glob
import os
from net import utils, aams
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", dest='dataset', type=str)
args = parser.parse_args()


def _get_init_fn():
    vgg_checkpoint_path = "vgg_19.ckpt"
    if tf.gfile.IsDirectory(vgg_checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(vgg_checkpoint_path)
    else:
        checkpoint_path = vgg_checkpoint_path

    variables_to_restore = []
    for var in slim.get_model_variables():
        tf.logging.info('model_var: %s' % var)
        excluded = False
        for exclusion in ['vgg_19/fc']:
            if var.op.name.startswith(exclusion):
                excluded = True
                tf.logging.info('exclude:%s' % exclusion)
                break
        if not excluded:
            variables_to_restore.append(var)

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)
    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=True)


if __name__ == "__main__":
    with tf.Graph().as_default():
        global_step = slim.create_global_step()

        img_paths = glob.glob(os.path.join(args.dataset, "*.jpg"))

        path_queue = tf.train.string_input_producer(img_paths, shuffle=True)

        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_img = tf.image.decode_jpeg(contents)
        raw_img = 255.0 * tf.image.convert_image_dtype(raw_img, dtype=tf.float32)

        image_clip = utils.preprocessing_image(
            raw_img,
            256, 256, 512,
            is_training=True)

        image_batch = tf.train.shuffle_batch([image_clip], batch_size=8, capacity=50000, num_threads=4,
                                             min_after_dequeue=10000)
  
        model = aams.AAMS()
        total_loss = model.build_graph(image_batch)
        
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        scopes = ['self_attention', 'decoder']
        
        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)
        train_op = model.get_training_op(global_step, variables_to_train)
        update_ops.append(train_op)
    
        summaries |= set(model.summaries)

        update_op = tf.group(*update_ops)
        watched_loss = control_flow_ops.with_dependencies([update_op], total_loss, name="train_op")

        # merge the summaries
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        def train_step_fn(session, *args, **kwargs):
            total_loss, should_stop = slim.learning.train_step(session, *args, **kwargs)
            train_step_fn.step += 1
            if train_step_fn.step % 200 == 0:
                texts = ['aams step is :{}, total loss: {}'.format(train_step_fn.step, total_loss)]
                print(texts)
      
            return [total_loss, should_stop]
        
        train_step_fn.step = 0
        print("start learning\n" + '_' * 30)
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8

        slim.learning.train(
            watched_loss,
            train_step_fn = train_step_fn,
            logdir= './tfmodel',
            init_fn=_get_init_fn(),
            summary_op = summary_op,
            number_of_steps= 80000,
            log_every_n_steps= 100,
            save_interval_secs= 600,
            save_summaries_secs= 120,
            session_config= sess_config
            )
 