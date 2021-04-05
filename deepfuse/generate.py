# Use a trained DeepFuse Net to generate

import tensorflow as tf
import numpy as np

from deep_fuse_net import DeepFuseNet
from utils import get_images, save_images, get_train_images

def generate(content_path, style_path, model_path, model_pre_path, index, output_path=None):

    outputs = _handler(content_path, style_path, model_path, model_pre_path, index, output_path=output_path)
    return list(outputs)


def _handler(content_name, style_name, model_path, model_pre_path, index, output_path=None):
    content_path = content_name
    style_path = style_name

    content_img = get_train_images(content_path, flag=False)
    style_img   = get_train_images(style_path, flag=False)
    dimension = content_img.shape

    content_img = content_img.reshape([1, dimension[0], dimension[1], dimension[2]])
    style_img   = style_img.reshape([1, dimension[0], dimension[1], dimension[2]])

    content_img = np.transpose(content_img, (0, 2, 1, 3))
    style_img = np.transpose(style_img, (0, 2, 1, 3))
    print('content_img shape final:', content_img.shape)

    with tf.Graph().as_default(), tf.Session() as sess:

        # build the dataflow graph
        content = tf.placeholder(
            tf.float32, shape=content_img.shape, name='content')
        style = tf.placeholder(
            tf.float32, shape=style_img.shape, name='style')

        dfn = DeepFuseNet(model_pre_path)

        output_image = dfn.transform_addition(content,style)
        # output_image = dfn.transform_recons(style)
        # output_image = dfn.transform_recons(content)

        # restore the trained model and run the style transferring
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        output = sess.run(output_image, feed_dict={content: content_img, style: style_img})
        save_images(content_path, output, output_path,
                    prefix=index, suffix='_deepfuse_bs2_epoch2')

    return output
