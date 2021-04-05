# Use a trained DenseFuse Net to generate fused images

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imsave
from datetime import datetime
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
from Generator import Generator
from Discriminator import Discriminator1, Discriminator2
import time


def generate(ir_path, vis_path, model_path, index, output_path = None):
	ir_img = imread(ir_path) / 255.0
	vis_img = imread(vis_path) / 255.0
	ir_dimension = list(ir_img.shape)
	vis_dimension = list(vis_img.shape)
	ir_dimension.insert(0, 1)
	ir_dimension.append(1)
	vis_dimension.insert(0, 1)
	vis_dimension.append(1)
	ir_img = ir_img.reshape(ir_dimension)
	vis_img = vis_img.reshape(vis_dimension)

	with tf.Graph().as_default(), tf.Session() as sess:

		SOURCE_VIS = tf.placeholder(tf.float32, shape = vis_dimension, name = 'SOURCE_VIS')
		SOURCE_ir = tf.placeholder(tf.float32, shape = ir_dimension, name = 'SOURCE_ir')

		G = Generator('Generator')
		output_image = G.transform(vis = SOURCE_VIS, ir = SOURCE_ir)
		# D1 = Discriminator1('Discriminator1')
		# D2 = Discriminator2('Discriminator2')

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)
		output = sess.run(output_image, feed_dict = {SOURCE_VIS: vis_img, SOURCE_ir: ir_img})
		output = output[0, :, :, 0]
		imsave(output_path + str(index) + '.bmp', output)
