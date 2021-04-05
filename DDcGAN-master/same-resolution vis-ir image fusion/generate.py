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

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Graph().as_default(), tf.Session(config=config) as sess:
		SOURCE_VIS = tf.placeholder(tf.float32, shape = vis_dimension, name = 'SOURCE_VIS')
		SOURCE_ir = tf.placeholder(tf.float32, shape = ir_dimension, name = 'SOURCE_ir')
		# source_field = tf.placeholder(tf.float32, shape = source_shape, name = 'source_imgs')

		G = Generator('Generator')
		output_image = G.transform(vis = SOURCE_VIS, ir = SOURCE_ir)
		# D1 = Discriminator1('Discriminator1')
		# D2 = Discriminator2('Discriminator2')

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)
		output = sess.run(output_image, feed_dict = {SOURCE_VIS: vis_img, SOURCE_ir: ir_img})
		output = output[0, :, :, 0]
		imsave("result/" + str(index) + '.bmp', output)


# print('generated image shape:', output_image.shape)

# imsave(output_path + str(index) + '/' + str(model_num) + '_ir_us.bmp', IR[0, :, :, 0])
# imsave(output_path + str(index) + '/' + str(model_num) + '_vis_de.bmp', vis_de[0, :, :, 0])


def save_images(paths, datas, save_path, prefix = None, suffix = None):
	if isinstance(paths, str):
		paths = [paths]

	assert (len(paths) == len(datas))

	if not exists(save_path):
		mkdir(save_path)

	if prefix is None:
		prefix = ''
	if suffix is None:
		suffix = ''

	for i, path in enumerate(paths):
		data = datas[i]
		# print('data ==>>\n', data)
		if data.shape[2] == 1:
			data = data.reshape([data.shape[0], data.shape[1]])
		# print('data reshape==>>\n', data)

		name, ext = splitext(path)
		name = name.split(sep)[-1]

		path = join(save_path, prefix + suffix + ext)
		print('data path==>>', path)
		imsave(path, data)
