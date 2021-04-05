# Use a trained DenseFuse Net to generate fused images
import numpy as np
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imsave
from datetime import datetime
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
from Generator import Generator
import time
import matplotlib.pyplot as plt


from skimage import transform, data
import scipy.io as scio




def generate(oe_path, ue_path, model_path, index, output_path = None, format=None):
	oe_img = imread(oe_path) / 255.0
	ue_img = imread(ue_path) / 255.0
	# gt_img=imread(gt_path)/255.0

	# H, W, C = oe_img.shape
	# h=(H//3) #//8 * 8
	# w=(W//3) #//8 * 8
	# oe_img = transform.resize(oe_img, (h, w))
	# # ue_img = transform.resize(ue_img, (h, w))
	# # gt_img = transform.resize(gt_img, (h, w))
	# path = '/home/jxy/xh/project/EPF/40/'
	# Format = '.JPG'
	# imsave(path + 'g' + Format, oe_img)
	#
	# # imsave(path + 'o' + str(index) + Format, oe_img)
	# # imsave(path + 'u' + str(index) + Format, ue_img)
	# # imsave(path + 'g' + str(index) + Format, gt_img)

	oe_img = np.expand_dims(oe_img, axis=2)
	oe_img = np.concatenate((oe_img, oe_img, oe_img), axis=-1)
	ue_img = np.expand_dims(ue_img, axis=2)
	ue_img = np.concatenate((ue_img, ue_img, ue_img), axis=-1)

	H, W, C = oe_img.shape
	h = H // 8 * 8
	w = W // 8 * 8
	oe_img = oe_img[0:h, 0:w, :]
	ue_img = ue_img[0:h, 0:w, :]
	oe_img = oe_img.reshape([1, h, w, C])
	ue_img = ue_img.reshape([1, h, w, C])
	shape = oe_img.shape
	print('oe img shape', oe_img.shape)

	start=time.time()

	with tf.Graph().as_default(), tf.Session() as sess:
		SOURCE_oe = tf.placeholder(tf.float32, shape = shape, name = 'SOURCE_oe')
		SOURCE_ue = tf.placeholder(tf.float32, shape = shape, name = 'SOURCE_ue')

		print('SOURCE_oe shape:', SOURCE_oe.shape)

		G = Generator('Generator')
		output_image = G.transform(oe_img=SOURCE_oe, ue_img=SOURCE_ue, is_training=False)

		# restore the trained model and run the style transferring
		g_list = tf.global_variables()
		# for g in g_list:
		# 	print(g.name)
		saver = tf.train.Saver(var_list = g_list)

		model_save_path = model_path +  'model.ckpt'
		print(model_save_path)
		saver.restore(sess, model_save_path)

		output = sess.run(output_image, feed_dict = {SOURCE_oe: oe_img, SOURCE_ue: ue_img})
		output=output[0,:,:,0]*0.299 +output[0,:,:,1]*0.587+output[0,:,:,2]*0.114
		imsave("result/" + str(index + 1) + '.bmp', output)
		# imsave(output_path + str(index)  + format, output)

		end=time.time()
		return (end-start)