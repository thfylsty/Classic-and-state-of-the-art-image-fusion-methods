import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import h5py
from scipy.misc import imread, imsave
import math
import tensorflow as tf

# f = h5py.File('Dataset2.h5', 'r')
# # for key in f.keys():
# #   print(f[key].name)
# a = f['data'][:]
# sources = np.transpose(a, (0, 3, 2, 1))
#
# vis = sources[100, :, :, 0]
# ir = sources[100, :, :, 1]
#
# ir_ds = scipy.ndimage.zoom(ir, 0.25)
# ir_ds_us = scipy.ndimage.zoom(ir_ds, 4, order = 3)
#
# fig = plt.figure()
# V = fig.add_subplot(221)
# I = fig.add_subplot(222)
# I_ds = fig.add_subplot(223)
# I_ds_us = fig.add_subplot(224)
#
# V.imshow(vis, cmap = 'gray')
# I.imshow(ir, cmap = 'gray')
# I_ds.imshow(ir_ds, cmap = 'gray')
# I_ds_us.imshow(ir_ds_us, cmap = 'gray')
# plt.show()


# path = 'C:/Users/Administrator/Desktop/DualDGAN(MR)/测指标/test_imgs/'
# for i in range(20):
# 	VIS = imread(path + 'VIS' + str(i + 1) + '.bmp')
# 	IR = imread(path + 'IR' + str(i + 1) + '.bmp')
# 	[m, n] = VIS.shape
# 	VIS = VIS[0:math.floor(m / 4) * 4, 0:math.floor(n / 4) * 4]
# 	IR = IR[0:math.floor(m / 4) * 4, 0:math.floor(n / 4) * 4]
# 	imsave(path + 'VIS' + str(i + 1) + '.bmp', VIS)
# 	imsave(path + 'IR' + str(i + 1) + '.bmp', IR)
# 	IR_ds = scipy.ndimage.zoom(IR, 0.25)
# 	IR_ds_us = scipy.ndimage.zoom(IR_ds, 4, order = 3)
# 	# plt.imshow(VIS)
# 	# plt.show()
# 	imsave(path + 'IR' + str(i + 1) + '_ds.bmp', IR_ds)
# 	imsave(path + 'IR' + str(i + 1) + '_ds_us.bmp', IR_ds_us)


ROW = 268
COL = 360
IMG = tf.placeholder(tf.float32, shape = (1, ROW, COL, 1))
output1 = tf.image.resize_images(IMG, (int(ROW / 4), int(COL / 4)), method = 2)
output = tf.nn.avg_pool(IMG, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')
output2 = tf.nn.avg_pool(output, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')
outputo = tf.nn.max_pool(IMG, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')
output3 = tf.nn.max_pool(outputo, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')
path = 'C:/Users/Administrator/Desktop/DualDGAN(MR)/测指标/test_imgs/'
img = imread(path + 'VIS1.bmp') / 255.0
img1 = np.expand_dims(np.expand_dims(img, -1), 0)

with tf.Session() as sess:
	o2 = sess.run(output2, feed_dict = {IMG: img1})
	o3 = sess.run(output3, feed_dict = {IMG: img1})
	o2 = o2[0, :, :, 0]
	o3 = o3[0, :, :, 0]
	fig = plt.figure()
	V = fig.add_subplot(131)
	I = fig.add_subplot(132)
	IA = fig.add_subplot(133)
	V.imshow(img, cmap = 'gray')
	I.imshow(o2, cmap = 'gray')
	IA.imshow(o3, cmap = 'gray')
	plt.show()
