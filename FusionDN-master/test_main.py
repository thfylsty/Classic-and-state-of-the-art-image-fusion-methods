from __future__ import print_function

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.ndimage
# from Generator_DenseRes import Generator
from scipy.misc import imread, imsave

from skimage import transform, data
from glob import glob
from model import Model

LAM=400

MODEL_SAVE_PATH = './models_withoutadd/lam'+str(LAM)+'/task3/'
model_num = 1900

output_path='./results/lam'+str(LAM)+'/far_near/'
path = './test_imgs/far_near/'
path1 = path + 'far_Y'
path2 = path + 'near_Y'

# output_path='./results/lam'+str(LAM)+'/vis_ir/'
# path = './test_imgs/vis_ir/'
# path1 = path + 'vis_gray'
# path2 = path + 'ir'

# output_path='./results/lam'+str(LAM)+'/oe_ue/'
# path = './test_imgs/oe_ue/'
# path1 = path + 'oe_Y'
# path2 = path + 'ue_Y'

# output_path='./results/lam'+str(LAM)+'/medical_only/'
# path = './test_imgs/pet_mri/'
# path1 = path + 'pet_Y'
# path2 = path + 'mri'
def main():
	print('\nBegin to generate pictures ...\n')
	for i in range(50):
		if i<29:
			continue
		file_name1 = '../road/vi/' + str(i + 1) + '.jpg'
		file_name2 =  '../road/ir/' + str(i + 1) + '.jpg'

		img1 = imread(file_name1) / 255.0
		img2 = imread(file_name2) / 255.0
		print('file1:', file_name1)
		print('file2:', file_name2)

		# imsave(output_path + str(i+1) + '_pet' + Format, img1)
		# imsave(output_path + str(i+1) + '_mri' + Format, img2)

		Shape1 = img1.shape
		if len(Shape1)>2:
			img1 = img1[:,:,0] * 0.3 + img1[:,:,1] * 0.59 + img1[:,:,2] * 0.11
		Shape2 = img2.shape
		h=Shape2[0]
		w=Shape2[1]
		if len(Shape2)>2:
			img2 = img2[:,:,0] * 0.3 + img2[:,:,1] * 0.59 + img2[:,:,2] * 0.11
		img1 = transform.resize(img1, (h, w))
		img2 = transform.resize(img2, (h, w))
		img1 = img1.reshape([1, h, w, 1])
		img2 = img2.reshape([1, h, w, 1])

		with tf.Graph().as_default(), tf.Session() as sess:
			# SOURCE1 = tf.placeholder(tf.float32, shape = shape, name = 'SOURCE1')
			# SOURCE2 = tf.placeholder(tf.float32, shape = shape, name = 'SOURCE2')
			# print('SOURCE1 shape:', SOURCE1.shape)


			M = Model(BATCH_SIZE=1, INPUT_H=h, INPUT_W=w, is_training=False)

			# G = Generator('Generator')
			# output_image= G.transform(I1=SOURCE1, I2=SOURCE2)

			# restore the trained model and run the style transferring
			g_list = tf.global_variables()
			# for i in g_list:
			# 	print(i.name)
			# g_list=tf.trainable_variables()

			saver = tf.train.Saver(var_list = g_list)
			model_save_path = 'model/model.ckpt'
			print(model_save_path)
			sess.run(tf.global_variables_initializer())
			saver.restore(sess, model_save_path)
			output = sess.run(M.generated_img, feed_dict = {M.SOURCE1: img1, M.SOURCE2: img2})

			output=output[0,:,:,0]

			# fig=plt.figure()
			# f1=fig.add_subplot(311)
			# f2=fig.add_subplot(312)
			# f3=fig.add_subplot(313)
			# f1.imshow(img1[0, :, :, 0], cmap='gray')
			# f2.imshow(img2[0, :, :, 0], cmap='gray')
			# f3.imshow(output, cmap='gray')
			# plt.show()

			if not os.path.exists(output_path):
				os.makedirs(output_path)
			# imsave(output_path+ str(i+11)+ '_task3_'+str(model_num)+ Format, output)
			imsave('result/'+str(i+1)+'.bmp', output)


			del M

if __name__ == '__main__':
	main()
