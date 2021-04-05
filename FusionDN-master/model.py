from __future__ import print_function
import tensorflow as tf
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display

import scipy.io as scio
import time
from datetime import datetime
from scipy.misc import imsave
import scipy.ndimage
from skimage import img_as_ubyte

from Generator import Generator
from LOSS import SSIM_LOSS, Per_LOSS, L1_LOSS, Fro_LOSS
from deepIQA_evaluate import IQA
from VGGnet.vgg16 import Vgg16
WEIGHT_INIT_STDDEV = 0.05

eps = 1e-8

class Model(object):
	def __init__(self, BATCH_SIZE, INPUT_H, INPUT_W, is_training):
		self.batchsize=BATCH_SIZE
		self.G = Generator('Generator')
		self.var_list = []

		self.step = 0
		self.score_f=tf.placeholder(tf.float32,shape=(), name='score_f')

		if not hasattr(self, "ewc_loss"):
			self.Add_loss = 0


		self.SOURCE1 = tf.placeholder(tf.float32, shape = (BATCH_SIZE, INPUT_H, INPUT_W, 1), name = 'SOURCE1')
		self.SOURCE2 = tf.placeholder(tf.float32, shape = (BATCH_SIZE, INPUT_H, INPUT_W, 1), name = 'SOURCE2')
		self.W1 = tf.placeholder(tf.float32, shape = (BATCH_SIZE, 1), name = 'W1')
		self.W2 = tf.placeholder(tf.float32, shape = (BATCH_SIZE, 1), name = 'W2')
		print('source shape:', self.SOURCE1.shape)

		self.generated_img = self.G.transform(I1 = self.SOURCE1, I2 = self.SOURCE2, is_training = is_training)
		self.var_list.extend(tf.trainable_variables())

		''' SSIM loss'''
		SSIM1 = 1-SSIM_LOSS(self.SOURCE1, self.generated_img)
		SSIM2 = 1-SSIM_LOSS(self.SOURCE2, self.generated_img)
		self.ssim_loss = tf.reduce_mean(self.W1 * SSIM1 + self.W2 * SSIM2)


		''' Perceptual loss'''
		with tf.device('/gpu:0'):
			S1_VGG_in = tf.image.resize_nearest_neighbor(self.SOURCE1, size = [224, 224])
			S1_VGG_in = tf.concat((S1_VGG_in, S1_VGG_in, S1_VGG_in), axis = -1)
			S2_VGG_in = tf.image.resize_nearest_neighbor(self.SOURCE2, size = [224, 224])
			S2_VGG_in = tf.concat((S2_VGG_in, S2_VGG_in, S2_VGG_in), axis = -1)
			F_VGG_in = tf.image.resize_nearest_neighbor(self.generated_img, size = [224, 224])
			F_VGG_in = tf.concat((F_VGG_in, F_VGG_in, F_VGG_in), axis = -1)
			self.vgg1 = Vgg16()
			with tf.name_scope("content_vgg"):
				self.S1_FEAS=self.vgg1.build(S1_VGG_in)
			self.vgg2 = Vgg16()
			with tf.name_scope("content_vgg"):
				self.S2_FEAS = self.vgg2.build(S2_VGG_in)
			self.vggF = Vgg16()
			with tf.name_scope("content_vgg"):
				self.F_FEAS = self.vggF.build(F_VGG_in)
			self.perloss_1=0
			self.perloss_2=0
			for i in range(len(self.S1_FEAS)):
				self.perloss_1 += Per_LOSS(self.F_FEAS[i]-self.S1_FEAS[i])
				self.perloss_2 += Per_LOSS(self.F_FEAS[i] - self.S2_FEAS[i])
			self.perloss_1=self.perloss_1/len(self.S1_FEAS)
			self.perloss_2=self.perloss_2/len(self.S2_FEAS)
			self.perloss=tf.reduce_mean(self.W1*self.perloss_1+self.W2*self.perloss_2)


		''' Grad loss'''
		self.grad_loss1 =Fro_LOSS(grad(self.generated_img) - grad(self.SOURCE1))
		self.grad_loss2 =Fro_LOSS(grad(self.generated_img)-grad(self.SOURCE2))
		self.grad_loss = tf.reduce_mean(self.W1 * self.grad_loss1 + self.W2 * self.grad_loss2)

		self.content_loss = self.ssim_loss  + 4e-5 * self.perloss + 1800*self.grad_loss

		# if hasattr(self, "ewc_loss"):
		# 	self.Add_loss = self.ewc_loss - self.content_loss

		self.theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Generator')
		self.clip_G = [p.assign(tf.clip_by_value(p, -8, 8)) for p in self.theta_G]

		# self.var_change=[]
		# for v in range(len(self.var_list)):
		# 	self.var_change.append(np.zeros(self.var_list[v].get_shape().as_list()))


	def compute_fisher(self, imgset, sess, IQA_model, num_samples = 200):
		# computer Fisher information for each parameter
		# initialize Fisher information for most recent task
		self.F_accum = []
		for v in range(len(self.var_list)):
			self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))

		start_time_cf = datetime.now()
		w_iqa = 1
		w_en = 12
		c = 15
		for i in range(num_samples):
			# select random input image
			im_ind = np.random.randint(imgset.shape[0]-self.batchsize)
			# compute first-order derivatives
			s1_index = np.random.choice([0, 1], 1)
			s1 = np.expand_dims(imgset[im_ind:im_ind + self.batchsize, :, :, s1_index[0]], -1)
			s2 = np.expand_dims(imgset[im_ind:im_ind + self.batchsize, :, :, 1 - s1_index[0]], -1)

			with tf.device('/gpu:0'):
				iqa1 = IQA(inputs = s1, trained_model_path = IQA_model)
				iqa2 = IQA(inputs = s2, trained_model_path = IQA_model)

				en1 = EN(s1)
				en2 = EN(s2)
				sco1 = w_iqa * iqa1 + w_en * en1
				sco2 = w_iqa * iqa2 + w_en * en2
				w1 = np.exp(sco1 / c) / (np.exp(sco1 / c) + np.exp(sco2 / c))
				w2 = np.exp(sco2 / c) / (np.exp(sco1 / c) + np.exp(sco2 / c))

			ders = sess.run(tf.gradients(-self.content_loss, self.var_list), feed_dict = {self.SOURCE1: s1, self.SOURCE2: s2, self.W1:w1, self.W2:w2})

			elapsed_time_cf = datetime.now() - start_time_cf
			print("compute fisher: %s/%s, elapsed_time: %s" % (i+1, num_samples, elapsed_time_cf))
			# square the derivatives and add to total

			for v in range(len(self.F_accum)):
				self.F_accum[v] += np.square(ders[v])

		# divide totals by number of samples
		for v in range(len(self.F_accum)):
			self.F_accum[v] /= num_samples


	def star(self):
		# used for saving optimal weights after most recent task training
		self.star_vars = []
		for v in range(len(self.var_list)):
			self.star_vars.append(self.var_list[v].eval())

	def restore(self, sess):
		# reassign optimal weights for latest task
		if hasattr(self, "star_vars"):
			for v in range(len(self.var_list)):
				sess.run(self.var_list[v].assign(self.star_vars[v]))


	def update_ewc_loss(self, lam):
		# elastic weight consolidation
		# lam is weighting for previous task(s) constraints
		if not hasattr(self, "ewc_loss"):
			self.ewc_loss = self.content_loss

		for v in range(len(self.var_list)):
			# self.ewc_loss += (lam / 2) * tf.reduce_sum(
			# 	tf.multiply(self.F_accum[v].astype(np.float32), tf.square(self.var_list[v] - self.star_vars[v])))
			self.Add_loss += tf.reduce_sum(
				tf.multiply(self.F_accum[v].astype(np.float32), tf.square(self.var_list[v] - self.star_vars[v])))
			self.ewc_loss += (lam / 2) *self.Add_loss


def EN(inputs):
	len = inputs.shape[0]
	entropies = np.zeros(shape = (len, 1))
	grey_level = 256
	counter = np.zeros(shape = (grey_level, 1))

	for i in range(len):
		input_uint8 = (inputs[i, :, :, 0] * 255).astype(np.uint8)
		input_uint8 = input_uint8 + 1
		W = inputs.shape[1]
		H = inputs.shape[2]
		for m in range(W):
			for n in range(H):
				indexx = input_uint8[m, n]
				counter[indexx] = counter[indexx] + 1
		total = np.sum(counter)
		p = counter / total
		for k in range(grey_level):
			if p[k] != 0:
				entropies[i] = entropies[i] - p[k] * np.log2(p[k])
	return entropies


def grad(img):
	kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	g = tf.nn.conv2d(img, kernel, strides = [1, 1, 1, 1], padding = 'SAME')
	return g
