import tensorflow as tf
import numpy as npimport

import tensorflow as tf

from tensorflow.python import pywrap_tensorflow
import numpy as np

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np

WEIGHT_INIT_STDDEV = 0.1

device = "/gpu:0"

class Discriminator(object):
	def __init__(self, scope_name):
		self.weight_vars = []
		self.scope = scope_name
		with tf.variable_scope(scope_name):
			self.weight_vars.append(self._create_variables(3, 64, 3, scope = 'conv1'))
			self.weight_vars.append(self._create_variables(64, 64, 3, scope = 'conv2'))
			self.weight_vars.append(self._create_variables(64, 96, 3, scope = 'conv3'))
			self.weight_vars.append(self._create_variables(96, 128, 3, scope = 'conv4'))
			self.weight_vars.append(self._create_variables(128, 256, 3, scope = 'conv5'))
			self.weight_vars.append(self._create_variables(256, 512, 3, scope = 'conv6'))
			# self.weight_vars.append(self._create_variables(128, 256, 3, scope = 'conv5'))
			# self.weight_vars.append(self._create_variables(256, 512, 3, scope = 'conv6'))
			# self.weight_vars.append(self._create_variables(512, 512, 3, scope = 'conv7'))

		# self.weight_vars.append(self._create_variables(12, 1, 3, scope = 'conv6'))

	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		shape = [kernel_size, kernel_size, input_filters, output_filters]
		with tf.device("/cpu:0"):
			with tf.variable_scope(scope):
				kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV), name = 'kernel')
				bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
		return (kernel, bias)

	def discrim(self, img, reuse):
		with tf.device(device):
			conv_num = len(self.weight_vars)
			if len(img.shape) != 4:
				img = tf.expand_dims(img, -1)
			out = img
			for i in range(conv_num):
				kernel, bias = self.weight_vars[i]
				if i == 0:
					out = conv2d(out, kernel, bias, [1, 2, 2, 1], use_relu = True, use_BN = False, sn=True,
					               Scope = self.scope + '/b' + str(i), Reuse = reuse)
				# elif i == conv_num - 1:
				# 	out = tf.nn.conv2d(out, kernel, [1, 1, 1, 1], padding = 'VALID')
				# 	out = tf.nn.bias_add(out, bias)
				# 	out = tf.nn.tanh(out)
				# 	out = out / 2 + 0.5
				elif i == conv_num - 1:
					out = conv2d(out, kernel, bias, [1, 1, 1, 1], use_relu = True, use_BN = False, sn= False,
					               Scope = self.scope + '/b' + str(i), Reuse = reuse)
				else:
					out = conv2d(out, kernel, bias, [1, 2, 2, 1], use_relu = True, use_BN = False, sn = True,
					             Scope = self.scope + '/b' + str(i), Reuse = reuse)

			# out = self_attention(out, channel_factor = 8, scope_name=self.scope, name = 'self_attention4', reuse= reuse)
			out = tf.reshape(out, [-1, int(out.shape[1]) * int(out.shape[2]) * int(out.shape[3])])

			with tf.variable_scope(self.scope):
				with tf.variable_scope('flatten1'):
					# 	out = tf.layers.dense(out, 512, activation = tf.nn.relu, use_bias = True, trainable = True,
					# 	# 	                      reuse = reuse)
					# 	out = tf.layers.batch_normalization(out, training = True, reuse = reuse)
					# 	# with tf.variable_scope('flatten2'):
					out = tf.layers.dense(out, 1, activation = tf.nn.tanh, use_bias = True, trainable = True,
					                      reuse = reuse)
			out = out / 2 + 0.5
		return out


def conv2d(x, kernel, bias, strides, use_relu = True, use_BN = True, Scope = None, sn=True, Reuse = None):
	with tf.device(device):
		# padding image with reflection mode
		x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
		# conv and add bias
		if sn:
			out = tf.nn.conv2d(input=x_padded, filter = spectral_norm(kernel, scope_name = Scope, reuse=Reuse), strides=strides, padding = 'VALID')
		else:
			out = tf.nn.conv2d(input = x_padded, filter = kernel, strides = strides, padding = 'VALID')
		out = tf.nn.bias_add(out, bias)
		if use_BN:
			with tf.variable_scope(Scope):
				out = tf.layers.batch_normalization(out, training = True, reuse = Reuse)
		if use_relu:
			# out = tf.nn.relu(out)
			out = tf.maximum(out, 0.2 * out)
	return out



def self_attention(inputs, channel_factor = 8, scope_name = None, name = 'self_attention', reuse=False):
	num_filters = inputs.shape[-1].value // channel_factor
	with tf.variable_scope(scope_name):
		with tf.variable_scope(name, reuse= reuse):
			flat_inputs = tf.reshape(inputs, shape = [int(inputs.shape[0]), int(inputs.shape[1])*int(inputs.shape[2]), int(inputs.shape[-1])])
			print('flat_inputs shape:', flat_inputs.shape)
			f = tf.layers.conv1d(flat_inputs, kernel_size = 1, filters = num_filters)
			g = tf.layers.conv1d(flat_inputs, kernel_size = 1, filters = num_filters)
			h = tf.layers.conv1d(flat_inputs, kernel_size = 1, filters = inputs.shape[-1])
			beta = tf.nn.softmax(tf.matmul(f, g, transpose_b = True))
			o = tf.matmul(beta, h)
			gamma = tf.get_variable('gamma', [], initializer = tf.zeros_initializer)
			y = gamma * o + flat_inputs
			y = tf.reshape(y, inputs.shape)
			print('attention output shape:', y.shape)
	return inputs


def spectral_norm(w, iteration = 1, scope_name = None, reuse=False):
	w_shape = w.shape.as_list()
	w = tf.reshape(w, [-1, w_shape[-1]])

	with tf.variable_scope(scope_name, reuse=reuse):
		u = tf.get_variable("u", [1, w_shape[-1]], initializer = tf.truncated_normal_initializer(), trainable = False)

	u_hat = u
	v_hat = None
	for i in range(1):
		"""
		power iteration
		Usually iteration = 1 will be enough
		"""
		v_ = tf.matmul(u_hat, tf.transpose(w))
		v_hat = l2_norm(v_)

		u_ = tf.matmul(v_hat, w)
		u_hat = l2_norm(u_)

	sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
	w_norm = w / sigma

	with tf.control_dependencies([u.assign(u_hat)]):
		w_norm = tf.reshape(w_norm, w_shape)

	return w_norm


def l2_norm(v, eps=1e-12):
	return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

