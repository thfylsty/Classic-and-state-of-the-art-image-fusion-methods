import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np


WEIGHT_INIT_STDDEV = 0.05


class Generator(object):

	def __init__(self, sco):
		self.encoder = Encoder(sco)
		self.decoder = Decoder(sco)
		self.var_list=[]

	def transform(self, I1, I2, is_training):
		img = tf.concat([I1, I2], 3)
		code = self.encoder.encode(img, is_training)
		self.target_features = code
		generated_img = self.decoder.decode(self.target_features, is_training)

		self.var_list.extend(self.encoder.var_list)
		self.var_list.extend(self.decoder.var_list)

		return generated_img


class Encoder(object):
	def __init__(self, scope_name):
		self.scope = scope_name
		self.var_list=[]


	def encode(self, image, is_training):
		shape = [3, 3, 2, 48]
		with tf.variable_scope(self.scope):
			with tf.variable_scope('encoder'):
				kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV), name = 'kernel')
				bias = tf.Variable(tf.zeros(shape[-1]), name = 'bias')
				self.var_list.append(kernel)
				self.var_list.append(bias)

		out = image
		out = conv2d(out, kernel, bias, use_relu = True, is_training = is_training, dense=False,
		             Scope = self.scope + '/encoder/conv1', BN=True)

		out = tf.concat([out, self.residual_block(out, ch=48, scope = self.scope + '/encoder/res_block_conv1')], 3)
		out = tf.concat([out, self.residual_block(out, ch=48, scope = self.scope + '/encoder/res_block_conv2')], 3)
		out = tf.concat([out, self.residual_block(out, ch=48, scope = self.scope + '/encoder/res_block_conv3')], 3)
		out = tf.concat([out, self.residual_block(out, ch = 48, scope = self.scope + '/encoder/res_block_conv4')], 3)

		return out

	def residual_block(self, input, ch, scope):
		with tf.variable_scope(scope):
			W1 = tf.Variable(tf.truncated_normal([3, 3, int(input.shape[3]), ch], stddev = tf.sqrt(2 / ch)), dtype = np.float32, name = 'kernel1')
			W2 = tf.Variable(tf.truncated_normal([3, 3, ch, ch], stddev = tf.sqrt(2 / ch)), dtype = np.float32, name = 'kernel2')
			B1 = tf.Variable(tf.zeros([ch]), name = 'bias1')
			B2 = tf.Variable(tf.zeros([ch]), name = 'bias2')

			self.var_list.append(W1)
			self.var_list.append(B1)
			self.var_list.append(W2)
			self.var_list.append(B2)

		x_padded = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
		L1 = tf.nn.conv2d(input=x_padded, filter=W1, strides = [1, 1, 1, 1], padding = 'VALID')
		L1 = tf.nn.bias_add(L1, B1)
		with tf.variable_scope(scope + '/b1/'):
			L1 = tf.layers.batch_normalization(L1)
		L1 = tf.nn.relu(L1)

		# tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(W2), self.wd))
		L1_padded = tf.pad(L1, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
		L2 = tf.nn.conv2d(input=L1_padded, filter=W2, strides = [1, 1, 1, 1], padding = 'VALID')
		L2 = tf.nn.bias_add(L2, B2)
		with tf.variable_scope(scope + '/b2/'):
			L2 = tf.layers.batch_normalization(L2)
		# L3 = tf.add(L2, L1)
		L3 = tf.nn.relu(L2)
		return L3


class Decoder(object):
	def __init__(self, scope_name):
		self.weight_vars = []
		self.var_list = []
		self.scope = scope_name
		with tf.name_scope(scope_name):
			with tf.variable_scope('decoder'):
				self.weight_vars.append(self._create_variables(240, 240, 3, scope = 'conv2_1'))
				self.weight_vars.append(self._create_variables(240, 128, 3, scope = 'conv2_2'))
				self.weight_vars.append(self._create_variables(128, 64, 3, scope = 'conv2_3'))
				self.weight_vars.append(self._create_variables(64, 1, 3, scope = 'conv2_4'))
				# self.weight_vars.append(self._create_variables(32, 1, 3, scope = 'conv2_4'))

	def _create_variables(self, input_filters, output_filters, kernel_size, scope):
		with tf.variable_scope(scope):
			shape = [kernel_size, kernel_size, input_filters, output_filters]
			kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV), name = 'kernel')
			bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
			self.var_list.append(kernel)
			self.var_list.append(bias)
		return (kernel, bias)

	def decode(self, image, is_training):
		final_layer_idx = len(self.weight_vars) - 1

		out = image
		for i in range(len(self.weight_vars)):
			kernel, bias = self.weight_vars[i]
			if i == 0:
				out = conv2d(out, kernel, bias, dense = False, use_relu = True,
				             Scope = self.scope + '/decoder/b' + str(i), BN = False, is_training = is_training)
			if i == final_layer_idx:
				out = conv2d(out, kernel, bias, dense = False, use_relu = False,
				             Scope = self.scope + '/decoder/b' + str(i), BN = False, is_training = is_training)
				out = tf.nn.tanh(out) / 2 + 0.5
			else:
				out = conv2d(out, kernel, bias, dense = False, use_relu = True, BN = True,
				             Scope = self.scope + '/decoder/b' + str(i), is_training = is_training)
		return out


def conv2d(x, kernel, bias, use_relu = True, dense=False, Scope = None, BN = True, is_training = False):
	# padding image with reflection mode
	x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
	# conv and add bias
	out = tf.nn.conv2d(input = x_padded, filter = kernel, strides = [1, 1, 1, 1], padding = 'VALID')
	out = tf.nn.bias_add(out, bias)
	if BN:
		with tf.variable_scope(Scope):
			out = tf.layers.batch_normalization(out, training = True)
	if use_relu:
		out = tf.nn.relu(out)
		# out=tf.maximum(out, 0.2 * out)
	if dense:
		out = tf.concat([out, x], 3)
	return out


