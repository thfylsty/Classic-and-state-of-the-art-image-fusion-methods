import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
# from Deconv import deconv_vis, deconv_ir

WEIGHT_INIT_STDDEV = 0.5

device1 = "/gpu:0"
device2 = "/gpu:0"

class Generator(object):

	def __init__(self, sco):
		self.local = Local(sco)
		self.sa = Sa_net(sco)
		self.merge = Merge_net(sco)
		self.scope_name = sco

	def transform(self, oe_img, ue_img, is_training):
		img = tf.concat([oe_img, ue_img], 3)
		local_feature = self.local.local_generate(img, is_training)
		sa_feature = self.sa.sa_generate(img, is_training)
		feature = tf.concat([local_feature, sa_feature], axis=-1)
		generated_img = self.merge.merge(feature, is_training)
		return generated_img


class Local(object):
	def __init__(self, scope_name):
		self.scope = scope_name

	def _create_variables(self, shape, scope):
		# with tf.device("/cpu:0"):
		with tf.variable_scope(self.scope):
			with tf.variable_scope('Local_net'):
				with tf.variable_scope(scope):
					kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV), name = 'kernel')
					bias = tf.Variable(tf.zeros(shape[-1]), name = 'bias')
		return (kernel, bias)

	def local_generate(self, image, is_training):
		out = image
		shape1 = [3, 3, 6, 40]
		kernel1, bias1 = self._create_variables(shape1, scope='conv1')
		out1 = conv2d(out, kernel1, bias1, use_relu = True, sn = False, is_training = is_training, Scope = self.scope + '/Local_net/conv1/b')

		shape2 = [3, 3, 40, 40]
		kernel2, bias2 = self._create_variables(shape2, scope = 'conv2')
		out2 = conv2d(out1, kernel2, bias2, use_relu = True, sn = True, is_training = is_training,
		              Scope = self.scope + '/Local_net/conv2/b')


		shape3 = [3, 3, 80, 40]
		kernel3, bias3 = self._create_variables(shape3, scope = 'conv3')
		out3 = conv2d(tf.concat([out1, out2], axis=-1), kernel3, bias3, use_relu = True, sn = True, is_training = is_training,
		              Scope = self.scope + '/Local_net/conv3/b')

		shape4 = [3, 3, 120, 40]
		kernel4, bias4 = self._create_variables(shape4, scope = 'conv4')
		out4 = conv2d(tf.concat([out1, out2, out3], axis=-1), kernel4, bias4, use_relu = True, sn = True, is_training = is_training,
		              Scope = self.scope + '/Local_net/conv4/b')

		shape5 = [3, 3, 160, 40]
		kernel5, bias5 = self._create_variables(shape5, scope = 'conv5')
		out5 = conv2d(tf.concat([out1, out2, out3, out4], axis = -1), kernel5, bias5, use_relu = True, sn = True,
		              is_training = is_training,
		              Scope = self.scope + '/Local_net/conv5/b')

		out=tf.concat([out1, out2, out3, out4, out5], axis=-1)



		# shape = [3, 3, 32, 64]
		# kernel, bias = self._create_variables(shape, scope = 'conv2')
		# out = conv2d(out, kernel, bias, use_relu = True, sn = True, is_training = is_training, Scope = self.scope + '/Local_net/conv2/b')

		# out = residual_block(out, 164, is_training = is_training, scope = self.scope + '/Local_net/res_block_conv1')
		# out = residual_block(out, 164, is_training = is_training, scope = self.scope + '/Local_net/res_block_conv2')
		# out = residual_block(out, 48, is_training = is_training, scope = self.scope + '/Local_net/res_block_conv3')
		# out = residual_block(out, 64, is_training = is_training, scope = self.scope + '/Local_net/res_block_conv4')
		# out = residual_block(out, 64, is_training = is_training, scope = self.scope + '/Local_net/res_block_conv5')

		# shape = [3, 3, 164, 196]
		# kernel, bias = self._create_variables(shape, scope = 'conv3')
		# out = conv2d(out, kernel, bias, use_relu = True, sn = True, is_training = is_training, Scope = self.scope + '/Local_net/conv3/b')

		# shape = [3, 3, 128, 256]
		# kernel, bias = self._create_variables(shape, scope = 'conv4')
		# out = conv2d(out, kernel, bias, use_relu = True, sn = True, is_training = is_training, Scope = self.scope + '/Local_net/conv4/b')
		#
		#
		# shape = [3, 3, 32, 3]
		# kernel, bias = self._create_variables(shape, scope = 'conv5')
		# out = conv2d(out, kernel, bias, use_relu = True, sn = False, is_training = is_training, Scope = self.scope + '/Local_net/conv5/b')
		return out



class Sa_net(object):
	def __init__(self, scope_name):
		self.scope = scope_name

	def _create_variables(self, shape, scope):
		# with tf.device("/cpu:0"):
		with tf.variable_scope(self.scope):
			with tf.variable_scope('Sa_net'):
				with tf.variable_scope(scope):
					kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV), name = 'kernel')
					bias = tf.Variable(tf.zeros(shape[-1]), name = 'bias')
		return (kernel, bias)

	def _create_de_variables(self, shape, scope):
		# with tf.device("/cpu:0"):
		with tf.variable_scope(self.scope):
			with tf.variable_scope('Sa_net'):
				with tf.variable_scope(scope):
					kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV), name = 'kernel')
		return kernel


	def sa_generate(self, image, is_training):
		out = image
		# encoder

		shape = [3, 3, 6, 16]
		kernel, bias = self._create_variables(shape, scope = 'encoder/conv1')
		out = sa_conv2d(out, kernel, bias, use_relu = True, sn = False, is_training = is_training, Scope = self.scope + '/Sa_net/encoder/conv1/b')
		out = tf.nn.max_pool(out, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')

		shape = [3, 3, 16, 32]
		kernel, bias = self._create_variables(shape, scope = 'encoder/conv2')
		out = sa_conv2d(out, kernel, bias, use_relu = True, sn = True, is_training = is_training, Scope = self.scope + '/Sa_net/encoder/conv2/b')
		# out = tf.nn.max_pool(out, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')

		# print('Sa encoder shape:', out.shape)

		## out = self_attention(inputs = out, channel_factor = 8, scope_name = self.scope, name = "self-attention")

		out, attention_map = attention(x=out, ch=out.shape[-1].value, sn = True, scope_name = self.scope, name = 'self_attention')
		# print('Sa self attention shape:', out.shape)

		out = up_sample(out, 2)

		shape = [3, 3, 32, 20]
		kernel, bias = self._create_variables(shape, scope = 'decoder/conv1')
		out = sa_conv2d(out, kernel, bias, use_relu = True, sn = True, is_training = is_training, Scope = self.scope + '/Sa_net/decoder/conv1/b', strides = [1, 1, 1, 1])
		# print('Sa decoder conv1 shape:', out.shape)

		out = up_sample(out, 2)

		shape = [3, 3, 20, 16]
		kernel, bias = self._create_variables(shape, scope = 'decoder/conv2')
		out = sa_conv2d(out, kernel, bias, use_relu = True, sn = True, is_training = is_training, Scope = self.scope + '/Sa_net/decoder/conv2/b', strides = [1, 1, 1, 1])
		# print('Sa decoder conv2 shape:', out.shape)

		out = up_sample(out, 2)
		# out = up_sample(out, 2)

		# shape = [3, 3, 32, 16]
		# kernel, bias = self._create_variables(shape, scope = 'decoder/conv2')
		# out = sa_conv2d(out, kernel, bias, use_relu = True, sn = False, is_training = is_training, Scope = self.scope + '/Sa_net/decoder/conv2/b', strides = [1, 1, 1, 1])
		# print('Sa decoder conv2 shape:', out.shape)

		# ks = 3
		# shape = [ks, ks, 64, 128]
		# kernel = self._create_de_variables(shape, scope = 'decoder/deconv1')
		# out = sa_deconv2d(out, kernel, strides = [1, 4, 4, 1])
		# print('Sa decoder deconv1 shape:', out.shape)
		return out


class Merge_net(object):
	def __init__(self, scope_name):
		self.scope = scope_name

	def _create_variables(self, shape, scope):
		# with tf.device("/cpu:0"):
		with tf.variable_scope(self.scope):
			with tf.variable_scope('Merge_net'):
				with tf.variable_scope(scope):
					kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV), name = 'kernel')
					bias = tf.Variable(tf.zeros(shape[-1]), name = 'bias')
		return (kernel, bias)

	def merge(self, feature, is_training):
		# print('feature shape:', feature.shape)
		out = feature
		shape = [3, 3, 216, 128] #192
		kernel, bias = self._create_variables(shape, scope='conv1')
		out = conv2d(out, kernel, bias, use_relu = True, sn = True, is_training = is_training, Scope = self.scope + '/Merge_net/conv1/b')

		# shape = [3, 3, 196, 128]
		# kernel, bias = self._create_variables(shape, scope = 'conv2')
		# out = conv2d(out, kernel, bias, use_relu = True, sn = True, is_training = is_training, Scope = self.scope + '/Merge_net/conv2/b')

		shape = [3, 3, 128, 64]
		kernel, bias = self._create_variables(shape, scope = 'conv3')
		out = conv2d(out, kernel, bias, use_relu = True, sn = True, is_training = is_training, Scope = self.scope + '/Merge_net/conv3/b')

		shape = [3, 3, 64, 3]
		kernel, bias = self._create_variables(shape, scope = 'conv4')
		out = conv2d(out, kernel, bias, use_relu = False, sn = False, is_training = is_training, Scope = self.scope + '/Merge_net/conv4/b')


		out = tf.nn.tanh(out) / 2 + 0.5

		# print('Merge output shape:', out.shape)
		return out


def residual_block(input, ch, is_training, scope):
	# with tf.device("/cpu:0"):
	with tf.variable_scope(scope):
		W1 = tf.Variable(tf.truncated_normal([3, 3, ch, ch], stddev = tf.sqrt(2 / ch)), dtype = np.float32, name = 'kernel1')
	with tf.device(device1):
		# tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(W1), self.wd))
		x_padded = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
		L1 = tf.nn.conv2d(input=x_padded, filter=spectral_norm(W1, scope_name = scope+'/sn1'), strides = [1, 1, 1, 1], padding = 'VALID')

		with tf.variable_scope(scope + '/sn_b1/'):
			L1 = tf.layers.batch_normalization(L1, training = is_training)
		L1 = tf.nn.relu(L1)

	# with tf.device("/cpu:0"):
	with tf.variable_scope(scope):
		W2 = tf.Variable(tf.truncated_normal([3, 3, ch, ch], stddev = tf.sqrt(2 / ch)), dtype = np.float32, name = 'kernel2')

	with tf.device(device1):
		# tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(W2), self.wd))
		L1_padded = tf.pad(L1, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
		L2 = tf.nn.conv2d(input=L1_padded, filter=spectral_norm(W2, scope_name = scope+'/sn2'), strides = [1, 1, 1, 1], padding = 'VALID')
		with tf.variable_scope(scope + '/sn_b2/'):
			L2 = tf.layers.batch_normalization(L2, training = is_training)

		L3 = tf.add(L2, input)
		L3 = tf.nn.relu(L3)
	return L3


def conv2d(x, kernel, bias, use_relu = True, Scope = None, BN = True, sn = True, is_training = False):
	with tf.device(device1):
		# padding image with reflection mode
		x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
		# conv and add bias
		if sn:
			out = tf.nn.conv2d(input = x_padded, filter = spectral_norm(kernel, scope_name = Scope), strides = [1, 1, 1, 1],
			                   padding = 'VALID')
		else:
			out = tf.nn.conv2d(input = x_padded, filter = kernel, strides = [1, 1, 1, 1], padding = 'VALID')
		out = tf.nn.bias_add(out, bias)
		if BN:
			with tf.variable_scope(Scope):
				out = tf.layers.batch_normalization(out, training = is_training)
		if use_relu:
			#out = tf.nn.relu(out)
			out = tf.maximum(out, 0.2 * out)
	return out


def sa_conv2d(x, kernel, bias, use_relu = True, Scope = None, BN = True, sn = True, is_training = False, strides=[1, 2, 2, 1]):
	with tf.device(device2):
		# padding image with reflection mode
		x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = 'REFLECT')
		# conv and add bias
		if sn:
			out = tf.nn.conv2d(input = x_padded, filter = spectral_norm(kernel, scope_name = Scope), strides = strides,
			                   padding = 'VALID')
		else:
			out = tf.nn.conv2d(input = x_padded, filter = kernel, strides = strides, padding = 'VALID')
		out = tf.nn.bias_add(out, bias)
		if BN:
			with tf.variable_scope(Scope):
				out = tf.layers.batch_normalization(out, training = is_training)
		if use_relu:
			# out = tf.nn.relu(out)
			out = tf.maximum(out, 0.2 * out)
	return out


def sa_deconv2d(x, kernel, strides):
	with tf.device(device2):
		out = tf.nn.conv2d_transpose(x, filter = kernel, output_shape = [int(x.shape[0]), int(x.shape[1]) * int(strides[2]), int(x.shape[2])*int(strides[2]), int(kernel.shape[2])], strides = strides, padding = 'SAME')
	return out

# def self_attention(inputs, channel_factor = 8, scope_name = None, name = 'self_attention'):
# 	num_filters = inputs.shape[-1].value // channel_factor
# 	with tf.device(device2):
# 		with tf.variable_scope(scope_name):
# 			with tf.variable_scope(name):
# 				flat_inputs = tf.reshape(inputs, shape = [int(inputs.shape[0]), int(inputs.shape[1])*int(inputs.shape[2]), int(inputs.shape[-1])])
# 				f = tf.layers.conv1d(flat_inputs, kernel_size = 1, filters = num_filters)
# 				g = tf.layers.conv1d(flat_inputs, kernel_size = 1, filters = num_filters)
# 				h = tf.layers.conv1d(flat_inputs, kernel_size = 1, filters = inputs.shape[-1])
# 				beta = tf.nn.softmax(tf.matmul(f, g, transpose_b = True))
# 				o = tf.matmul(beta, h)
# 				gamma = tf.get_variable('gamma', [], initializer = tf.zeros_initializer)
# 				y = gamma * o + flat_inputs
# 				y = tf.reshape(y, inputs.shape)
# 	return y


def attention(x, ch, sn = False, scope_name=None, name = 'self_attention', reuse = False):
	with tf.device(device2):
		with tf.variable_scope(scope_name):
			with tf.variable_scope(name, reuse = reuse):
				f = conv(x, ch // 4, kernel_size = 1, stride = 1, sn = sn, scope = 'f_conv')  # [bs, h, w, c']
				g = conv(x, ch // 4, kernel_size = 1, stride = 1, sn = sn, scope = 'g_conv')  # [bs, h, w, c']
				h = conv(x, ch, kernel_size = 1, stride = 1, sn = sn, scope = 'h_conv')  # [bs, h, w, c]

				# N = h * w
				s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b = True)  # # [bs, N, N]

				beta = tf.nn.softmax(s)  # attention map

				o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
				gamma = tf.Variable(tf.truncated_normal([1], stddev=WEIGHT_INIT_STDDEV), name='gamma')

				o = tf.reshape(o, shape = x.shape)  # [bs, h, w, C]
				x = gamma * o + x

	return x, beta




def conv(x, channels, kernel_size = 3, stride = 2, pad = 0, pad_type = 'zero', use_bias = True, sn = True, scope = 'conv_0'):
	weight_init = tf.random_normal_initializer(mean = 0.0, stddev = 0.02)
	weight_regularizer = None
	with tf.variable_scope(scope):
		if pad_type == 'zero':
			x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
		if pad_type == 'reflect':
			x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode = 'REFLECT')
		if sn:
			# with tf.device("/cpu:0"):
			kernel = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, x.shape[-1].value, channels], stddev = WEIGHT_INIT_STDDEV),
						                     name = 'kernel')
			# print('Regularization kernel shape:', kernel.shape)
			# kernel_loss=tf.reduce_sum(tf.square(kernel))/(kernel.shape[0].value*kernel.shape[1].value*kernel.shape[2].value*kernel.shape[3].value)
			# print('kernel_loss shape:', kernel_loss.shape)
			# tf.add_to_collection('Regularization_Losses', kernel_loss)

			x = tf.nn.conv2d(input = x, filter = spectral_norm(kernel, scope_name = scope), strides = [1, stride, stride, 1], padding = 'VALID')
			if use_bias:
				#with tf.device("/cpu:0"):
				bias = tf.Variable(tf.truncated_normal([channels], stddev = WEIGHT_INIT_STDDEV), name = 'bias')
			x = tf.nn.bias_add(x, bias)
		# else:
		# 	x = tf.layers.conv2d(inputs = x, filters = channels, kernel_size = kernel_size, kernel_initializer = weight_init,
		# 	                     kernel_regularizer = weight_regularizer, strides = stride, use_bias = use_bias)

		return x


def hw_flatten(x):
	return tf.reshape(x, shape = [x.shape[0].value, -1, x.shape[-1].value])


def spectral_norm(w, scope_name = None):
	w_shape = w.shape.as_list()
	w = tf.reshape(w, [-1, w_shape[-1]])

	# with tf.device("/cpu:0"):
	with tf.variable_scope(scope_name):
		u = tf.get_variable("u", [1, w_shape[-1]], initializer = tf.truncated_normal_initializer(), trainable = False)
	u_hat = u
	v_hat = None
	v_ = tf.matmul(u_hat, tf.transpose(w))
	v_hat = l2_norm(v_)
	u_ = tf.matmul(v_hat, w)
	u_hat = l2_norm(u_)

	sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
	w_norm = w / sigma

	with tf.control_dependencies([u.assign(u_hat)]):
		w_norm = tf.reshape(w_norm, w_shape)

	return w_norm


def l2_norm(v, eps = 1e-12):
	return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def up_sample(x, scale_factor = 2):
	_, h, w, _ = x.get_shape().as_list()
	new_size = [h * scale_factor, w * scale_factor]
	return tf.image.resize_nearest_neighbor(x, size = new_size)
