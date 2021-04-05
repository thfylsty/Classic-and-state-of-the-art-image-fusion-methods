import tensorflow as tf

WEIGHT_INIT_STDDEV = 0.1


def deconv_ir(input, strides, scope_name):
	weight_vars = []
	scope = ['deconv1']
	with tf.variable_scope('Generator'):
		with tf.variable_scope(scope_name):
			weight_vars.append(_create_variables(1, 1, 3, scope = scope[0]))
			# weight_vars.append(_create_variables(1, 1, 3, scope = scope[1]))
	deconv_num = len(weight_vars)
	out = input
	for i in range(deconv_num):
		input_shape = out.shape
		kernel = weight_vars[i]
		out = tf.nn.conv2d_transpose(out, filter = kernel, output_shape = [int(input_shape[0]), int(input_shape[1]) * 4,
		                                                                   int(input_shape[2]) * 4,
		                                                                   int(input_shape[3])],
		                             strides = strides, padding = 'SAME')
	#
	return out


def deconv_vis(input, strides, scope_name):
	weight_vars = []
	scope = ['deconv1']
	with tf.variable_scope('Generator'):
		with tf.variable_scope(scope_name):
			weight_vars.append(_create_variables(1, 1, 3, scope = scope[0]))
			# weight_vars.append(_create_variables(1, 1, 3, scope = scope[1]))
	deconv_num = len(weight_vars)
	out = input
	for i in range(deconv_num):
		input_shape = out.shape
		kernel = weight_vars[i]
		out = tf.nn.conv2d_transpose(out, filter = kernel, output_shape = [int(input_shape[0]), int(input_shape[1]),
		                                                                   int(input_shape[2]), int(input_shape[3])],
		                             strides = strides, padding = 'SAME')
	#
	return out


def _create_variables(input_filters, output_filters, kernel_size, scope):
	shape = [kernel_size, kernel_size, output_filters, input_filters]
	with tf.variable_scope(scope):
		kernel = tf.Variable(tf.truncated_normal(shape, stddev = WEIGHT_INIT_STDDEV), name = 'kernel')
	# bias = tf.Variable(tf.zeros([output_filters]), name = 'bias')
	return kernel  # , bias)
