#!/usr/bin/python2
import numpy as np
from numpy.lib.stride_tricks import as_strided

import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers

# import argparse
import six
import imageio
import numbers

# from IQA.nr_model import Model
# from IQA.deepIQA.fr_model import FRModel

# parser = argparse.ArgumentParser(description = 'evaluate.py')
# parser.add_argument('INPUT', help = 'path to input image')
# parser.add_argument('REF', default = "", nargs = "?", help = 'path to reference image, if omitted NR IQA is assumed')
# parser.add_argument('--model', '-m', default = '',
#                     help = 'path to the trained model')
# parser.add_argument('--top', choices = ('patchwise', 'weighted'),
#                     default = 'weighted', help = 'top layer and loss definition')
# parser.add_argument('--gpu', '-g', default = 0, type = int,
#                     help = 'GPU ID')
# args = parser.parse_args()

chainer.global_config.train = False
chainer.global_config.cudnn_deterministic = True


def extract_patches(arr, patch_shape = (32, 32, 3), extraction_step = 32):
	arr_ndim = arr.ndim

	if isinstance(patch_shape, numbers.Number):
		patch_shape = tuple([patch_shape] * arr_ndim)
	if isinstance(extraction_step, numbers.Number):
		extraction_step = tuple([extraction_step] * arr_ndim)

	patch_strides = arr.strides

	slices = tuple(slice(None, None, st) for st in extraction_step)
	indexing_strides = arr[slices].strides

	patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
	                       np.array(extraction_step)) + 1

	shape = tuple(list(patch_indices_shape) + list(patch_shape))
	strides = tuple(list(indexing_strides) + list(patch_strides))

	patches = as_strided(arr, shape = shape, strides = strides)
	return patches


def IQA(inputs, trained_model_path):
	# input: input image
	# top: choices={'patchwise','weighted'}
	# model: path to the trained model
	model = Model(top = 'weighted')
	cuda.cudnn_enabled = True
	cuda.check_cuda_available()
	xp = cuda.cupy
	serializers.load_hdf5(trained_model_path, model)
	model.to_gpu()

	len=inputs.shape[0]
	scores=np.zeros(shape=(len,1))
	for ii in range(len):
		input = np.concatenate((inputs[ii,:,:,:], inputs[ii,:,:,:], inputs[ii,:,:,:]), axis = -1)
		patches = extract_patches(input)
		X = np.transpose(patches.reshape((-1, 32, 32, 3)), (0, 3, 1, 2))

		# ref_img = np.concatenate((ref_img, ref_img, ref_img), axis = -1)
		# patches = extract_patches(ref_img)
		# X_ref = np.transpose(patches.reshape((-1, 32, 32, 3)), (0, 3, 1, 2))
		y = []
		weights = []
		batchsize = min(2000, X.shape[0])
		t = xp.zeros((1, 1), np.float32)
		for i in six.moves.range(0, X.shape[0], batchsize):
			X_batch = X[i:i + batchsize]
			X_batch = xp.array(X_batch.astype(np.float32))

			# NR:
			model.forward(X_batch, t, False, X_batch.shape[0])
			# FR:
			# X_ref_batch = X_ref[i:i + batchsize]
			# X_ref_batch = xp.array(X_ref_batch.astype(np.float32))
			# model.forward(X_batch, X_ref_batch, t, False, n_patches_per_image = X_batch.shape[0])

			y.append(xp.asnumpy(model.y[0].data).reshape((-1,)))
			weights.append(xp.asnumpy(model.a[0].data).reshape((-1,)))


			y = np.concatenate(y)
			weights = np.concatenate(weights)

		# print("%f" % (np.sum(y * weights) / np.sum(weights)))
		scores[ii] = np.sum(y * weights) / np.sum(weights)

	return scores
