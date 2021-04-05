from __future__ import print_function

import time

# from utils import list_images
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from train import train
import scipy.ndimage

BATCH_SIZE = 19
EPOCHES = 4
LOGGING = 10


MODEL_SAVE_PATH = './models_without_lgra/'
IS_TRAINING = True
# IS_TRAINING = False

f = h5py.File('/home/jxy/xh/project/EPF/EPF_Dataset3.h5', 'r')
# v_f = h5py.File('/home/jxy/xh/project/EPF/EPF_Validation.h5', 'r')
# for key in f.keys():
#   print(f[key].name)
sources = f['data'][:]
print('sources shape:', sources.shape)
sources = np.transpose(sources, (0, 3, 2, 1))

# v_sources = v_f['data'][:]
# print('validation sources shape:', v_sources.shape)
# v_sources = np.transpose(v_sources, (0, 3, 2, 1))

test_model_num = 600



## 下采�?上采�?
# for i in range(int(sources.shape[0])):
# 	ir_ds = scipy.ndimage.zoom(sources[i, :, :, 1], 0.25)
# 	ir_ds_us = scipy.ndimage.zoom(ir_ds, 4, order = 3)
# 	sources[i, :, :, 1] = ir_ds_us
#
# if not os.path.exists('Dataset3_ds_us.h5'):
# 	with h5py.File('Dataset3_ds_us.h5') as f2:
# 		f2['data'] = sources

def main():
	print(('\nBegin to train the network ...\n'))
	train(source_imgs = sources, save_path = MODEL_SAVE_PATH, EPOCHES_set = EPOCHES,
	      BATCH_SIZE = BATCH_SIZE)



#  output_path = './models/22.cGAN_deconv/' + str(epoch) + '/')


if __name__ == '__main__':
	main()
