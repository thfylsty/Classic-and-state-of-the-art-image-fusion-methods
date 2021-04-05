from __future__ import print_function

import time

# from utils import list_images
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from train import train
from generate import generate
import scipy.ndimage

BATCH_SIZE = 24
EPOCHES = 1
LOGGING = 40
MODEL_SAVE_PATH = './model/'
IS_TRAINING = False

# f = h5py.File('Medical_dataset.h5', 'r')
# # for key in f.keys():
# #   print(f[key].name)
# sources = f['data'][:]
# sources = np.transpose(sources, (0, 3, 2, 1))



def main():
	if IS_TRAINING:
		print(('\nBegin to train the network ...\n'))
		# train(sources, MODEL_SAVE_PATH, EPOCHES, BATCH_SIZE, logging_period = LOGGING)
	else:
		print('\nBegin to generate pictures ...\n')
		path = './test_imgs/'
		path1 = path + 'PET-I/'
		path2 = path + 'MR-T1/'
		savepath = './results/'
		# for root, dirs, files in os.walk(path):
		# 	test_num = len(files)

		Time=[]
		for i in range(20):
			index = i + 1
			pet_path = path1 + str(index) + '.png'
			mri_path = path2 + str(index) + '.png'
			begin = time.time()
			model_path = MODEL_SAVE_PATH + 'model.ckpt'
			generate(pet_path, mri_path, model_path, index, output_path = savepath)
			end = time.time()
			Time.append(end - begin)
			print("pic_num:%s" % index)
		print("Time: mean:%s, std: %s" % (np.mean(Time), np.std(Time)))


if __name__ == '__main__':
	main()
