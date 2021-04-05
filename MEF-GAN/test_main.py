from __future__ import print_function

import time

# from utils import list_images
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from train import train
# from generate import generate
from generate import generate
import scipy.ndimage
import scipy.io as scio


BATCH_SIZE = 18
EPOCHES = 4
LOGGING = 20

MODEL_SAVE_PATH = './models/'

def main():
	print('\nBegin to generate pictures ...\n')
	path = './test_imgs/'
	Format='.bmp'

	T=[]
	for i in range(50):
		index = i + 1
		ue_path = "../road/vi/" + str(i + 1) + ".jpg"
		oe_path = "../road/ir/" + str(i + 1) + ".jpg"
		# oe_path = 'vis/' + str(index) + Format
		# ue_path = 'ir/' + str(index) + Format

		t=generate(oe_path, ue_path, MODEL_SAVE_PATH, index-1, output_path = './results/', format=Format)

		T.append(t)
		print("%s time: %s" % (index, t))
	scio.savemat('time.mat', {'T': T})

if __name__ == '__main__':
	main()
