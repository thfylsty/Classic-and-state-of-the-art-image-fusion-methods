from __future__ import print_function
import time
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import tensorflow as tf

from train_task import train_task
from model import Model

IQA_model = './IQA/models/nr_tid_weighted.model'

data1_path = 'vis_ir_dataset64.h5'
data2_path = 'oe_ue_Y_dataset64.h5'
data3_path = 'far_near_Y_dataset64.h5'

patch_size = 64
LAM = 80000 # 400
NUM = 40

# vis_ir: W_EN=12, C=13
# oe_ue: W_EN=16, C=7
# far_near: W_EN=11, C=1

W_EN = [12, 16, 11]
C = [13, 7, 1]
EPOCHES = [3, 2, 4]


def main():
	with tf.Graph().as_default(), tf.Session() as sess:
		model = Model(BATCH_SIZE = 16, INPUT_W = patch_size, INPUT_H = patch_size, is_training = True)
		saver = tf.train.Saver(max_to_keep = 80)

		tf.summary.scalar('content_Loss', model.content_loss)
		tf.summary.scalar('ssim_Loss', model.ssim_loss)
		tf.summary.scalar('perceptual_Loss', model.perloss)
		tf.summary.scalar('gradient_Loss', model.grad_loss)
		tf.summary.image('source1', model.SOURCE1, max_outputs = 3)
		tf.summary.image('source2', model.SOURCE2, max_outputs = 3)
		tf.summary.image('fused_result', model.generated_img, max_outputs = 3)
		merged = tf.summary.merge_all()

		'''task1'''
		print('Begin to train the network on task1...\n')
		with tf.device('/cpu:0'):
			source_data1 = h5py.File(data1_path, 'r')
			source_data1 = source_data1['data'][:]
			source_data1 = np.transpose(source_data1, (0, 3, 2, 1))
			print("source_data1 shape:", source_data1.shape)
		writer1 = tf.summary.FileWriter("logs/lam" + str(LAM) + "/plot_1", sess.graph)
		train_task(model = model, sess = sess, merged = merged, writer = [writer1], saver = saver,
		           IQA_model = IQA_model,
		           trainset = source_data1, save_path = './models/lam' + str(LAM) + '/task1/', lam = LAM, task_ind = 1,
		           w_en = W_EN[0], c = C[0], EPOCHES = EPOCHES[0])

		# del source_data1
		# valid_source_data1 = h5py.File(valid_data1_path, 'r')
		# valid_source_data1 = valid_source_data1['data'][:]
		# valid_source_data1 = np.transpose(valid_source_data1, (0, 3, 2, 1))
		# print("valid source_data1 shape:", valid_source_data1.shape)

		'''task2'''
		num_imgs = source_data1.shape[0]
		n_batches1 = int(num_imgs // model.batchsize)
		model.step = n_batches1 * EPOCHES[0]
		print('model step:', model.step)

		print('Begin to train the network on task2...\n')
		saver.restore(sess, './models/lam' + str(LAM) + '/task1/' + str(n_batches1 * EPOCHES[0]) + '/' + str(
			n_batches1 * EPOCHES[0]) + '.ckpt')
		model.compute_fisher(source_data1, sess, IQA_model = IQA_model, num_samples = NUM)
		with tf.device('/gpu:0'):
			source_data2 = h5py.File(data2_path, 'r')
			source_data2 = source_data2['data'][:]
			source_data2 = np.transpose(source_data2, (0, 3, 2, 1))
			print("source_data2 shape:", source_data2.shape)

		writer2 = tf.summary.FileWriter("logs/lam" + str(LAM) + "/plot_2", sess.graph)
		model.star()
		train_task(model = model, sess = sess, merged = merged, writer = [writer1, writer2],
		           validset = [source_data1], saver = saver, IQA_model = IQA_model,
		           trainset = source_data2, save_path = './models/lam' + str(LAM) + '/task2/', lam = LAM, task_ind = 2,
		           w_en = W_EN[1], c = C[1], EPOCHES = EPOCHES[1])

		## model.compute_fisher(np.append(source_data1, source_data2,axis=0), sess, IQA_model = IQA_model, num_samples = NUM)
		## model.compute_fisher(source_data2, sess, IQA_model = IQA_model, num_samples = NUM)

		'''task3'''
		num_imgs = source_data2.shape[0]
		n_batches2 = int(num_imgs // model.batchsize)
		model.step += n_batches2 * EPOCHES[1]
		print('model step:', model.step)
		print('Begin to train the network on task2...\n')
		saver.restore(sess, './models/lam' + str(LAM) + '/task2/' + str(n_batches2 * EPOCHES[1]) + '/' + str(
			n_batches2 * EPOCHES[1]) + '.ckpt')
		model.compute_fisher(np.append(source_data1, source_data2, axis = 0), sess, IQA_model = IQA_model,
		                     num_samples = NUM)
		with tf.device('/gpu:0'):
			source_data3 = h5py.File(data3_path, 'r')
			source_data3 = source_data3['data'][:]
			source_data3 = np.transpose(source_data3, (0, 3, 2, 1))
			print("source_data3 shape:", source_data3.shape)

		writer3 = tf.summary.FileWriter("logs/lam" + str(LAM) + "/plot_3", sess.graph)
		model.star()
		train_task(model = model, sess = sess, merged = merged, writer = [writer1, writer2, writer3],
		           validset = [source_data1, source_data2], saver = saver, IQA_model = IQA_model,
		           trainset = source_data3, save_path = './models/lam' + str(LAM) + '/task3/', lam = LAM, task_ind = 3,
		           w_en = W_EN[2], c = C[2], EPOCHES = EPOCHES[2])


if __name__ == '__main__':
	main()
