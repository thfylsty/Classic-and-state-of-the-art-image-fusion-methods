from __future__ import print_function

import scipy.io as scio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from datetime import datetime
from scipy.misc import imsave
import scipy.ndimage
from skimage import img_as_ubyte
import os

from model import Model
from deepIQA_evaluate import IQA

EPSILON = 1e-5

eps = 1e-8

logging_period = 50
patch_size = 64
LEARNING_RATE = 0.0001


def train_task(model, sess, trainset, validset=[], save_path=None, lam = 0, IQA_model=None, task_ind=1, merged=None, writer=None, saver=None, w_en=1, c=1, EPOCHES=2):
	start_time = datetime.now()
	num_imgs = trainset.shape[0]
	mod = num_imgs % model.batchsize
	n_batches = int(num_imgs // model.batchsize)
	print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))
	if mod > 0:
		trainset = trainset[:-mod]

	model.restore(sess)

	if task_ind == 1:
		model.G_solver = tf.train.RMSPropOptimizer(learning_rate = LEARNING_RATE, decay = 0.6,
		                                           momentum = 0.15).minimize(model.content_loss,
		                                                                     var_list = model.theta_G)

	else:
		model.update_ewc_loss(lam=lam)
		model.G_solver = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, decay=0.6, momentum=0.15).minimize(model.ewc_loss,
		                                                                  var_list = model.theta_G)

	theta = tf.trainable_variables()
	model.clip = [p.assign(tf.clip_by_value(p, -30, 30)) for p in theta]

	initialize_uninitialized(sess)

	# ** Start Training **
	step = 0
	for epoch in range(EPOCHES):
		np.random.shuffle(trainset)
		# for batch in range(5):
		for batch in range(n_batches):
			model.step += 1
			step += 1
			# current_iter = step
			s1_index = np.random.choice([0, 1], 1)
			source1_batch = trainset[batch * model.batchsize:(batch * model.batchsize + model.batchsize), :, :, s1_index[0]]
			source2_batch = trainset[batch * model.batchsize:(batch * model.batchsize + model.batchsize), :, :, 1 - s1_index[0]]
			source1_batch = np.expand_dims(source1_batch, -1)
			source2_batch = np.expand_dims(source2_batch, -1)
			w1, w2 = W(inputs1 = source1_batch, inputs2 = source2_batch, trained_model_path = IQA_model, w_en=w_en,c=c)
			FEED_DICT= {model.SOURCE1: source1_batch, model.SOURCE2: source2_batch, model.W1: w1, model.W2: w2}

			it_g = 0
			sess.run([model.G_solver, model.clip], feed_dict = FEED_DICT)
			it_g += 1

			generated_img=sess.run(model.generated_img, feed_dict=FEED_DICT)
			with tf.device('/gpu:0'):
				iqa_f=IQA(inputs = generated_img, trained_model_path = IQA_model)
				en_f=EN(generated_img)
				score_f=np.mean(iqa_f + w_en * en_f)

			result = sess.run(merged, feed_dict = {model.SOURCE1: source1_batch, model.SOURCE2: source2_batch, model.W1: w1, model.W2: w2, model.score_f:score_f})
			writer[task_ind - 1].add_summary(result, model.step)
			writer[task_ind - 1].flush()



			### validation
			if len(validset):
				for i in range(len(validset)):
					sub_validset=validset[i]
					batch_ind = np.random.randint(int(sub_validset.shape[0]//model.batchsize))
					s_index = np.random.choice([0, 1], 1)
					valid_source1_batch = sub_validset[batch_ind * model.batchsize:(batch_ind * model.batchsize + model.batchsize), :, :, s_index[0]]
					valid_source2_batch = sub_validset[batch_ind * model.batchsize:(batch_ind * model.batchsize +model.batchsize), :, :, 1 - s_index[0]]
					valid_source1_batch = np.expand_dims(valid_source1_batch, -1)
					valid_source2_batch = np.expand_dims(valid_source2_batch, -1)
					valid_w1, valid_w2 = W(inputs1 = valid_source1_batch, inputs2 = valid_source2_batch, trained_model_path = IQA_model,w_en=w_en,c=c)

					valid_FEED_DICT = {model.SOURCE1: valid_source1_batch, model.SOURCE2: valid_source2_batch,
					                   model.W1: valid_w1, model.W2: valid_w2}

					valid_generated_img = sess.run(model.generated_img, feed_dict = valid_FEED_DICT)

					with tf.device('/gpu:0'):
						valid_iqa_f = IQA(inputs = valid_generated_img, trained_model_path = IQA_model)
						valid_en_f = EN(valid_generated_img)
						valid_score_f = np.mean(valid_iqa_f + w_en * valid_en_f)

					valid_result = sess.run(merged, feed_dict = {model.SOURCE1: valid_source1_batch, model.SOURCE2: valid_source2_batch,
					                                       model.W1: valid_w1, model.W2: valid_w2, model.score_f: valid_score_f})
					writer[i].add_summary(valid_result, model.step)
					writer[i].flush()

			is_last_step = (epoch == EPOCHES - 1) and (batch == n_batches - 1)
			if is_last_step or step % logging_period == 0:
				elapsed_time = datetime.now() - start_time
				content_loss, sloss,ploss, gradloss = sess.run([model.content_loss,model.ssim_loss, model.perloss, model.grad_loss], feed_dict = FEED_DICT)
				print('epoch:%d/%d, step:%d/%d, model step:%d, elapsed_time:%s' % (
					epoch + 1, EPOCHES, step%n_batches,n_batches, model.step, elapsed_time))
				print('content loss:%s \nssim loss:%s \nperceptual loss:%s \ngradient loss:%s\n' % (content_loss, sloss, ploss, gradloss))
				if hasattr(model, "ewc_loss"):
					add_loss=sess.run(model.Add_loss, feed_dict=FEED_DICT)
					print("Add_loss:%s\n" % add_loss)


			if is_last_step or step % 100 == 0:
				if not os.path.exists(save_path):
					os.makedirs(save_path)
				saver.save(sess, save_path + str(step) + '/' + str(step) + '.ckpt')
	# writer.close()
	# saver.save(sess, save_path + str(epoch) + '/' + str(epoch) + '.ckpt')


def initialize_uninitialized(sess):
	global_vars = tf.global_variables()
	is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
	not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
	# print('not_initialized_vars:')
	# for i in not_initialized_vars:
	# 	print(str(i.name))
	if len(not_initialized_vars):
		sess.run(tf.variables_initializer(not_initialized_vars))


def W(inputs1,inputs2, trained_model_path, w_en, c):
	# with tf.device('/gpu:1'):
	iqa1 = IQA(inputs = inputs1, trained_model_path = trained_model_path)
	iqa2 = IQA(inputs = inputs2, trained_model_path = trained_model_path)

	with tf.device('/gpu:0'):
		en1 = EN(inputs1)
		en2 = EN(inputs2)
		score1 = iqa1 + w_en * en1
		score2 = iqa2 + w_en * en2
		w1 = np.exp(score1 / c) / (np.exp(score1 / c) + np.exp(score2 / c))
		w2 = np.exp(score2 / c) / (np.exp(score1 / c) + np.exp(score2 / c))

	# print('IQA:   1: %f, 2: %f' % (iqa1[0], iqa2[0]))
	# print('EN:    1: %f, 2: %f' % (en1[0], en2[0]))
	# print('total: 1: %f, 2: %f' % (score1[0], score2[0]))
	# print('w1: %s, w2: %s\n' % (w1[0], w2[0]))
	# print('IQA:   1: %f, 2: %f' % (iqa1[1], iqa2[1]))
	# print('EN:    1: %f, 2: %f' % (en1[1], en2[1]))
	# print('total: 1: %f, 2: %f' % (score1[1], score2[1]))
	# print('w1: %s, w2: %s\n' % (w1[1], w2[1]))
	# fig = plt.figure()
	# fig1 = fig.add_subplot(221)
	# fig2 = fig.add_subplot(222)
	# fig3 = fig.add_subplot(223)
	# fig4 = fig.add_subplot(224)
	# fig1.imshow(inputs1[0, :, :, 0], cmap = 'gray')
	# fig2.imshow(inputs2[0, :, :, 0], cmap = 'gray')
	# fig3.imshow(inputs1[1,:,:,0],cmap='gray')
	# fig4.imshow(inputs2[1,:,:,0],cmap='gray')
	# plt.show()
	return (w1,w2)



def EN(inputs):
	len = inputs.shape[0]
	entropies = np.zeros(shape = (len, 1))
	grey_level = 256
	counter = np.zeros(shape = (grey_level, 1))

	for i in range(len):
		input_uint8 = (inputs[i, :, :, 0] * 255).astype(np.uint8)
		input_uint8 = input_uint8 + 1
		for m in range(patch_size):
			for n in range(patch_size):
				indexx = input_uint8[m, n]
				counter[indexx] = counter[indexx] + 1
		total = np.sum(counter)
		p = counter / total
		for k in range(grey_level):
			if p[k] != 0:
				entropies[i] = entropies[i] - p[k] * np.log2(p[k])
	return entropies
