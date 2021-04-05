# Train the DenseFuse Net

from __future__ import print_function

import scipy.io as scio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from scipy.misc import imsave
import scipy.ndimage

from Generator import Generator
from Discriminator import Discriminator
# import tensorlayer as tl
from LOSS import SSIM_LOSS, L1_LOSS, Fro_LOSS, _tf_fspecial_gauss
from datetime import datetime

patch_size = 144
# TRAINING_IMAGE_SHAPE = (patch_size, patch_size, 2)  # (height, width, color_channels)

LEARNING_RATE = 0.0003
EPSILON = 1e-5
DECAY_RATE = 0.7
eps = 1e-8

retrain = False
model_path ='./models/'


def train(source_imgs, save_path, EPOCHES_set, BATCH_SIZE):

	start_time = datetime.now()
	EPOCHS = EPOCHES_set
	print('Epoches: %d, Batch_size: %d' % (EPOCHS, BATCH_SIZE))

	num_imgs = source_imgs.shape[0]
	mod = num_imgs % BATCH_SIZE
	n_batches = int(num_imgs // BATCH_SIZE)
	print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))

	if mod > 0:
		print('Train set has been trimmed %d samples...\n' % mod)
		source_imgs = source_imgs[:-mod]

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	# create the graph
	with tf.Graph().as_default(), tf.Session(config=config) as sess:
		SOURCE_oe = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 3), name = 'OE_IMG')
		SOURCE_ue = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 3), name = 'UE_IMG')
		GT = tf.placeholder(tf.float32, shape = (BATCH_SIZE, patch_size, patch_size, 3), name = 'GT')
		print('source img shape:', SOURCE_oe.shape)

		# upsampling vis and ir images
		G = Generator('Generator')
		generated_img = G.transform(oe_img = SOURCE_oe, ue_img = SOURCE_ue, is_training=True)
		print('generate img shape:', generated_img.shape)

		D = Discriminator('Discriminator')
		D_real = D.discrim(GT, reuse = False)
		D_fake = D.discrim(generated_img, reuse = True)

		#######  LOSS FUNCTION
		# Loss for Generator
		G_loss_adv = -tf.reduce_mean(tf.log(D_fake + eps))

		r_loss = tf.reduce_mean(
			tf.reduce_sum(tf.square(GT[:, :, :, 0] - generated_img[:, :, :, 0]), [1, 2]) / (patch_size * patch_size))
		g_loss = tf.reduce_mean(
			tf.reduce_sum(tf.square(GT[:, :, :, 1] - generated_img[:, :, :, 1]), [1, 2]) / (patch_size * patch_size))
		b_loss = tf.reduce_mean(
			tf.reduce_sum(tf.square(GT[:, :, :, 2] - generated_img[:, :, :, 2]), [1, 2]) / (patch_size * patch_size))
		mse_loss = (r_loss + g_loss + b_loss) / 3

		grad_GT = grad(tf.image.rgb_to_grayscale(GT))
		grad_fuse = grad(tf.image.rgb_to_grayscale(generated_img))
		gradient_loss = L1_LOSS(grad_GT-grad_fuse)

		G_loss_content = mse_loss + 0.6 * gradient_loss

		G_loss = G_loss_adv + 500 * G_loss_content


		# Loss for Discriminator
		D_loss_real = -tf.reduce_mean(tf.log(D_real + eps))
		D_loss_fake = -tf.reduce_mean(tf.log(1. - D_fake + eps))
		D_loss = D_loss_fake + D_loss_real

		theta_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Generator')
		theta_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Discriminator')

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		current_iter = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(learning_rate = LEARNING_RATE, global_step = current_iter,
		                                           decay_steps = int(n_batches), decay_rate = DECAY_RATE,
		                                           staircase = False)

		with tf.control_dependencies(update_ops):
			# G_solver_adv = tf.train.RMSPropOptimizer(learning_rate).minimize(G_loss_adv, global_step = current_iter, var_list = theta_G)
			# G_solver = tf.train.RMSPropOptimizer(learning_rate).minimize(G_loss, global_step = current_iter, var_list = theta_G)
			# D_solver = tf.train.GradientDescentOptimizer(learning_rate).minimize(D_loss, global_step = current_iter, var_list = theta_D)

			G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.0, beta2=0.9).minimize(G_loss, global_step = current_iter, var_list = theta_G)
			D_solver = tf.train.GradientDescentOptimizer(learning_rate=learning_rate*2).minimize(D_loss, global_step = current_iter, var_list = theta_D)

		clip_G = [p.assign(tf.clip_by_value(p, -8, 8)) for p in theta_G]
		clip_D = [p.assign(tf.clip_by_value(p, -8, 8)) for p in theta_D]

		sess.run(tf.global_variables_initializer())
		g_list = tf.global_variables()
		saver = tf.train.Saver(var_list = g_list, max_to_keep = 200)

		tf.summary.scalar('G_Loss_content', G_loss_content)
		tf.summary.scalar('G_Loss_adv', G_loss_adv)
		tf.summary.scalar('D_Loss', D_loss)
		tf.summary.scalar('D_real', tf.reduce_mean(D_real))
		tf.summary.scalar('D_fake', tf.reduce_mean(D_fake))
		tf.summary.scalar('Learning_rate', learning_rate)

		tf.summary.image('oe', SOURCE_oe, max_outputs = 3)
		tf.summary.image('ue', SOURCE_ue, max_outputs = 3)
		tf.summary.image('fused_result', generated_img, max_outputs = 3)
		tf.summary.image('groundtruth', GT, max_outputs = 3)
		# tf.summary.image('attention_map', tf.expand_dims(attention_map, axis=-1), max_outputs = 3)

		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter("logs/", sess.graph)

		# ** Start Training **
		if retrain:
			model_save_path = model_path + str(model_num) + '/' + str(model_num) + '.ckpt'
			print("retrain: model:", model_save_path)
			saver.restore(sess, model_save_path)
			step = model_num
		else:
			step = 0


		for epoch in range(EPOCHS):
			np.random.shuffle(source_imgs)

			for batch in range(n_batches):
				step += 1
				current_iter = step
				oe_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 0:3]
				ue_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 3:6]
				gt_batch = source_imgs[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE), :, :, 6:9]

				FEED_DICT = {SOURCE_oe: oe_batch, SOURCE_ue: ue_batch, GT: gt_batch}

				it_d = 0
				it_g = 0
				# run the training step
				sess.run([D_solver, clip_D], feed_dict = FEED_DICT)
				it_d += 1
				sess.run([G_solver, clip_G], feed_dict = FEED_DICT)
				it_g += 1

				if batch % 3 == 0:
					d_fake, d_real = sess.run([tf.reduce_mean(D_fake), tf.reduce_mean(D_real)], feed_dict = FEED_DICT)
					while ((d_fake > 0.6) or (d_real < 0.4)) and (it_d < 4):
						sess.run([D_solver, clip_D], feed_dict = FEED_DICT)
						d_fake, d_real = sess.run([tf.reduce_mean(D_fake), tf.reduce_mean(D_real)], feed_dict = FEED_DICT)
						it_d += 1
					while ((d_real > 0.6) or (d_fake < 0.4)) and (it_g < 8):
						sess.run([G_solver, clip_G], feed_dict = FEED_DICT)
						d_fake, d_real = sess.run([tf.reduce_mean(D_fake), tf.reduce_mean(D_real)], feed_dict = FEED_DICT)
						it_g += 1

				# print('batch:%s, it_g:%s, it_d:%s' % ((step % n_batches), it_g, it_d))

				result = sess.run(merged, feed_dict = FEED_DICT)
				writer.add_summary(result, step)

				if batch % 20 == 0:
					elapsed_time = datetime.now() - start_time
					g_loss_content = sess.run(G_loss_content, feed_dict = FEED_DICT)
					g_loss_adv, d_loss = sess.run([G_loss_adv, D_loss], feed_dict = FEED_DICT)
					d_fake, d_real = sess.run([tf.reduce_mean(D_fake), tf.reduce_mean(D_real)], feed_dict = FEED_DICT)
					m_loss, gra_loss=sess.run([tf.reduce_mean(mse_loss), tf.reduce_mean(gradient_loss)], feed_dict=FEED_DICT)

					print("Epoch:%s, batch: %s/%s, step: %s" % (epoch + 1, (step % n_batches), n_batches, step))
					print('G_loss_content:%s' % (g_loss_content))
					print('mse_loss:%s, gradient_loss:%s' % (m_loss, gra_loss))
					print('G_loss_adv:%s, D_loss:%s' % (g_loss_adv, d_loss))
					print('D_fake:%s, D_real:%s' % (d_fake, d_real))
					print("elapsed_time:%s\n" % (elapsed_time))


				if (step % 100 == 0) or (step % n_batches == 0):
					print("save path:", save_path)
					saver.save(sess, save_path + str(step) + '/' + str(step) + '.ckpt')

	writer.close()
	saver.save(sess, save_path + str(epoch) + '/' + str(epoch) + '.ckpt')


def grad(img):
	kernel = tf.constant([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
	kernel = tf.expand_dims(kernel, axis = -1)
	kernel = tf.expand_dims(kernel, axis = -1)
	g = tf.nn.conv2d(img, kernel, strides = [1, 1, 1, 1], padding = 'SAME')
	return g


def rgb2ihs(rgbimg):
	r = rgbimg[:, :, 0]
	g = rgbimg[:, :, 1]
	b = rgbimg[:, :, 2]
	i = tf.expand_dims(1 / np.sqrt(3) * r + 1 / np.sqrt(3) * g + 1 / np.sqrt(3) * b, -1)
	h = tf.expand_dims(1 / np.sqrt(6) * r + 1 / np.sqrt(6) * g - 2 / np.sqrt(6) * b, -1)
	v = tf.expand_dims(1 / np.sqrt(2) * r - 1 / np.sqrt(2) * g, -1)
	ihsimg = tf.concat([i, h, v], -1)
	return ihsimg


def L1_LOSS(batchimg):
	L1_norm = tf.reduce_sum(tf.abs(batchimg), axis = [1, 2])/(patch_size * patch_size)
	E = tf.reduce_mean(L1_norm)
	return E
