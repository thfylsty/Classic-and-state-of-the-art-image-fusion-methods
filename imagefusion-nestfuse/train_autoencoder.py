# -*- coding:utf-8 -*-
#@Project: NestFuse for image fusion
#@Author: Li Hui, Jiangnan University
#@Email: hui_li_jnu@163.com
#@File : train_autoencoder.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time
import numpy as np
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
from net import NestFuse_autoencoder
from args_fusion import args
import pytorch_msssim


def main():
	original_imgs_path = utils.list_images(args.dataset)
	train_num = 80000
	original_imgs_path = original_imgs_path[:train_num]
	random.shuffle(original_imgs_path)
	for i in range(2,3):
		# i = 3
		train(i, original_imgs_path)


def train(i, original_imgs_path):

	batch_size = args.batch_size

	# load network model
	# nest_model = FusionNet_gra()
	input_nc = 1
	output_nc = 1
	deepsupervision = False  # true for deeply supervision
	nb_filter = [64, 112, 160, 208, 256]

	nest_model = NestFuse_autoencoder(nb_filter, input_nc, output_nc, deepsupervision)

	if args.resume is not None:
		print('Resuming, initializing using weight from {}.'.format(args.resume))
		nest_model.load_state_dict(torch.load(args.resume))
	print(nest_model)
	optimizer = Adam(nest_model.parameters(), args.lr)
	mse_loss = torch.nn.MSELoss()
	ssim_loss = pytorch_msssim.msssim

	if args.cuda:
		nest_model.cuda()

	tbar = trange(args.epochs)
	print('Start training.....')

	Loss_pixel = []
	Loss_ssim = []
	Loss_all = []
	count_loss = 0
	all_ssim_loss = 0.
	all_pixel_loss = 0.
	for e in tbar:
		print('Epoch %d.....' % e)
		# load training database
		image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size)
		nest_model.train()
		count = 0
		for batch in range(batches):
			image_paths = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
			img = utils.get_train_images_auto(image_paths, height=args.HEIGHT, width=args.WIDTH, flag=False)
			count += 1
			optimizer.zero_grad()
			img = Variable(img, requires_grad=False)
			if args.cuda:
				img = img.cuda()
			# get fusion image
			# encoder
			en = nest_model.encoder(img)
			# decoder
			outputs = nest_model.decoder_train(en)
			# resolution loss: between fusion image and visible image
			x = Variable(img.data.clone(), requires_grad=False)

			ssim_loss_value = 0.
			pixel_loss_value = 0.
			for output in outputs:
				pixel_loss_temp = mse_loss(output, x)
				ssim_loss_temp = ssim_loss(output, x, normalize=True)
				ssim_loss_value += (1-ssim_loss_temp)
				pixel_loss_value += pixel_loss_temp
			ssim_loss_value /= len(outputs)
			pixel_loss_value /= len(outputs)

			# total loss
			total_loss = pixel_loss_value + args.ssim_weight[i] * ssim_loss_value
			total_loss.backward()
			optimizer.step()

			all_ssim_loss += ssim_loss_value.item()
			all_pixel_loss += pixel_loss_value.item()
			if (batch + 1) % args.log_interval == 0:
				mesg = "{}\t SSIM weight {}\tEpoch {}:\t[{}/{}]\t pixel loss: {:.6f}\t ssim loss: {:.6f}\t total: {:.6f}".format(
					time.ctime(), i, e + 1, count, batches,
								  all_pixel_loss / args.log_interval,
								  (args.ssim_weight[i] * all_ssim_loss) / args.log_interval,
								  (args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval
				)
				tbar.set_description(mesg)
				Loss_pixel.append(all_pixel_loss / args.log_interval)
				Loss_ssim.append(all_ssim_loss / args.log_interval)
				Loss_all.append((args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval)
				count_loss = count_loss + 1
				all_ssim_loss = 0.
				all_pixel_loss = 0.

			if (batch + 1) % (200 * args.log_interval) == 0:
				# save model
				nest_model.eval()
				nest_model.cpu()
				save_model_filename = args.ssim_path[i] + '/' + "Epoch_" + str(e) + "_iters_" + str(count) + "_" + \
									  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[
										  i] + ".model"
				save_model_path = os.path.join(args.save_model_dir_autoencoder, save_model_filename)
				torch.save(nest_model.state_dict(), save_model_path)
				# save loss data
				# pixel loss
				loss_data_pixel = Loss_pixel
				loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "loss_pixel_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				scio.savemat(loss_filename_path, {'loss_pixel': loss_data_pixel})
				# SSIM loss
				loss_data_ssim = Loss_ssim
				loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "loss_ssim_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				scio.savemat(loss_filename_path, {'loss_ssim': loss_data_ssim})
				# all loss
				loss_data = Loss_all
				loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "loss_all_epoch_" + str(e) + "_iters_" + \
									 str(count) + "-" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				scio.savemat(loss_filename_path, {'loss_all': loss_data})

				nest_model.train()
				nest_model.cuda()
				tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

	# pixel loss
	loss_data_pixel = Loss_pixel
	loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "Final_loss_pixel_epoch_" + str(
		args.epochs) + "_" + str(
		time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
	scio.savemat(loss_filename_path, {'final_loss_pixel': loss_data_pixel})
	loss_data_ssim = Loss_ssim
	loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "Final_loss_ssim_epoch_" + str(
		args.epochs) + "_" + str(
		time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
	scio.savemat(loss_filename_path, {'final_loss_ssim': loss_data_ssim})
	# SSIM loss
	loss_data = Loss_all
	loss_filename_path = args.save_loss_dir + args.ssim_path[i] + '/' + "Final_loss_all_epoch_" + str(
		args.epochs) + "_" + str(
		time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".mat"
	scio.savemat(loss_filename_path, {'final_loss_all': loss_data})
	# save model
	nest_model.eval()
	nest_model.cpu()
	save_model_filename = args.ssim_path[i] + '/' "Final_epoch_" + str(args.epochs) + "_" + \
						  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".model"
	save_model_path = os.path.join(args.save_model_dir_autoencoder, save_model_filename)
	torch.save(nest_model.state_dict(), save_model_path)

	print("\nDone, trained model saved at", save_model_path)


def check_paths(args):
	try:
		if not os.path.exists(args.vgg_model_dir):
			os.makedirs(args.vgg_model_dir)
		if not os.path.exists(args.save_model_dir):
			os.makedirs(args.save_model_dir)
	except OSError as e:
		print(e)
		sys.exit(1)


if __name__ == "__main__":
	main()
