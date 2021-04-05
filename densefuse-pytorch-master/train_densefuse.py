# Training DenseFuse network
# auto-encoder

import os
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
from net import DenseFuse_net
from args_fusion import args
import pytorch_msssim


def main():
	# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
	original_imgs_path = utils.list_images(args.dataset)
	train_num = 40000
	original_imgs_path = original_imgs_path[:train_num]
	random.shuffle(original_imgs_path)
	# for i in range(5):
	i = 2
	train(i, original_imgs_path)


def train(i, original_imgs_path):

	batch_size = args.batch_size

	# load network model, RGB
	in_c = 1 # 1 - gray; 3 - RGB
	if in_c == 1:
		img_model = 'L'
	else:
		img_model = 'RGB'
	input_nc = in_c
	output_nc = in_c
	densefuse_model = DenseFuse_net(input_nc, output_nc)

	if args.resume is not None:
		print('Resuming, initializing using weight from {}.'.format(args.resume))
		densefuse_model.load_state_dict(torch.load(args.resume))
	print(densefuse_model)
	optimizer = Adam(densefuse_model.parameters(), args.lr)
	mse_loss = torch.nn.MSELoss()
	ssim_loss = pytorch_msssim.msssim

	if args.cuda:
		densefuse_model.cuda()

	tbar = trange(args.epochs)
	print('Start training.....')

	# creating save path
	temp_path_model = os.path.join(args.save_model_dir, args.ssim_path[i])
	if os.path.exists(temp_path_model) is False:
		os.mkdir(temp_path_model)

	temp_path_loss = os.path.join(args.save_loss_dir, args.ssim_path[i])
	if os.path.exists(temp_path_loss) is False:
		os.mkdir(temp_path_loss)

	Loss_pixel = []
	Loss_ssim = []
	Loss_all = []
	all_ssim_loss = 0.
	all_pixel_loss = 0.
	for e in tbar:
		print('Epoch %d.....' % e)
		# load training database
		image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size)
		densefuse_model.train()
		count = 0
		for batch in range(batches):
			image_paths = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
			img = utils.get_train_images_auto(image_paths, height=args.HEIGHT, width=args.WIDTH, mode=img_model)

			count += 1
			optimizer.zero_grad()
			img = Variable(img, requires_grad=False)

			if args.cuda:
				img = img.cuda()
			# get fusion image
			# encoder
			en = densefuse_model.encoder(img)
			# decoder
			outputs = densefuse_model.decoder(en)
			# resolution loss
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
				mesg = "{}\tEpoch {}:\t[{}/{}]\t pixel loss: {:.6f}\t ssim loss: {:.6f}\t total: {:.6f}".format(
					time.ctime(), e + 1, count, batches,
								  all_pixel_loss / args.log_interval,
								  all_ssim_loss / args.log_interval,
								  (args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval
				)
				tbar.set_description(mesg)
				Loss_pixel.append(all_pixel_loss / args.log_interval)
				Loss_ssim.append(all_ssim_loss / args.log_interval)
				Loss_all.append((args.ssim_weight[i] * all_ssim_loss + all_pixel_loss) / args.log_interval)

				all_ssim_loss = 0.
				all_pixel_loss = 0.

			if (batch + 1) % (200 * args.log_interval) == 0:
				# save model
				densefuse_model.eval()
				densefuse_model.cpu()
				save_model_filename = args.ssim_path[i] + '/' + "Epoch_" + str(e) + "_iters_" + str(count) + "_" + \
									  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[
										  i] + ".model"
				save_model_path = os.path.join(args.save_model_dir, save_model_filename)
				torch.save(densefuse_model.state_dict(), save_model_path)
				# save loss data
				# pixel loss
				loss_data_pixel = np.array(Loss_pixel)
				loss_filename_path = args.ssim_path[i] + '/' + "loss_pixel_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
				scio.savemat(save_loss_path, {'loss_pixel': loss_data_pixel})
				# SSIM loss
				loss_data_ssim = np.array(Loss_ssim)
				loss_filename_path = args.ssim_path[i] + '/' + "loss_ssim_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
				scio.savemat(save_loss_path, {'loss_ssim': loss_data_ssim})
				# all loss
				loss_data_total = np.array(Loss_all)
				loss_filename_path = args.ssim_path[i] + '/' + "loss_total_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
				scio.savemat(save_loss_path, {'loss_total': loss_data_total})

				densefuse_model.train()
				densefuse_model.cuda()
				tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

	# pixel loss
	loss_data_pixel = np.array(Loss_pixel)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_pixel_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':','_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_pixel': loss_data_pixel})
	# SSIM loss
	loss_data_ssim = np.array(Loss_ssim)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_ssim_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_ssim': loss_data_ssim})
	# all loss
	loss_data_total = np.array(Loss_all)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_total_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_total': loss_data_total})
	# save model
	densefuse_model.eval()
	densefuse_model.cpu()
	save_model_filename = args.ssim_path[i] + '/' "Final_epoch_" + str(args.epochs) + "_" + \
						  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".model"
	save_model_path = os.path.join(args.save_model_dir, save_model_filename)
	torch.save(densefuse_model.state_dict(), save_model_path)

	print("\nDone, trained model saved at", save_model_path)


if __name__ == "__main__":
	main()
