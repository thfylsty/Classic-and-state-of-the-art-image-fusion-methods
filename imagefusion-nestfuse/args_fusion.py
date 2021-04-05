
class args():
	# training args
	epochs = 2  #"number of training epochs, default is 2"
	batch_size = 1  #"batch size for training, default is 4"
	# dataset = "/data/Disk_B/MSCOCO2014/train2014/"  # the dataset path in your computer
	dataset = r'F:\database\MS-COCO2014\train2014'
	HEIGHT = 256
	WIDTH = 256

	save_model_dir_autoencoder = "models/nestfuse_autoencoder"
	save_loss_dir = './models/loss_autoencoder/'

	cuda = 1
	ssim_weight = [1,10,100,1000,10000]
	ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4']

	lr = 1e-4  #"learning rate, default is 0.001"
	lr_light = 1e-4  # "learning rate, default is 0.001"
	log_interval = 10  #"number of images after which the training loss is logged, default is 500"
	resume = None

	# for test, model_default is the model used in paper
	model_default = './models/nestfuse_1e2.model'
	model_deepsuper = './models/nestfuse_1e2_deep_super.model'


