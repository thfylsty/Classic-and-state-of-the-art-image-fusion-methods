import os
import cv2
import time
import torch
from model import myIFCNN

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from torchvision import transforms
from torch.autograd import Variable

from PIL import Image
import numpy as np

from utils.myTransforms import denorm, norms, detransformcv2
from utils.tools import list_images_with_name

# ## 2. Load the well-trained image fusion model (IFCNN-MAX)

# In[2]:


# we use fuse_scheme to choose the corresponding model,
# choose 0 (IFCNN-MAX) for fusing multi-focus, infrare-visual and multi-modal medical images, 2 (IFCNN-MEAN) for fusing multi-exposure images
fuse_scheme = 0
if fuse_scheme == 0:
	model_name = 'IFCNN-MAX'
elif fuse_scheme == 1:
	model_name = 'IFCNN-SUM'
elif fuse_scheme == 2:
	model_name = 'IFCNN-MEAN'
else:
	model_name = 'IFCNN-MAX'

# load pretrained model
model = myIFCNN(fuse_scheme=fuse_scheme)
model.load_state_dict(torch.load('snapshots/' + model_name + '.pth'))
model.eval()
model = model.cuda()

# ## 3. Use IFCNN to respectively fuse CMF, IV and MD datasets
# Fusion images are saved in the 'results' folder under your current folder.

# In[3]:


from utils.myDatasets import ImagePair

is_save = True  # if you do not want to save images, then change its value to False

# infrared and visible
test_path = "datasets/IV_images//"

# infrare and visual image dataset. Number: 14
is_gray = True  # Color (False) or Gray (True)
mean = [0, 0, 0]  # normalization parameters
std = [1, 1, 1]

img_ir_pathes, names_ir = list_images_with_name('./datasets/IV_images//IR/')
vis_path_root = './datasets/IV_images//VIS/'

begin_time = time.time()
for i in range(len(img_ir_pathes)):
	index = i
	ir_path = img_ir_pathes[index]
	if names_ir[index].__contains__('IR'):
		vis_name = names_ir[index].replace('IR', 'VIS')
	else:
		vis_name = names_ir[index].replace('i.', 'v.')
	vis_path = vis_path_root + vis_name

	# load source images
	pair_loader = ImagePair(impath1=ir_path, impath2=vis_path,
	                        transform=transforms.Compose([
		                        transforms.ToTensor(),
		                        transforms.Normalize(mean=mean, std=std)
	                        ]))
	img1, img2 = pair_loader.get_pair()
	img1.unsqueeze_(0)
	img2.unsqueeze_(0)

	# perform image fusion
	with torch.no_grad():
		res = model(Variable(img1.cuda()), Variable(img2.cuda()))
		res = denorm(mean, std, res[0]).clamp(0, 1) * 255
		res_img = res.cpu().data.numpy().astype('uint8')
		img = res_img.transpose([1, 2, 0])

	# save fused images
	if is_save:
		filename = 'fused_ifcnn_'
		if is_gray:
			img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			img = Image.fromarray(img)
			img.save('results/new_dataset/' + filename + names_ir[index], format='PNG', compress_level=0)
		else:
			img = Image.fromarray(img)
			img.save('results/IR-VIS/' + filename + '.png', format='PNG', compress_level=0)

# when evluating time costs, remember to stop writing images by setting is_save = False
proc_time = time.time() - begin_time
print('Total processing time of {} dataset: {:.3}s'.format('infrared and visible', proc_time))