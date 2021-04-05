import os
import cv2
import time
import torch
from model import myIFCNN

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

from torchvision import transforms
from torch.autograd import Variable

from PIL import Image
import numpy as np

from utils.myTransforms import denorm, norms, detransformcv2

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
model.load_state_dict(torch.load('snapshots/'+ model_name + '.pth'))
model.eval()
model = model.cuda()

from utils.myDatasets import ImagePair

IV_filenames = ['Camp', 'Camp1', 'Dune', 'Gun', 'Navi', 'Kayak', 'Octec', 'Road', 'Road2', 'Steamboat', 'T2', 'T3',
                'Trees4906', 'Trees4917', 'Window']
MF_filenames = ['clock', 'lab', 'pepsi', 'book', 'flower', 'desk', 'seascape', 'temple', 'leopard', 'wine', 'balloon',
                'calendar', 'corner', 'craft', 'leaf', 'newspaper', 'girl', 'grass', 'toy']

datasets = ['IV']  # Color MultiFocus, Infrared-Visual, MeDical image datasets
datasets_num = [20, 15, 8]  # number of image sets in each dataset
is_save = True  # if you do not want to save images, then change its value to False

for j in range(1):
    j = 1
    begin_time = time.time()
    for ind in range(50):
        if j == 0:
            # lytro-dataset: two images. Number: 20
            dataset = datasets[j]  # Color Multifocus Images
            is_gray = False  # Color (False) or Gray (True)
            mean = [0.485, 0.456, 0.406]  # normalization parameters
            std = [0.229, 0.224, 0.225]

            root = 'datasets/CMFDataset/'
            filename = 'lytro-{:02}'.format(ind + 1)
            path1 = os.path.join('{0}-A.jpg'.format(root + filename))
            path2 = os.path.join('{0}-B.jpg'.format(root + filename))
        elif j == 1:
            # infrare and visual image dataset. Number: 14
            # dataset = datasets[j]  # Infrared and Visual Images
            is_gray = True  # Color (False) or Gray (True)
            mean = [0, 0, 0]  # normalization parameters
            std = [1, 1, 1]

            root = 'datasets/IVDataset/'
            # filename = IV_filenames[ind]
            path1 = "../road/vi/%d.jpg" % (ind+1)
            path2 = "../road/ir/%d.jpg" % (ind+1)
        elif j == 2:
            # medical image dataset: CT (MR) and MR. Number: 8
            dataset = datasets[j]  # Medical Image
            is_gray = True  # Color (False) or Gray (True)
            mean = [0, 0, 0]  # normalization parameters
            std = [1, 1, 1]

            root = 'datasets/MDDataset/'
            filename = 'c{:02}'.format(ind + 1)
            path1 = os.path.join('{0}_1.tif'.format(root + filename))
            path2 = os.path.join('{0}_2.tif'.format(root + filename))

        # load source images
        pair_loader = ImagePair(impath1=path1, impath2=path2,
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
            # filename = model_name + '-' + dataset + '-' + filename
            if is_gray:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = Image.fromarray(img)
                img.save('results/' + str(ind+1) + '.bmp', format='PNG', compress_level=0)
            else:
                img = Image.fromarray(img)
                img.save('results/'+ str(ind+1) +  '.bmp', format='PNG', compress_level=0)

    # when evluating time costs, remember to stop writing images by setting is_save = False
    proc_time = time.time() - begin_time
    print('Total processing time of  {:.3}s'.format( proc_time))