
# coding: utf-8

# # Demo for running IFCNN to fuse multiple types of images

# Project page of IFCNN is https://github.com/uzeful/IFCNN.
# 
# If you find this code is useful for your research, please consider to cite our paper.
# ```
# @article{zhang2019IFCNN,
#   title={IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network},
#   author={Zhang, Yu and Liu, Yu and Sun, Peng and Yan, Han and Zhao, Xiaolin and Zhang, Li},
#   journal={Information Fusion},
#   volume={54},
#   pages={99--118},
#   year={2020},
#   publisher={Elsevier}
# }
# ```
# 
# Detailed procedures to use IFCNN are introduced as follows.

# ## 1. Load required libraries

# In[1]:


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
model.load_state_dict(torch.load('snapshots/'+ model_name + '.pth'))
model.eval()
model = model.cuda()


# ## 3. Use IFCNN to respectively fuse CMF, IV and MD datasets
# Fusion images are saved in the 'results' folder under your current folder.

# In[3]:


from utils.myDatasets import ImagePair

IV_filenames = ['Camp', 'Camp1', 'Dune', 'Gun', 'Navi', 'Kayak', 'Octec', 'Road', 'Road2', 'Steamboat', 'T2', 'T3', 'Trees4906', 'Trees4917', 'Window']
MF_filenames = ['clock',  'lab', 'pepsi', 'book', 'flower', 'desk', 'seascape', 'temple', 'leopard', 'wine', 'balloon', 'calendar', 'corner', 'craft', 'leaf', 'newspaper', 'girl', 'grass', 'toy']

datasets = ['CMF', 'IV', 'MD'] # Color MultiFocus, Infrared-Visual, MeDical image datasets
datasets_num = [20, 15, 8]     # number of image sets in each dataset
is_save = True                 # if you do not want to save images, then change its value to False

for j in range(len(datasets)):
    begin_time = time.time()
    for ind in range(datasets_num[j]):
        if j == 0:
            # lytro-dataset: two images. Number: 20
            dataset = datasets[j]      # Color Multifocus Images
            is_gray = False            # Color (False) or Gray (True)
            mean=[0.485, 0.456, 0.406] # normalization parameters
            std=[0.229, 0.224, 0.225]
            
            root = 'datasets/CMFDataset/'
            filename = 'lytro-{:02}'.format(ind+1)
            path1 = os.path.join('{0}-A.jpg'.format(root + filename))
            path2 = os.path.join('{0}-B.jpg'.format(root + filename))
        elif j == 1:
            # infrare and visual image dataset. Number: 14
            dataset = datasets[j]  # Infrared and Visual Images
            is_gray = True         # Color (False) or Gray (True)
            mean=[0, 0, 0]         # normalization parameters
            std=[1, 1, 1]
            
            root = 'datasets/IVDataset/'
            filename = IV_filenames[ind]
            path1 = os.path.join(root, '{0}_Vis.png'.format(filename))
            path2 = os.path.join(root, '{0}_IR.png'.format(filename))
        elif j == 2:
            # medical image dataset: CT (MR) and MR. Number: 8
            dataset = datasets[j]  # Medical Image
            is_gray = True         # Color (False) or Gray (True)
            mean=[0, 0, 0]         # normalization parameters
            std=[1, 1, 1]
            
            root = 'datasets/MDDataset/'
            filename = 'c{:02}'.format(ind+1)
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
            img = res_img.transpose([1,2,0])

        # save fused images
        if is_save:
            filename = model_name + '-' + dataset + '-' + filename
            if is_gray:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = Image.fromarray(img)
                img.save('results/'+filename+'.png', format='PNG', compress_level=0)
            else:
                img = Image.fromarray(img)
                img.save('results/'+filename+'.png', format='PNG', compress_level=0)

    # when evluating time costs, remember to stop writing images by setting is_save = False
    proc_time = time.time() - begin_time
    print('Total processing time of {} dataset: {:.3}s'.format(datasets[j], proc_time))


# ## 4. Use IFCNN to fuse triple multi-focus images in CMF dataset
# Fusion images are saved in the 'results' folder under your current folder.

# In[4]:


from utils.myDatasets import ImageSequence

dataset = 'CMF3'           # Triple Color MultiFocus
is_save = True             # Whether to save the results
is_gray = False            # Color (False) or Gray (True)
is_folder=False            # one parameter in ImageSequence
mean=[0.485, 0.456, 0.406] # Color (False) or Gray (True)
std=[0.229, 0.224, 0.225]

begin_time = time.time()
for ind in range(4):
    # load the sequential source images
    root = 'datasets/CMFDataset/Triple Series/'
    filename = 'lytro-{:02}'.format(ind+1)
    paths = []
    paths.append(os.path.join('{0}-A.jpg'.format(root + filename)))
    paths.append(os.path.join('{0}-B.jpg'.format(root + filename)))
    paths.append(os.path.join('{0}-C.jpg'.format(root + filename)))
    filename = model_name + '-' + dataset + '-' + 'lytro-{:02}'.format(ind+1)

    seq_loader = ImageSequence(is_folder, 'RGB', transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=mean, std=std),
                                ]), *paths)
    imgs = seq_loader.get_imseq()
    
    # perform image fusion
    with torch.no_grad():
        vimgs = []
        for idx, img in enumerate(imgs):
            img.unsqueeze_(0)
            vimgs.append(Variable(img.cuda()))
        vres = model(*vimgs)
        res = denorm(mean, std, vres[0]).clamp(0, 1) * 255
        res_img = res.cpu().data.numpy().astype('uint8')
        img = Image.fromarray(res_img.transpose([1,2,0]))
    
    # save the fused image
    if is_save:
        if is_gray:
            img.convert('L').save('results/'+filename+'.png', format='PNG', compress_level=0)
        else:
            img.save('results/'+filename+'.png', format='PNG', compress_level=0)
            
# when evluating time costs, remember to stop writing images by setting is_save = False
proc_time = time.time() - begin_time
print('Total processing time of {} dataset: {:.3}s'.format(dataset ,proc_time))


# # 5. Load the well-trained image fusion model (IFCNN-MEAN)

# In[5]:


# we use fuse_scheme to choose the corresponding model, 
# choose 0 (IFCNN-MAX) for fusing multi-focus, infrare-visual and multi-modal medical images, 2 (IFCNN-MEAN) for fusing multi-exposure images
fuse_scheme = 2
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


# ## 6. Use IFCNN to fuse various number of multi-exposure images in ME Dataset
# Fusion images are saved in the 'results' folder under your current folder.

# In[6]:


from utils.myDatasets import ImageSequence

dataset = 'ME'
is_save = True
is_gray = False
is_folder = True
toggle = True
is_save_Y = False
mean=[0, 0, 0]
std=[1, 1, 1]
begin_time = time.time()
root = 'datasets/MEDataset/'

for subdir, dirs, files in os.walk(root):
    if toggle:
        toggle = False
    else:
       # Load the sequential images in each subfolder
        paths = [subdir]
        seq_loader = ImageSequence(is_folder, 'YCbCr', transforms.Compose([
                                      transforms.ToTensor()]), *paths)
        imgs = seq_loader.get_imseq()

        # separate the image channels
        NUM = len(imgs)
        c, h, w = imgs[0].size()
        Cbs = torch.zeros(NUM, h, w)
        Crs = torch.zeros(NUM, h, w)
        Ys = []
        for idx, img in enumerate(imgs):
                #print(img)
                Cbs[idx,:,:] = img[1]
                Crs[idx,:,:] = img[2]
                Ys.append(img[0].unsqueeze_(0).unsqueeze_(0).repeat(1,3,1,1)) #Y

        # Fuse the color channels (Cb and Cr) of the image sequence
        Cbs *= 255
        Crs *= 255
        Cb128 = abs(Cbs - 128);
        Cr128 = abs(Crs - 128);
        CbNew = sum((Cbs * Cb128) / (sum(Cb128).repeat(NUM, 1, 1)));
        CrNew = sum((Crs * Cr128) / (sum(Cr128).repeat(NUM, 1, 1)));
        CbNew[torch.isnan(CbNew)] = 128
        CrNew[torch.isnan(CrNew)] = 128

        # Fuse the Y channel of the image sequence
        imgs = norms(mean, std, *Ys) # normalize the Y channels
        with torch.no_grad():
            vimgs = []
            for idx, img in enumerate(imgs):
                vimgs.append(Variable(img.cuda()))
            vres = model(*vimgs)

        # Enhance the Y channel using CLAHE
        img = detransformcv2(vres[0], mean, std)                    # denormalize the fused Y channel
        y = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)                   # generate the single y channel
        
        y = y / 255                                                 # initial enhancement
        y = y * 235 + (1-y) * 16;
        y = y.astype('uint8')

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # clahe enhancement
        cy = clahe.apply(y)
        
        # Merge the YCbCr channels back and covert to RGB color space
        cyrb = np.zeros([h,w,c]).astype('uint8')
        cyrb[:,:,0] = cy
        cyrb[:,:,1] = CrNew
        cyrb[:,:,2] = CbNew
        rgb = cv2.cvtColor(cyrb, cv2.COLOR_YCrCb2RGB)
        
        
        # Save the fused image
        img = Image.fromarray(rgb)
        filename = subdir.split('/')[-1]
        filename = model_name + '-' + dataset + '-' + filename  # y channels are fused by IFCNN, cr and cb are weighted fused
        
        if is_save:
            if is_gray:
                img.convert('L').save('results/'+filename+'.png', format='PNG', compress_level=0)
            else:
                img.save('results/'+filename+'.png', format='PNG', compress_level=0)

# when evluating time costs, remember to stop writing images by setting is_save = False
proc_time = time.time() - begin_time
print('Total processing time of {} dataset: {:.3}s'.format(dataset, proc_time))

