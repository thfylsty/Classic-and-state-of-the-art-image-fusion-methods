import argparse
import os

import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data
import torchvision.utils as utils
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import  RandomCrop, ToTensor, ToPILImage

from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize,Grayscale
# import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, TestDatasetFromFolder
from model import Generator
from data_utils import display_transform


parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=1, type=int, choices=[1, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=1, type=int, help='train epoch number')

opt = parser.parse_args()

CROP_SIZE = opt.crop_size #裁剪会带来拼尽问题嘛
UPSCALE_FACTOR = opt.upscale_factor #上采样
NUM_EPOCHS = opt.num_epochs #轮数

val_set = TestDatasetFromFolder('../', upscale_factor=UPSCALE_FACTOR) #测试集导入

MODEL_NAME = 'netG_epoch_1_152.pth'
netG = Generator(UPSCALE_FACTOR).eval()
netG.cuda()
netG.load_state_dict(torch.load('./epochs/' + MODEL_NAME))
val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

epoch =1
out_path = '/data/lpw/ResnetFusion/test_results/SRF_' + str(UPSCALE_FACTOR) + '/'#输出路径


val_bar = tqdm(val_loader) #验证集的进度条
val_images = []
for val_lr , val_lr_restore, val_hr in val_bar:
    batch_size = val_lr.size(0)
    lr = Variable(val_lr)
    hr = Variable(val_hr)
    if torch.cuda.is_available():
        lr = lr.cuda()
        hr = hr.cuda()
    sr = netG(lr)#验证集生成超分图片

    val_images.extend(
             display_transform()(sr.data.cpu().squeeze(0)))

# val_images = torch.chunk(val_images, val_images.size(0) // 15)#看不懂，骚操作

for i,image in enumerate(val_images):
    print('{}th size {}'.format(i,image.size()))
val_save_bar = tqdm(val_images, desc='[saving training results]')
index = 1
for image in val_images:
    # image = utils.make_grid(image, nrow=3, padding=2,scale_each=True)
    utils.save_image(image,'o/%d.bmp' % (index), nrow=3,padding=2)#验证集存储数据
    index += 1
