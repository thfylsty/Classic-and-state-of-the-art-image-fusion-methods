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
import pytorch_ssim
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

val_set = TestDatasetFromFolder('/data/lpw/FusionDataset/tmp_val/', upscale_factor=UPSCALE_FACTOR) #测试集导入
for pthdir in os.listdir('/data/lpw/ResnetFusion/epochs/'):
    for MODEL_NAME in os.listdir('/data/lpw/ResnetFusion/epochs/'+pthdir):
        # MODEL_NAME = 'netG_epoch_1_4000.pth'

        netG = Generator(UPSCALE_FACTOR).eval()
        netG.cuda()
        netG.load_state_dict(torch.load('/data/lpw/ResnetFusion/epochs/' + pthdir+'/'+MODEL_NAME))
        val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1, shuffle=False)

        epoch =1
        out_path = '/data/lpw/ResnetFusion/quota_results/SRF_' + str(UPSCALE_FACTOR) + '/'+pthdir+'/'+MODEL_NAME+'/'#输出路径
        print(out_path)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        val_bar = tqdm(val_loader) #验证集的进度条
        val_images = []
        index = 1
        for val_lr , val_lr_restore, val_hr in val_bar:
            batch_size = val_lr.size(0)
            lr = Variable(val_lr)
            hr = Variable(val_hr)
            if torch.cuda.is_available():
                lr = lr.cuda()
            sr = netG(lr)#验证集生成超分图片
                
            sr =  Compose([ToPILImage(),Grayscale()])(sr.data.cpu().squeeze(0))
            sr.save( out_path + 'epoch_%d_index_%02d.png' % (epoch, index))#验证集存储数据
            index += 1
               
#    val_images.extend(
#            [display_transform()(val_lr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
#             display_transform()(sr.data.cpu().squeeze(0))])
#val_images = torch.stack(val_images)#看不懂
#val_images = torch.chunk(val_images, val_images.size(0) // 15)#看不懂，骚操作
#val_save_bar = tqdm(val_images, desc='[saving training results]')
#index = 1
#for image in val_save_bar:
#    image = utils.make_grid(image, nrow=3, padding=2,scale_each=True)
#    utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), nrow=3,padding=2)#验证集存储数据
#    index += 1
