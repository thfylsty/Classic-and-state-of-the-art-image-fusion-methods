import argparse
import os
from math import log10

import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn

from tensorboardX import SummaryWriter

from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import Generator, Discriminator
from torchvision.transforms import Grayscale

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=1, type=int, choices=[1, 2, 4],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=40, type=int, help='train epoch number')

opt = parser.parse_args()

CROP_SIZE = opt.crop_size #裁剪会带来拼尽问题嘛
UPSCALE_FACTOR = opt.upscale_factor #上采样
NUM_EPOCHS = opt.num_epochs #轮数

train_set = TrainDatasetFromFolder('../../data/', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR) #训练集导入
val_set = ValDatasetFromFolder('../../data/', upscale_factor=UPSCALE_FACTOR) #测试集导入
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True) #训练集制作
val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

netG = Generator(UPSCALE_FACTOR)	#网络模型
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD = Discriminator()
small_netD = nn.Sequential(*list((netD.net))[:23]).eval()
#small_shadow_netD = nn.Sequential(*list((netD.net))[:2]).eval()
#for param in small_netD.parameters():
#	param.requires_grad = False
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

generator_criterion = GeneratorLoss(batchSize=64) #生成器损失
adversarial_criterion = nn.BCELoss()

if torch.cuda.is_available():
    netG.cuda()
    netD.cuda()
    generator_criterion.cuda()

optimizerG = optim.Adam(netG.parameters())
optimizerD = optim.Adam(netD.parameters())

results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
writer = SummaryWriter()
count = 0

for epoch in range(1, NUM_EPOCHS + 1):
    train_bar = tqdm(train_loader)#进度条
        # Adversarial Loss
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0, 'g_adversarial_loss':0,'g_image_loss':0, 'g_tv_loss':0,'g_ir_tv_loss':0, 'g_d_perception_loss':0}

    netG.train()
    netD.train()
    for data, ir, target in train_bar:
        g_update_first = True
        batch_size = data.size(0)
        running_results['batch_sizes'] += batch_size #pytorch batch_size 不一定一致

        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        real_img = Variable(target)
        target_ir = Variable(ir)
        if torch.cuda.is_available():
            real_img = real_img.cuda()
            target_ir = target_ir.cuda()
            target_real = Variable(torch.rand(batch_size,1)*0.5 + 0.7).cuda()
            target_fake = Variable(torch.rand(batch_size,1)*0.3).cuda()

        z = Variable(data)
        if torch.cuda.is_available():
            z = z.cuda()
        fake_img = netG(z) #生成图片

        netD.zero_grad()
        real_out = netD(real_img) #判决器判真
        fake_out = netD(fake_img) #判决器判假
        d_loss = adversarial_criterion(real_out, target_real) + \
                             adversarial_criterion(fake_out, target_fake)
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        small_netD = nn.Sequential(*list((netD.net))[:23]).eval()
        for param in small_netD.parameters():
            param.requires_grad = False
        lamda_d_loss_real = small_netD(Variable(target,requires_grad=False).cuda())
        lamda_d_loss_fake = small_netD(Variable(fake_img.data,requires_grad=False).cuda())
        with torch.no_grad():
            lamda_d_loss_ir = small_netD(Variable(ir).cuda())

        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################
        netG.zero_grad()
        g_loss,adversarial_loss, img_loss ,tv_loss, d_perception_loss = generator_criterion(fake_out, fake_img, real_img,target_ir,lamda_d_loss_fake, lamda_d_loss_real)
        g_loss.backward()
        optimizerG.step()
        fake_img = netG(z)#生成图片
        fake_out = netD(fake_img)#判决器判假

        g_loss,adversrial_loss, img_loss, tv_loss, d_perception_loss = generator_criterion(fake_out, fake_img, real_img,target_ir, lamda_d_loss_fake, lamda_d_loss_real)#计算两次G_loss
        running_results['g_loss'] += g_loss.item() * batch_size
        d_loss = adversarial_criterion(real_out, target_real) + \
                             adversarial_criterion(fake_out, target_fake)
        running_results['d_loss'] += d_loss.item() * batch_size
        running_results['d_score'] += real_out.data[0] * batch_size
        running_results['g_score'] += fake_out.data[0] * batch_size
        running_results['g_image_loss'] += img_loss.item() * batch_size
        # Adversarial Loss
        running_results['g_tv_loss'] += tv_loss.item() * batch_size
        running_results['g_adversarial_loss'] += adversarial_loss.item() * batch_size
        running_results['g_d_perception_loss'] += d_perception_loss.item() * batch_size

        train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
            epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
            running_results['g_loss'] / running_results['batch_sizes'],
            running_results['d_score'] / running_results['batch_sizes'],
            running_results['g_score'] / running_results['batch_sizes']))
        writer.add_scalar('d_loss', running_results['d_loss'] / running_results['batch_sizes'],count)
        writer.add_scalar('g_loss', running_results['g_loss'] / running_results['batch_sizes'],count)
        writer.add_scalar('d_score', running_results['d_score'] / running_results['batch_sizes'],count)
        writer.add_scalar('g_score', running_results['g_score'] / running_results['batch_sizes'],count)
        writer.add_scalar('g_image_loss', running_results['g_image_loss'] / running_results['batch_sizes'],count)
        writer.add_scalar('g_adversarial_loss', running_results['g_adversarial_loss'] / running_results['batch_sizes'],count)
        writer.add_scalar('g_tv_loss', running_results['g_tv_loss'] / running_results['batch_sizes'],count)
        writer.add_scalar('g_d_perception_loss', running_results['g_d_perception_loss'] / running_results['batch_sizes'],count)
        count += 1
    netG.eval()
    out_path = './training_results/SRF_' + str(UPSCALE_FACTOR) + '/'#输出路径
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    val_bar = tqdm(val_loader) #验证集的进度条
    valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
    val_images = []
    for val_lr , val_lr_restore, val_hr in val_bar:
        batch_size = val_lr.size(0)
        valing_results['batch_sizes'] += batch_size
        with torch.no_grad():
            lr = Variable(val_lr)
            hr = Variable(val_hr)
            if torch.cuda.is_available():
                lr = lr.cuda()
                hr = hr.cuda()
            sr = netG(lr)#验证集生成图片

        val_images.extend(
            [display_transform()(val_lr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
             display_transform()(sr.data.cpu().squeeze(0))])
    val_images = torch.stack(val_images)#
    val_images = torch.chunk(val_images, val_images.size(0) // 15)#骚操作
    # val_save_bar = tqdm(val_images, desc='[saving training results]')
    # index = 1
    # for image in val_save_bar:
    #     image = utils.make_grid(image, nrow=3, padding=5)
    #     utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)#验证集存储数据
    #     index += 1

    # save model parameters
    torch.save(netG.state_dict(), './epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))#存储网络参数
    torch.save(netD.state_dict(), './epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))#
    results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
    results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
    results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
    results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])

writer.close()		
