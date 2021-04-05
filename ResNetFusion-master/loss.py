import torch
from torch import nn
from torchvision.models.vgg import vgg16,vgg13
import math
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.transforms import ToPILImage
import random
class GeneratorLoss(nn.Module):
    def __init__(self,batchSize):
        super(GeneratorLoss, self).__init__()
        vgg = vgg13(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:25]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.laplace = LaplacianLoss()
        self.blur_kernel = get_gaussian_kernel(kernel_size=3)
        self.blur_kernel2 = get_gaussian_kernel(kernel_size=5)
        self.blur_kernel3 = get_gaussian_kernel(kernel_size=7)
        self.adversarial_criterion = nn.BCELoss()

    def forward(self, out_labels, out_images, target_images, target_ir, fake_data, vi_data):
        # Adversarial Loss
        ones_const = Variable(torch.ones(out_labels.size(0), 1)).cuda()
        adversarial_loss = self.adversarial_criterion(out_labels, ones_const)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        addition_penalty = self.blur_kernel(self.tv_loss(target_ir))
        addition_penalty = F.pad(addition_penalty,(1,2,1,2),mode='replicate')
        alpha = 100

        # gradient compare
        tv_image = F.pad(self.tv_loss(target_ir),(1,0,1,0),mode='replicate')
        laplacian_image = F.pad(self.laplace(target_ir),(1,1,1,1),mode='replicate')
        addition_laplacian_image = F.pad(self.blur_kernel(self.laplace(target_ir)),(2,2,2,2),mode='replicate')
        addition_laplacian_image2 = F.pad(self.blur_kernel2(self.laplace(target_ir)),(3,3,3,3),mode='replicate')
        addition_laplacian_image3 = F.pad(self.blur_kernel3(self.laplace(target_ir)),(4,4,4,4),mode='replicate')
        pyramid_addition = addition_laplacian_image + addition_laplacian_image2 + addition_laplacian_image3
        
        tuple_ir = torch.chunk(target_ir,out_labels.size(0),dim=0)
        show_ir = torch.cat(tuple_ir,2).squeeze()
        
        
        tuple_ir_laplace = torch.chunk(laplacian_image * alpha, out_labels.size(0),dim=0)
        show_ir_laplace = torch.cat(tuple_ir_laplace, 2).squeeze()
        tuple_ir_laplace_heatmap = torch.chunk(addition_laplacian_image*alpha/2,out_labels.size(0),dim=0)
        show_ir_laplace_heatmap = torch.cat(tuple_ir_laplace_heatmap,2).squeeze()
        tuple_ir_laplace_heatmap2 = torch.chunk(addition_laplacian_image2*alpha/2,out_labels.size(0),dim=0)
        show_ir_laplace_heatmap2 = torch.cat(tuple_ir_laplace_heatmap2,2).squeeze()
        tuple_ir_laplace_heatmap3 = torch.chunk(addition_laplacian_image3*alpha/2,out_labels.size(0),dim=0)
        show_ir_laplace_heatmap3 = torch.cat(tuple_ir_laplace_heatmap3,2).squeeze()
        tuple_pyramid_heatmap = torch.chunk(pyramid_addition*alpha/2,out_labels.size(0),dim=0)
        show_pyramid_heatmap = torch.cat(tuple_pyramid_heatmap,2).squeeze()
        
        ir_heatmap = torch.cat((show_ir,show_ir_laplace,show_ir_laplace_heatmap,show_ir_laplace_heatmap2,show_ir_laplace_heatmap3,show_pyramid_heatmap),2)
        coefficient = pyramid_addition * alpha/2 + 1
        image_loss = ((out_images - target_ir).abs()*coefficient).mean()
        # TV Loss
        fusion_tv = self.tv_loss(target_images) + self.tv_loss(target_ir)
        tv_loss = self.mse_loss(self.tv_loss(out_images) , fusion_tv)
        ir_tv_loss = self.mse_loss(self.tv_loss(out_images) , self.tv_loss(target_ir))

        d_perception_loss = self.mse_loss(fake_data, vi_data)
        return image_loss + 0.005 * adversarial_loss + 0.2 * d_perception_loss + 1000 * tv_loss,0.005 * adversarial_loss, image_loss, 1000 * tv_loss,0.2 * d_perception_loss# + 1* ir_tv_loss


def get_gaussian_kernel(kernel_size=5, sigma=5, channels=3):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter
    
class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2)
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2)
        return self.tv_loss_weight * 2 * (h_tv[:, :, :h_x - 1, :w_x - 1] + w_tv[:, :, :h_x - 1, :w_x - 1])

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class LaplacianLoss(nn.Module):
    def __init__(self,channels=3):
        super(LaplacianLoss, self).__init__()
        # laplacian_kernel = torch.tensor([[1,1,1],[1,-8,1],[1,1,1]]).float()
        laplacian_kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]]).float()
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
        laplacian_kernel = laplacian_kernel.repeat(channels, 1, 1, 1)
        self.laplacian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=3, groups=channels, bias=False)

        self.laplacian_filter.weight.data = laplacian_kernel
        self.laplacian_filter.weight.requires_grad = False
    def forward(self,x):
        return self.laplacian_filter(x) ** 2
if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
