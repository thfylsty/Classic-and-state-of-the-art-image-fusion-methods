from os import listdir
from os.path import join
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize,Grayscale
from imagecrop import FusionRandomCrop
from torchvision.transforms import functional as F

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.tif','.bmp','.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        FusionRandomCrop(crop_size),
    ])

def train_vis_ir_transform():
    return Compose([
		Grayscale(num_output_channels=3),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
		Grayscale(),
        ToTensor()
    ])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.visible_image_filenames = [join(dataset_dir+'Train_vi/', x) for x in listdir(dataset_dir+'Train_vi/') if is_image_file(x)]
        self.infrared_image_filenames = [x.replace('Train_vi/','Train_ir/') for x in self.visible_image_filenames]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.vis_ir_transform = train_vis_ir_transform()
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        visible_image = Image.open(self.visible_image_filenames[index])
        infrared_image = Image.open(self.infrared_image_filenames[index])
        crop_size = self.hr_transform(visible_image)#, infrared_image)
        visible_image, infrared_image = F.crop(visible_image,crop_size[0],crop_size[1],crop_size[2],crop_size[3])\
			,F.crop(infrared_image, crop_size[0],crop_size[1],crop_size[2],crop_size[3])
        visible_image = self.vis_ir_transform(visible_image)
        infrared_image = self.vis_ir_transform(infrared_image)
        data = torch.cat((self.lr_transform(infrared_image)[0].unsqueeze(0),self.lr_transform(visible_image)[0].unsqueeze(0)))	
        return data, infrared_image, visible_image

    def __len__(self):
        return len(self.visible_image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.visible_image_filenames = [join(dataset_dir+'vi/', x) for x in listdir(dataset_dir+'vi/') if is_image_file(x)]
        self.infrared_image_filenames = [x.replace('vi/V','ir/I') for x in self.visible_image_filenames]

    def __getitem__(self, index):
        visible_image = Image.open(self.visible_image_filenames[index])
        infrared_image = Image.open(self.infrared_image_filenames[index])
        w, h = visible_image.size

        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        visible_image1 = CenterCrop(crop_size)(visible_image)
        infrared_image1 = CenterCrop(crop_size)(infrared_image)
        visible_image1 = ToTensor()(Grayscale(num_output_channels=3)(visible_image1))
        infrared_image1 = ToTensor()(Grayscale(num_output_channels=3)(infrared_image1))
        data = torch.cat((infrared_image1[0].unsqueeze(0),visible_image1[0].unsqueeze(0)))		

        return data, infrared_image1, visible_image1

    def __len__(self):
        return len(self.infrared_image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        dataset_dir = "../road/"
        imagelist = listdir(dataset_dir+'vi/')
        imagelist.sort()
        self.visible_image_filenames = [join(dataset_dir+'vi/', x) for x in imagelist if is_image_file(x)]
        self.infrared_image_filenames = [x.replace('vi','ir') for x in self.visible_image_filenames]

    def __getitem__(self, index):
        visible_image = Image.open(self.visible_image_filenames[index])
        infrared_image = Image.open(self.infrared_image_filenames[index])
	
        visible_image = ToTensor()(Grayscale(num_output_channels=3)(visible_image))
        infrared_image = ToTensor()(Grayscale(num_output_channels=3)(infrared_image))
        data = torch.cat((infrared_image[0].unsqueeze(0),visible_image[0].unsqueeze(0)))		
        return data, infrared_image, visible_image

    def __len__(self):
        return len(self.infrared_image_filenames)
