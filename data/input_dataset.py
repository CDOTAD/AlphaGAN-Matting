import os
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import cv2 as cv
from PIL import Image


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def random_choice(trimap, crop_size=(320, 320)):
    crop_height, crop_width = crop_size

    (h, w) = trimap.size
    x = np.random.randint(int(crop_height/2), h - int(crop_height/2))
    y = np.random.randint(int(crop_width/2), w - int(crop_width/2))
    return x, y


def safe_crop(img, x, y):

    region = (x-160, y - 160, x + 160, y + 160)
    crop_img = img.crop(region)

    return crop_img


class InputDataset(data.Dataset):
    def __init__(self, dataroot):
        super(InputDataset, self).__init__()
        #self.opt = opt
        self.root = dataroot
        self.dir_input = os.path.join(dataroot, 'input')
        self.dir_trimap = os.path.join(dataroot, 'trimap')
        self.dir_alpha = os.path.join(dataroot, 'alpha')
        self.dir_bg = os.path.join(dataroot, 'bg')
        self.dir_fg = os.path.join(dataroot, 'fg')

        self.input_paths = make_dataset(self.dir_input)
        self.trimap_paths = make_dataset(self.dir_trimap)
        self.alpha_paths = make_dataset(self.dir_alpha)
        self.bg_paths = make_dataset(self.dir_bg)
        self.fg_paths = make_dataset(self.dir_fg)

        self.input_paths = sorted(self.input_paths)
        self.trimap_paths = sorted(self.trimap_paths)
        self.alpha_paths = sorted(self.alpha_paths)
        self.bg_paths = sorted(self.bg_paths)
        self.fg_paths = sorted(self.fg_paths)

        self.input_size = len(self.input_paths)
        self.trimap_size = len(self.trimap_paths)
        self.alpha_size = len(self.alpha_paths)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):

        input_path = self.input_paths[index]
        trimap_path = self.trimap_paths[index]
        alpha_path = self.alpha_paths[index]
        bg_path = self.bg_paths[index]
        fg_path = self.fg_paths[index]

        input_img = Image.open(input_path).convert('RGB')
        trimap_img = Image.open(trimap_path)
        alpha_img = Image.open(alpha_path)
        bg_img = Image.open(bg_path).convert('RGB')
        fg_img = Image.open(fg_path).convert('RGB')

        #x, y = random_choice(trimap_img)

        I = self.transform(input_img)
        T = self.transform(trimap_img)
        A = self.transform(alpha_img)
        B = self.transform(bg_img)
        F = self.transform(fg_img)

        return {'I': I, 'T': T, 'A': A,
                'B': B, 'F': F}

    def __len__(self):
        return self.input_size


