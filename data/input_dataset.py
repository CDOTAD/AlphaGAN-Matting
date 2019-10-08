import os
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import cv2 as cv
import random
from PIL import Image
import math


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def gen_trimap(alpha):
    k_size = random.choice(range(2, 5))
    iterations = np.random.randint(5, 15)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv.dilate(alpha, kernel, iterations=iterations)
    eroded = cv.erode(alpha, kernel, iterations=iterations)
    trimap = np.zeros(alpha.shape, dtype=np.uint8)
    trimap.fill(128)

    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0

    return trimap


class InputDataset(data.Dataset):
    def __init__(self, dataroot, training_file, bg_root):
        super(InputDataset, self).__init__()

        self.crop = [320, 480, 640]

        self.root = dataroot
        self.dir_input = os.path.join(dataroot, 'merged_cv')
        self.dir_alpha = os.path.join(dataroot, 'alpha')
        self.dir_fg = os.path.join(dataroot, 'fg')
        self.bg_root = bg_root

        self.training_sets = [training_set for training_set in open(training_file).read().splitlines()]

        self.input_size = len(self.training_sets)

        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.transform_l = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.transform_alpha = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):

        im_name, bg_name, input_img = self.training_sets[index].split(' ')

        input_img = cv.imread(os.path.join(self.dir_input, input_img))[:, :, :3]
        alpha_img = cv.imread(os.path.join(self.dir_alpha, im_name))[:, :, 0]
        bg_img = cv.imread(os.path.join(self.bg_root, bg_name))[:, :, :3]
        fg_img = cv.imread(os.path.join(self.dir_fg, im_name))[:, :, :3]

        h, w, c = input_img.shape
        bh, bw = bg_img.shape[:2]

        wratio = float(w) / float(bw)
        hratio = float(h) / float(bh)
        ratio = wratio if wratio > hratio else hratio
        if ratio > 1:
            bg_img = cv.resize(src=bg_img, dsize=(math.ceil(bw * ratio), math.ceil(bh*ratio)),
                               interpolation=cv.INTER_CUBIC)

        bg_img = np.array(bg_img[0:h, 0:w], np.uint8)

        rand_crop = random.randint(0, len(self.crop) - 1)
        crop_h = self.crop[rand_crop]
        crop_w = self.crop[rand_crop]

        wratio = float(crop_w) / w
        hratio = float(crop_h) / h
        ratio = wratio if wratio > hratio else hratio
        if ratio > 1:
            new_w = int(w * ratio + 1.0)
            new_h = int(w * ratio + 1.0)
            input_img = cv.resize(input_img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
            alpha_img = cv.resize(alpha_img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
            fg_img = cv.resize(fg_img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
            bg_img = cv.resize(bg_img, (new_w, new_h), interpolation=cv.INTER_LINEAR)

        input_img, alpha_img, fg_img, bg_img = self._random_crop(input_img, alpha_img, fg_img, bg_img, crop_h, crop_w)
        if input_img.shape[0] != 320 or input_img.shape[1] != 320:
            input_img = cv.resize(input_img, (320, 320), interpolation=cv.INTER_LINEAR)
            alpha_img = cv.resize(alpha_img, (320, 320), interpolation=cv.INTER_LINEAR)
            fg_img = cv.resize(fg_img, (320, 320), interpolation=cv.INTER_LINEAR)
            bg_img = cv.resize(bg_img, (320, 320), interpolation=cv.INTER_LINEAR)

        # random flip
        if random.random() < 0.5:
            input_img = cv.flip(input_img, 1)
            alpha_img = cv.flip(alpha_img, 1)
            fg_img = cv.flip(fg_img, 1)
            bg_img = cv.flip(bg_img, 1)

        trimap = gen_trimap(alpha_img)
        input_img = cv.cvtColor(input_img, cv.COLOR_BGR2RGB)
        fg_img = cv.cvtColor(fg_img, cv.COLOR_BGR2RGB)
        bg_img = cv.cvtColor(bg_img, cv.COLOR_BGR2RGB)

        I = self.transform_rgb(Image.fromarray(input_img))
        A = self.transform_alpha(Image.fromarray(alpha_img))
        T = self.transform_l(Image.fromarray(trimap))
        F = self.transform_rgb(Image.fromarray(fg_img))
        B = self.transform_rgb(Image.fromarray(bg_img))

        return {'I': I, 'A': A, 'T': T, 'F': F, 'B': B}

    def _random_crop(self, img, alpha, fg, bg, crop_h, crop_w):
        h, w = alpha.shape

        target = np.where((alpha > 0) & (alpha < 255))
        delta_h = center_h = crop_h / 2
        delta_w = center_w = crop_w / 2

        if len(target[0]) > 0:
            rand_int = np.random.randint(len(target[0]))
            center_h = min(max(target[0][rand_int], delta_h), h - delta_h)
            center_w = min(max(target[1][rand_int], delta_w), w - delta_w)

        start_h = int(center_h - delta_h)
        start_w = int(center_w - delta_w)
        end_h = int(center_h + delta_h)
        end_w = int(center_w + delta_w)

        img = img[start_h: end_h, start_w: end_w]
        alpha = alpha[start_h: end_h, start_w: end_w]
        bg = bg[start_h: end_h, start_w: end_w]
        fg = fg[start_h: end_h, start_w: end_w]

        return img, alpha, fg, bg

    def __len__(self):
        return self.input_size





