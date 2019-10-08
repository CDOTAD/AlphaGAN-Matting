import torch as t
import numpy as np
import tqdm
import os
from PIL import Image
import cv2 as cv
import torchvision.transforms as transforms


class Config(object):
    def __init__(self):
        return


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '% is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class Tester(object):
    def __init__(self, net_G, test_root):
        self.net_G = net_G
        self.dataroot = test_root
        self.dir_images = os.path.join(self.dataroot, 'input')
        self.dir_trimap = os.path.join(self.dataroot, 'trimaps')
        self.dir_alpha = os.path.join(self.dataroot, 'alpha')

        self.images_paths = sorted(make_dataset(self.dir_images))
        self.trimap_pahts = sorted(make_dataset(self.dir_trimap))

        self.size = len(self.images_paths)

    @t.no_grad()
    def test(self, vis):

        total_sad = 0
        total_mse = 0

        for index in range(self.size):
            data = self._get_next(index)

            image = data['I']
            trimap = data['T']
            alpha = data['A']

            fake_alpha = self.inference_img_whole(image, trimap, vis)
            alpha = alpha.astype(np.float32) / 255.

            fake_alpha[trimap == 255] = 1.
            fake_alpha[trimap == 0] = 0.

            mask = np.zeros(trimap.shape)
            mask[trimap == 0] = 1
            mask[trimap == 255] = 1
            mask = 1 - mask
            unknow_region_size = mask.sum()

            # print(np.shape(fake_alpha))
            # print(np.shape(alpha))

            sad = np.abs(fake_alpha - alpha)
            sad = sad.sum() / 1000

            mse = np.power(fake_alpha - alpha, 2)
            mse = mse.sum() / unknow_region_size

            total_sad += sad
            total_mse += mse

        # vis.images(alpha, win='real_alpha')
        # vis.images(fake_alpha, win='fake_alpha')
        # vis.images(np.array(trimap), win='tirmap')
        average_sad = total_sad / self.size
        average_mse = total_mse / self.size

        return {'sad': average_sad, 'mse': average_mse}

    @t.no_grad()
    def inference_onece(self, scale_image, scale_trimap, vis):

        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        scale_img_rgb = cv.cvtColor(scale_image, cv.COLOR_BGR2RGB)
        tensor_img = normalize(Image.fromarray(scale_img_rgb)).unsqueeze(0)

        tensor_trimap = normalize(Image.fromarray(scale_trimap)).unsqueeze(0)
        # print('np.unique(tensor_trimap.numpy())', np.unique(tensor_trimap.numpy()))

        tensor_img = tensor_img.cuda()
        tensor_trimap = tensor_trimap.cuda()
        # print(tensor_img.size())
        # print(tensor_trimap.size())
        input_t = t.cat((tensor_img, tensor_trimap), dim=1)

        pred_mattes = self.net_G(input_t)

        pred_mattes = pred_mattes.data
        pred_mattes = pred_mattes.cpu()
        pred_mattes = pred_mattes.numpy()[0, 0, :, :]
        # print(np.unique(scale_trimap))
        mask = np.zeros(scale_trimap.shape)
        mask[scale_trimap == 0] = 1
        mask[scale_trimap == 255] = 1
        mask = 1 - mask
        alpha = (1 - mask) * (scale_trimap / 255.) + mask * pred_mattes

        vis.images(tensor_img.cpu().numpy()*0.5 + 0.5, win='real_img')
        vis.images(np.array(scale_trimap), win='trimap')
        vis.images(np.array(alpha), win='fake_alpha')
        vis.images(np.array(pred_mattes), win='pred_mattes')

        return alpha

    @t.no_grad()
    def inference_img_whole(self, img, trimap, vis):
        h, w, c = img.shape
        new_h = min(320, h-(h%32))
        new_w = min(320, w-(w%32))

        scale_img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        scale_trimap = cv.resize(trimap, (new_w, new_h), interpolation=cv.INTER_LINEAR)

        pred_mattes = self.inference_onece(scale_img, scale_trimap, vis)

        origin_pred_mattes = cv.resize(pred_mattes, (w, h), interpolation=cv.INTER_LINEAR)
        return origin_pred_mattes

    def _get_relate_alpha(self, image_path):
        paths = image_path.split('/')
        fname = paths[-1].split('.')[0]
        max_len = 0
        for i, char in enumerate(fname):
            if char == '_':
                max_len = i

        alpha_name = fname[:max_len] + '.png'
        # print(alpha_name)
        # print('os.path.exists(self.dir_alpha)', os.path.exists(self.dir_alpha))

        return os.path.join(self.dir_alpha, alpha_name)

    def _get_next(self, index):
        image_path = self.images_paths[index]
        trimap_path = self.trimap_pahts[index]

        alpha_path = self._get_relate_alpha(image_path)

        image = cv.imread(image_path)[:, :, :3]
        trimap = cv.imread(trimap_path)[:, :, 0]
        alpha = cv.imread(alpha_path)[:, :, 0]

        return {'I': image, 'T': trimap, 'A': alpha}











