import torchvision.transforms as transforms
import numpy as np
import torch as t
from model.AlphaGAN import NetG
from PIL import Image
import os
from visualize import Visualizer
import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
vis = Visualizer('alphaGAN_eval')

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

to_pil = transforms.Compose([
    transforms.ToPILImage()
])

MODEL_DIR = '/data1/zzl/model/alphaGAN/new_trainset/netG/netG_40.pth'


def padding_img(img):

    img_size = np.shape(img)
    if len(img_size) == 3:
        (h, w, c) = img_size
        w_padding = (int(w/320) + 1) * 320
        h_padding = (int(h/320) + 1) * 320

        padding_result = np.pad(img, ((0, h_padding - h), (0, w_padding - w), (0, 0)), 'mean')

        return Image.fromarray(padding_result), int(w_padding/320), int(h_padding/320)
    elif len(img_size) == 2:
        (h, w) = img_size
        w_padding = (int(w/320) + 1) * 320
        h_padding = (int(h/320) + 1) * 320

        padding_result = np.pad(img, ((0, h_padding - h), (0, w_padding - w)), 'constant', constant_values=0)

        return Image.fromarray(padding_result), int(w_padding/320), int(h_padding/320)
    else:
        exit(1)


def clip_img(img, h_clip, w_clip):

    img_list = []
    for x in range(w_clip):
        for y in range(h_clip):
            region = (x*320, y*320, x*320+320, y*320+320)
            crop_img = img.crop(region)
            crop_img = transform(crop_img)
            crop_img = crop_img[None]

            img_list.append(crop_img)

    crop_img = img_list[0]
    for i in range(1, len(img_list)):
        crop_img = t.cat((crop_img, img_list[i]), dim=0)

    return crop_img


def combination(img_list, h_clip, w_clip):

    column =[]
    for y in range(w_clip):
        for x in range(h_clip):
            if x == 0:
                column.append(img_list[y*h_clip + x])
            else:
                column[y] = t.cat((column[y], img_list[y*h_clip + x]), dim=1)

    com = column[0]

    for i in range(1, len(column)):
        com = t.cat((com, column[i]), dim=2)

    return com


def clip_input():
    with t.no_grad():
        net_G = NetG().cuda()
        net_G.eval()
        net_G.load_state_dict(t.load(MODEL_DIR, map_location=t.device('cpu')))

        img_root = '/data1/zzl/dataset/matting/alphamatting/input_lowers'
        trimap_root = '/data1/zzl/dataset/matting/alphamatting/trimap_lowres'

        img_name = os.listdir(img_root)

        for name in tqdm.tqdm(img_name):
            for i in range(1, 4):

                trimap_floder = 'Trimap' + str(i)

                img = Image.open(os.path.join(img_root, name))
                print('img_size', img.size)
                print('img_shape', np.shape(img))
                img, w_clip, h_clip = padding_img(img)
                print('img.shape', np.shape(img))
                # print('img', w_clip, h_clip)

                crop_img = clip_img(img, h_clip, w_clip)

                img = transform(img)

                trimap = Image.open(os.path.join(trimap_root, trimap_floder, name))
                (h_r, w_r) = np.shape(trimap)
                trimap, w_clip, h_clip = padding_img(trimap)

                # print('trimap', w_clip, h_clip)

                crop_tri = clip_img(trimap, h_clip, w_clip)

                input_img = t.cat((crop_img, crop_tri), dim=1)
                input_img = input_img.cuda()

                fake_alpha = net_G(input_img)

                com_fake = combination(fake_alpha, h_clip, w_clip)

                vis.images(com_fake.cpu().numpy(), win='fake_alpha')
                vis.images(img.numpy() * 0.5 + 0.5, win='input')
                # print(fake_alpha[0].size())
                # print(com_fake.size())
                save_alpha = to_pil(com_fake.cpu())
                save_alpha = save_alpha.convert('L')
                print('fake_alpha.shape', np.shape(save_alpha))
                box = (0, 0, w_r, h_r)
                save_alpha = save_alpha.crop(box)

                if not os.path.exists(trimap_floder):
                    os.mkdir(trimap_floder)
                print('save_alpha.shape', np.shape(save_alpha))
                save_alpha.save(trimap_floder + '/' + name)
    return


def full_input():
    with t.no_grad():
        net_G = NetG().cuda()
        net_G.eval()
        net_G.load_state_dict(t.load(MODEL_DIR, map_location=t.device('cpu')))

        img_root = '/data1/zzl/dataset/matting/alphamatting/input_lowers'
        trimap_root = '/data1/zzl/dataset/matting/alphamatting/trimap_lowres'

        img_name = os.listdir(img_root)

        for name in tqdm.tqdm(img_name):
            for i in range(1, 4):

                trimap_floder = 'Trimap' + str(i)

                img = Image.open(os.path.join(img_root, name))
                img, _1, _2 = padding_img(img)
                img = transform(img)

                trimap = Image.open(os.path.join(trimap_root, trimap_floder, name))
                (h_r, w_r) = np.shape(trimap)
                trimap, _1, _2 = padding_img(trimap)
                (w, h) = np.shape(trimap)
                trimap = np.reshape(trimap, (w, h, 1))
                trimap = transform(trimap)

                input_img = t.cat((img, trimap), dim=0)
                input_img = input_img[None]
                input_img = input_img.cuda()

                fake_alpha = net_G(input_img)
                vis.images(fake_alpha.cpu().numpy(), win='fake_alpha')
                vis.images(img.numpy() * 0.5 + 0.5, win='input')
                #print(fake_alpha[0].size())
                save_alpha = to_pil(fake_alpha.cpu()[0])
                save_alpha = save_alpha.convert('L')

                box = (0, 0, w_r, h_r)
                save_alpha = save_alpha.crop(box)

                if not os.path.exists(trimap_floder):
                    os.mkdir(trimap_floder)
                print(np.shape(save_alpha))
                save_alpha.save(trimap_floder + '/' + name)


def resize_input():
    with t.no_grad():
        net_G = NetG().cuda()
        net_G.eval()
        net_G.load_state_dict(t.load(MODEL_DIR, map_location=t.device('cpu')))

        img_root = '/data1/zzl/dataset/matting/alphamatting/input_lowers'
        trimap_root = '/data1/zzl/dataset/matting/alphamatting/trimap_lowres'

        img_name = os.listdir(img_root)

        for name in tqdm.tqdm(img_name):
            for i in range(1, 4):

                trimap_floder = 'Trimap' + str(i)

                img = Image.open(os.path.join(img_root, name))
                (w, h) = img.size
                w_large = w//320 + 1
                h_large = h//320 + 1

                img = img.resize((w_large * 320, h_large * 320))

                img = transform(img)

                trimap = Image.open(os.path.join(trimap_root, trimap_floder, name))
                trimap = trimap.resize((w_large * 320, h_large * 320))
                (w, h) = np.shape(trimap)
                trimap = np.reshape(trimap, (w, h, 1))
                trimap = transform(trimap)

                input_img = t.cat((img, trimap), dim=0)
                input_img = input_img[None]
                input_img = input_img.cuda()

                fake_alpha = net_G(input_img)
                vis.images(fake_alpha.cpu().numpy(), win='fake_alpha')
                vis.images(img.numpy() * 0.5 + 0.5, win='input')
                # print(fake_alpha[0].size())
                save_alpha = to_pil(fake_alpha.cpu()[0])
                save_alpha = save_alpha.convert('L')
                box = (0, 0, w, h)
                save_alpha = save_alpha.crop(box)
                if not os.path.exists(trimap_floder):
                    os.mkdir(trimap_floder)
                print(np.shape(save_alpha))
                save_alpha.save(trimap_floder + '/' + name)


full_input()

