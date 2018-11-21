import torchvision.transforms as transforms
import numpy as np
import torch as t
from model.AlphaGAN import NetG
from visualize import Visualizer
from PIL import Image

device = t.device('cuda:0')
vis = Visualizer('alphaGAN_test')

net_G = NetG()
net_G.load_state_dict(t.load('/home/zzl/model/alphaGAN/netG/new_aspp/netG_5.pth'))

net_G.to(device)

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


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


@t.no_grad()
def test():

    real_img = Image.open('li_in.jpg').convert('RGB')
    w, h = real_img.size

    h_clip = int(h / 320)
    w_clip = int(w / 320)

    real_img = real_img.resize((w_clip * 320, h_clip * 320))
    crop_img = clip_img(real_img, h_clip, w_clip)
    print(h_clip, w_clip)

    tri_img = Image.open('li_tri.jpg').convert('L')
    tri_img = tri_img.resize((w_clip * 320, h_clip * 320))
    crop_tri = clip_img(tri_img, h_clip, w_clip)

    input_img = t.cat((crop_img, crop_tri), dim=1)
    input_img = input_img.to(device)

    fake_alpha = net_G(input_img)

    print(input_img.size())

    print(crop_img.size())

    com_img = combination(crop_img, h_clip, w_clip)
    print(com_img.size())

    com_fake = combination(fake_alpha, h_clip, w_clip)

    img = com_fake.detach().cpu() * com_img

    vis.images(com_fake.detach().cpu().numpy(), win='fake-alpha')
    vis.images(img.numpy() * 0.5 + 0.5, win='real')
    vis.images(crop_img.numpy() * 0.5 + 0.5, win='real_img')
    vis.images(com_img.numpy()*0.5 + 0.5, win='com_img')


@t.no_grad()
def test_1():

    real_img = Image.open('ball_input.png').convert('RGB')
    real_img = real_img.resize((320, 320))
    real_img = transform(real_img)
    real_img = real_img[None]
    tri_img = Image.open('ball_tri.png').convert('L')
    tri_img = tri_img.resize((320, 320))
    tri_img = transform(tri_img)
    tri_img = tri_img[None]
    input_img = t.cat((real_img, tri_img), dim=1)
    input_img = input_img.to(device)

    fake_alpha = net_G(input_img)

    com_img = real_img * fake_alpha.detach().cpu()

    vis.images(fake_alpha.detach().cpu().numpy(), win='fake-alpha')
    vis.images(real_img.numpy() * 0.5 + 0.5, win='real_img')
    vis.images(com_img.numpy()*0.5 + 0.5, win='com_img')


test()
