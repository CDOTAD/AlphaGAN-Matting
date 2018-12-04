import torchvision.transforms as transforms
import numpy as np
import torch as t
import os
import math
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

def com_input(fg, alpha):
    COCO_ROOT = '/data1/zzl/dataset/mscoco/train2014'

    ball = Image.open(fg).convert('RGB')
    alpha = Image.open(alpha).convert('L')

    to_tensor = transforms.Compose([
        transforms.ToTensor()
    ])

    # transform to tensor
    ball_tensor = to_tensor(ball)
    alpha_tensor = to_tensor(alpha)

    # random chose bg
    bg_list = os.listdir(COCO_ROOT)
    index = np.random.randint(0, len(bg_list))
    bg = Image.open(os.path.join(COCO_ROOT, bg_list[index]))

    # resize bg
    bg_bbox = bg.size
    fg_bbox = ball.size

    w = fg_bbox[0]
    h = fg_bbox[1]

    bw = bg_bbox[0]
    bh = bg_bbox[1]

    wratio = w / bw
    hratio = h / bh

    ratio = wratio if wratio > hratio else hratio
    if ratio > 1:
        bg = bg.resize((math.ceil(bw * ratio), math.ceil(bh * ratio)), Image.BICUBIC)

    bg = bg.crop((0, 0, w, h))

    bg_tensor = to_tensor(bg)

    input_tensor = alpha_tensor * ball_tensor + (1 - alpha_tensor) * bg_tensor

    to_pil = transforms.Compose([
        transforms.ToPILImage()
    ])

    return to_pil(input_tensor)

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

    return

@t.no_grad()
def no_clip():
    # get input trimap
    tri_img = Image.open('ball_tri.png').convert('L')
    tri_img_tensor = transform(tri_img)
    tri_img_tensor = tri_img_tensor[None]
    # compose the fg and bg
    com_img = com_input('ball_input.png', 'ball_alpha.png')
    com_img_tensor = transform(com_img)
    com_img_tensor = com_img_tensor[None]

    # compose the input
    input_img = t.cat((com_img_tensor, tri_img_tensor), dim=1)
    input_img = input_img.to(device)

    # get the generated alpha
    fake_alpha = net_G(input_img)
    print(net_G)

    print(input_img.size())

    # get the fg via the alpha
    img = fake_alpha.detach().cpu() * com_img_tensor

    vis.images(fake_alpha.detach().cpu().numpy(), win='fake-alpha')
    vis.images(img.numpy() * 0.5 + 0.5, win='real')
    # vis.images(crop_img.numpy() * 0.5 + 0.5, win='real_img')
    vis.images(com_img_tensor.numpy() * 0.5 + 0.5, win='com_img')
    vis.images(tri_img_tensor.numpy() * 0.5 + 0.5, win='tri_map')

    return


test()
