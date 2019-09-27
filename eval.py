import torchvision.transforms as transforms
import numpy as np
import torch as t
import torch.nn as nn
from model.AlphaGAN import NetG
import os
import cv2
from PIL import Image
import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

MODEL_DIR = 'checkpoint/adv_train/netG/netG_18.pth'


@t.no_grad()
def inference_onece(model, scale_img, scale_trimap):

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    scale_img_rgb = cv2.cvtColor(scale_img, cv2.COLOR_BGR2RGB)

    tensor_img = normalize(scale_img_rgb).unsqueeze(0)

    tensor_trimap = normalize(Image.fromarray(scale_trimap)).unsqueeze(0)

    tensor_img = tensor_img.cuda()
    tensor_trimap = tensor_trimap.cuda()

    input_t = t.cat((tensor_img, tensor_trimap), dim=1)

    pred_mattes = model(input_t)

    pred_mattes = pred_mattes.data
    pred_mattes = pred_mattes.cpu()
    pred_mattes = pred_mattes.numpy()[0, 0, :, :]
    pred_mattes = pred_mattes * 255

    mask = np.zeros(scale_trimap.shape)
    mask[scale_trimap == 0] = 1
    mask[scale_trimap == 255] = 1
    mask = 1 - mask

    alpha = (1. - mask) * scale_trimap + mask * pred_mattes

    return alpha


@t.no_grad()
def inference_img_whole(model, img, trimap):
    h, w, c = img.shape
    new_h = min(320, h - (h % 32))
    new_w = min(320, w - (w % 32))

    scale_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scale_trimap = cv2.resize(trimap, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pred_mattes = inference_onece(model, scale_img, scale_trimap)

    origin_pred_mattes = cv2.resize(pred_mattes, (w, h), interpolation=cv2.INTER_LINEAR)
    return origin_pred_mattes


@t.no_grad()
def main():
    netG = nn.DataParallel(NetG()).cuda()
    netG.load_state_dict(t.load(MODEL_DIR, map_location=t.device('cpu')))
    netG.eval()

    img_root = 'examples/image/dandelion-1335575_1920_1.png' # '/home/zzl/dataset/Combined_Dataset/Test_set/Adobe-licensed_images/image/antique-honiton-lace-1182740_1920_0.png'
    trimap_root = 'examples/trimap/dandelion-1335575_1920_1.png' #  '/home/zzl/dataset/Combined_Dataset/Test_set/Adobe-licensed_images/trimaps/antique-honiton-lace-1182740_1920_0.png'

    image = cv2.imread(img_root)
    trimap = cv2.imread(trimap_root)[:, :, 0]

    pred_mattes = inference_img_whole(netG, image, trimap)

    pred_mattes = pred_mattes.astype(np.uint8)
    # pred_mattes[trimap == 255] = 255
    # pred_mattes[trimap == 0] = 0

    cv2.imwrite('result.png', pred_mattes)


@t.no_grad()
def alphamatting():
    netG = nn.DataParallel(NetG()).cuda()
    netG.load_state_dict(t.load(MODEL_DIR, map_location=t.device('cpu')))
    netG.eval()

    img_root = '/data0/zzl/dataset/matting/alphamatting/input_lowers'
    trimap_root = '/data0/zzl/dataset/matting/alphamatting/trimap_lowres'

    img_name = os.listdir(img_root)

    for name in tqdm.tqdm(img_name):
        for i in range(1, 4):
            trimap_floder = 'Trimap' + str(i)

            img = cv2.imread(os.path.join(img_root, name))
            trimap = cv2.imread(os.path.join(trimap_root, trimap_floder, name))[:, :, 0]

            pred_mattes = inference_img_whole(netG, img, trimap)

            pred_mattes = pred_mattes.astype(np.uint8)

            cv2.imwrite(trimap_floder+'/'+name, pred_mattes)


if __name__ == '__main__':
    main()

