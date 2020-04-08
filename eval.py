import torchvision.transforms as transforms
import numpy as np
import torch as t
import torch.nn as nn
from model.AlphaGAN import NetG
import os
import cv2
from PIL import Image
import tqdm
from utils.Tester import Tester

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

MODEL_DIR = '/data1/zzl/checkpoint/alphaGAN/netG/netG_best_sad_68.pth'


@t.no_grad()
def inference_onece(model, scale_img, scale_trimap):

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    scale_img_rgb = cv2.cvtColor(scale_img, cv2.COLOR_BGR2RGB)

    tensor_img = normalize(scale_img_rgb).unsqueeze(0)

    tensor_trimap = transforms.ToTensor()(Image.fromarray(scale_trimap)).unsqueeze(0)

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
    new_h = min(6400, h - (h % 32))
    new_w = min(6400, w - (w % 32))

    scale_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scale_trimap = cv2.resize(trimap, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pred_mattes = inference_onece(model, scale_img, scale_trimap)

    origin_pred_mattes = cv2.resize(pred_mattes, (w, h), interpolation=cv2.INTER_LINEAR)
    return origin_pred_mattes


@t.no_grad()
def main():
    netG = NetG(False).cuda()
    netG.load_state_dict(t.load(MODEL_DIR, map_location=t.device('cpu')))
    netG.eval()

    img_root = './examples/images'
    trimap_root = './examples/trimaps'
    save_root = './result'
    images = os.listdir(img_root)
    
    for img in images:
    
        image = cv2.imread(os.path.join(img_root, img))
        trimap = cv2.imread(os.path.join(trimap_root, img))[:, :, 0]

        pred_mattes = inference_img_whole(netG, image, trimap)

        pred_mattes = pred_mattes.astype(np.uint8)
        # pred_mattes[trimap == 255] = 255
        # pred_mattes[trimap == 0] = 0
        if not os.path.exists(save_root):
            os.mkdir(save_root)
        cv2.imwrite(os.path.join(save_root, img), pred_mattes)


@t.no_grad()
def alphamatting():
    netG = NetG(False).cuda()
    netG.load_state_dict(t.load(MODEL_DIR, map_location=t.device('cpu')))
    netG.eval()

    img_root = '/data1/zzl/dataset/alphamatting/input_lowers'
    trimap_root = '/data1/zzl/dataset/alphamatting/trimap_lowres'

    img_name = os.listdir(img_root)

    current_path = os.getcwd()

    for name in tqdm.tqdm(img_name):
        for i in range(1, 4):
            trimap_floder = 'Trimap' + str(i)

            img = cv2.imread(os.path.join(img_root, name))
            trimap = cv2.imread(os.path.join(trimap_root, trimap_floder, name))[:, :, 0]

            pred_mattes = inference_img_whole(netG, img, trimap)

            pred_mattes = pred_mattes.astype(np.uint8)

            save_path = os.path.join(current_path, trimap_floder)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            cv2.imwrite(os.path.join(save_path, name), pred_mattes)


@t.no_grad()
def adobe():
    netG = NetG().cuda()
    netG.load_state_dict(t.load(MODEL_DIR, map_location=t.device('cpu')))
    netG.eval()

    ROOT = '/home/zzl/dataset/Combined_Dataset/Test_set/Adobe-licensed_images'
    img_root = os.path.join(ROOT, 'image')
    trimap_root = os.path.join(ROOT, 'trimaps')

    img_names = sorted(os.listdir(img_root))

    out_root = '/home/zzl/result'

    for name in img_names:
        img_path = os.path.join(img_root, name)
        trimap_path = os.path.join(trimap_root, name)

        img = cv2.imread(img_path)
        trimap = cv2.imread(trimap_path)[:, :, 0]

        pred_mattes = inference_img_whole(netG, img, trimap)

        pred_mattes = pred_mattes.astype(np.uint8)
        cv2.imwrite(out_root + '/' + name, pred_mattes)

@t.no_grad()
def whole_adobe():
    netG = NetG(False).cuda()
    netG.load_state_dict(t.load(MODEL_DIR, map_location=t.device('cpu')))
    netG.eval()
    tester = Tester(net_G=netG  ,
                    test_root='/data1/zzl/dataset/Combined_Dataset/Test_set/Adobe-licensed_images',
                    device='cuda:0')
    test_result = tester.test()
    for k, v in test_result.items():
        print(k, v)



if __name__ == '__main__':
    alphamatting()

