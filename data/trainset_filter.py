import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import tqdm


data_root = '/data1/zzl/dataset/matting/TrainRS_c'

TRIMAP = os.path.join(data_root, 'trimap')
BG = os.path.join(data_root, 'bg')
FG = os.path.join(data_root, 'fg')
INPUT = os.path.join(data_root, 'input')
ALPHA = os.path.join(data_root, 'alpha')

# img_name = '14211.jpg'

img_names = os.listdir(TRIMAP)

rm_list = []

for img_name in tqdm.tqdm(img_names):
    img = Image.open(os.path.join(TRIMAP, img_name))
    l = len(np.unique(img))

    if l <= 2:

        rm_list.append(img_name)

for rm in tqdm.tqdm(rm_list):
    os.remove(os.path.join(TRIMAP, rm))
    os.remove(os.path.join(BG, rm))
    os.remove(os.path.join(FG, rm))
    os.remove(os.path.join(INPUT, rm))
    os.remove(os.path.join(ALPHA, rm))
