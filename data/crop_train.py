import os
import numpy as np
from PIL import Image
import tqdm
DATA_ROOT = '/data1/zzl/dataset/matting/TrainRS'
SAVE_ROOT = '/data1/zzl/dataset/matting/TrainRS_c'

TRI_ROOT = os.path.join(DATA_ROOT, 'trimap')
ALPHA_ROOT = os.path.join(DATA_ROOT, 'alpha')
B_ROOT = os.path.join(DATA_ROOT, 'bg')
F_ROOT = os.path.join(DATA_ROOT, 'fg')
I_ROOT = os.path.join(DATA_ROOT, 'input')


TRI_SAVE = os.path.join(SAVE_ROOT, 'trimap')
ALPHA_SAVE = os.path.join(SAVE_ROOT, 'alpha')
B_SAVE = os.path.join(SAVE_ROOT, 'bg')
F_SAVE = os.path.join(SAVE_ROOT, 'fg')
I_SAVE = os.path.join(SAVE_ROOT, 'input')

if not os.path.exists(TRI_SAVE):
    os.mkdir(TRI_SAVE)
    os.mkdir(ALPHA_SAVE)
    os.mkdir(B_SAVE)
    os.mkdir(F_SAVE)
    os.mkdir(I_SAVE)


def random_choice(trimap, crop_size=(320, 320)):
    crop_height, crop_width = crop_size

    (h, w) = trimap.size
    x = np.random.randint(int(crop_height/2), h - int(crop_height/2))
    y = np.random.randint(int(crop_width/2), w - int(crop_width/2))
    return x, y


def safe_crop(img, x, y, crop_size=(320, 320)):
    crop_height, crop_width = crop_size

    region = (x-crop_width//2, y - crop_height//2, x + crop_width//2, y + crop_height//2)
    crop_img = img.crop(region)

    crop_img = crop_img.resize((320, 320))

    return crop_img


tri_paths = sorted(os.listdir(TRI_ROOT))

for i, img_name in tqdm.tqdm(enumerate(tri_paths)):

    tri_img = Image.open(os.path.join(TRI_ROOT, img_name))
    alpha_img = Image.open(os.path.join(ALPHA_ROOT, img_name))
    b_img = Image.open(os.path.join(B_ROOT, img_name))
    f_img = Image.open(os.path.join(F_ROOT, img_name))
    i_img = Image.open(os.path.join(I_ROOT, img_name))

    for rand_index in range(4):
        (w, h) = tri_img.size
        if h <= 320 or w <= 320:
            break

        x, y = random_choice(tri_img)

        tri_img_new = safe_crop(tri_img, x, y)
        alpha_img_new = safe_crop(alpha_img, x, y)
        b_img_new = safe_crop(b_img, x, y)
        f_img_new = safe_crop(f_img, x, y)
        i_img_new = safe_crop(i_img, x, y)

        save_name = str(i * 7 + rand_index)

        tri_img_new.save(os.path.join(TRI_SAVE, save_name + '.jpg'), 'jpeg')
        alpha_img_new.save(os.path.join(ALPHA_SAVE, save_name + '.jpg'), 'jpeg')
        b_img_new.save(os.path.join(B_SAVE, save_name + '.jpg'), 'jpeg')
        f_img_new.save(os.path.join(F_SAVE, save_name + '.jpg'), 'jpeg')
        i_img_new.save(os.path.join(I_SAVE, save_name + '.jpg'), 'jpeg')

    for rand_index in range(4, 6):
        crop_size = (480, 480)

        (w, h) = tri_img.size
        if w <= 480 or h <= 480:
            break

        x, y = random_choice(tri_img, crop_size=crop_size)

        tri_img_new = safe_crop(tri_img, x, y, crop_size=crop_size)
        alpha_img_new = safe_crop(alpha_img, x, y, crop_size=crop_size)
        b_img_new = safe_crop(b_img, x, y, crop_size=crop_size)
        f_img_new = safe_crop(f_img, x, y, crop_size=crop_size)
        i_img_new = safe_crop(i_img, x, y, crop_size=crop_size)

        save_name = str(i * 7 + rand_index)

        tri_img_new.save(os.path.join(TRI_SAVE, save_name + '.jpg'), 'jpeg')
        alpha_img_new.save(os.path.join(ALPHA_SAVE, save_name + '.jpg'), 'jpeg')
        b_img_new.save(os.path.join(B_SAVE, save_name + '.jpg'), 'jpeg')
        f_img_new.save(os.path.join(F_SAVE, save_name + '.jpg'), 'jpeg')
        i_img_new.save(os.path.join(I_SAVE, save_name + '.jpg'), 'jpeg')

    for rand_index in range(6, 7):
        crop_size = (640, 640)

        (w, h) = tri_img.size
        if w <= 640 or h <= 640:
            break

        x, y = random_choice(tri_img, crop_size=crop_size)

        tri_img_new = safe_crop(tri_img, x, y, crop_size=crop_size)
        alpha_img_new = safe_crop(alpha_img, x, y, crop_size=crop_size)
        b_img_new = safe_crop(b_img, x, y, crop_size=crop_size)
        f_img_new = safe_crop(f_img, x, y, crop_size=crop_size)
        i_img_new = safe_crop(i_img, x, y, crop_size=crop_size)

        save_name = str(i * 7 + rand_index)

        tri_img_new.save(os.path.join(TRI_SAVE, save_name + '.jpg'), 'jpeg')
        alpha_img_new.save(os.path.join(ALPHA_SAVE, save_name + '.jpg'), 'jpeg')
        b_img_new.save(os.path.join(B_SAVE, save_name + '.jpg'), 'jpeg')
        f_img_new.save(os.path.join(F_SAVE, save_name + '.jpg'), 'jpeg')
        i_img_new.save(os.path.join(I_SAVE, save_name + '.jpg'), 'jpeg')




