##Copyright 2017 Adobe Systems Inc.
##
##Licensed under the Apache License, Version 2.0 (the "License");
##you may not use this file except in compliance with the License.
##You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
##Unless required by applicable law or agreed to in writing, software
##distributed under the License is distributed on an "AS IS" BASIS,
##WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##See the License for the specific language governing permissions and
##limitations under the License.


##############################################################
# Set your paths here

# path to provided foreground images
fg_path = '/home/zzl/dataset/Combined_Dataset/Training_set/Adobe-licensed_images/fg/'
fg_path_other = '/home/zzl/dataset/Combined_Dataset/Training_set/Other/fg/'
# path to provided alpha mattes
a_path = '/home/zzl/dataset/Combined_Dataset/Training_set/Adobe-licensed_images/alpha/'
a_path_other = '/home/zzl/dataset/Combined_Dataset/Training_set/Other/alpha/'

# Path to background images (MSCOCO)
bg_path = '/data1/zzl/dataset/mscoco/val2014/'

# Path to folder where you want the composited images to go


# Path to trimap
#tri_path = '/home/zzl/dataset/Combined_Dataset/Test_set/Adobe-licensed_images/trimaps'

##############################################################

##############################################################
a_out = '/data1/zzl/dataset/matting/TrainRS/alpha'
bg_out = '/data1/zzl/dataset/matting/TrainRS/bg'
t_out = '/data1/zzl/dataset/matting/TrainRS/trimap'
fg_out = '/data1/zzl/dataset/matting/TrainRS/fg'
im_out = '/data1/zzl/dataset/matting/TrainRS/input'
#############################################################


from PIL import Image
import os
import math
import tqdm
import numpy as np
import cv2 as cv


def generate_trimap(alpha):
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    # fg = cv.erode(fg, kernel, iterations=np.random.randint(1, 3))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    unknown = cv.dilate(unknown, kernel, iterations=np.random.randint(1, 20))
    trimap = fg * 255 + (unknown - fg) * 128
    return trimap.astype(np.uint8)


def composite4(fg, bg, a, w, h):
    bbox = fg.getbbox()
    #bg = bg.crop((0, 0, w, h))

    fg_list = fg.load()
    bg_list = bg.load()
    a_list = a.load()

    for y in range(h):
        for x in range(w):
            alpha = a_list[x, y] / 255
            t = fg_list[x, y][0]
            t2 = bg_list[x, y][0]
            if alpha >= 1:
                r = int(fg_list[x, y][0])
                g = int(fg_list[x, y][1])
                b = int(fg_list[x, y][2])
                bg_list[x, y] = (r, g, b, 255)
            elif alpha > 0:
                r = int(alpha * fg_list[x, y][0] + (1 - alpha) * bg_list[x, y][0])
                g = int(alpha * fg_list[x, y][1] + (1 - alpha) * bg_list[x, y][1])
                b = int(alpha * fg_list[x, y][2] + (1 - alpha) * bg_list[x, y][2])
                bg_list[x, y] = (r, g, b, 255)

    return bg


if not os.path.exists(a_out):
    os.mkdir(a_out)
    os.mkdir(bg_out)
    os.mkdir(t_out)
    os.mkdir(fg_out)
    os.mkdir(im_out)


num_bgs = 10

L = len(os.listdir(fg_path))

fg_files = sorted(os.listdir(fg_path)) + sorted(os.listdir(fg_path_other))
a_files = sorted(os.listdir(a_path)) + sorted(os.listdir(a_path_other))
#t_files = sorted(os.listdir(tri_path))
bg_files = os.listdir(bg_path)

bg_iter = iter(bg_files)
for a_index, im_name in enumerate(fg_files):
    if a_index >= L:
        fg_path = fg_path_other
        a_path = a_path_other
    im = Image.open(fg_path + im_name)
    a = Image.open(a_path + im_name).convert('L')
    #print(a.mode)
    bbox = im.size
    w = bbox[0]
    h = bbox[1]

    if im.mode != 'RGB':
        im = im.convert('RGB')

    bcount = 0
    for i in tqdm.tqdm(range(num_bgs)):

        train_num = a_index*num_bgs + i

        tri = Image.fromarray(generate_trimap(a))

        bg_name = next(bg_iter)
        bg = Image.open(bg_path + bg_name)
        if bg.mode != 'RGB':
            bg = bg.convert('RGB')

        bg_bbox = bg.size
        bw = bg_bbox[0]
        bh = bg_bbox[1]
        wratio = w / bw
        hratio = h / bh
        ratio = wratio if wratio > hratio else hratio
        if ratio > 1:
            bg = bg.resize((math.ceil(bw * ratio), math.ceil(bh * ratio)), Image.BICUBIC)

        bg = bg.crop((0, 0, w, h))
        bg.save(os.path.join(bg_out, str(train_num) + '.jpg'), 'jpeg')
        out = composite4(im, bg, a, w, h)

        #out.save(out_path + im_name[:len(im_name) - 4] + '_' + str(bcount) + '.png', "PNG")
        # cv2.imwrite(os.path.join(im_out, str(train_num)+'.jpg'),out)
        out.save(os.path.join(im_out, str(train_num) + '.jpg'), "jpeg")
        # fg

        # cv2.imwrite(os.path.join(fg_out, str(train_num)+'.jpg'),im)
        im.save(os.path.join(fg_out, str(train_num) + '.jpg'), "jpeg")
        # alpha

        # cv2.imwrite(os.path.join(a_out, str(train_num)+'.jpg'),a)
        a.save(os.path.join(a_out, str(train_num) + '.jpg'), 'jpeg')
        # trimap
        tri.save(os.path.join(t_out, str(train_num) + '.jpg'), 'jpeg')
        # bg

        # cv2.imwrite(os.path.join(bg_out, str(train_num)+'.jpg'),bg)


        bcount += 1



