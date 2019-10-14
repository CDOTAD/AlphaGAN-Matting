# AlphaGAN

![](https://img.shields.io/badge/python-3.6.5-brightgreen.svg) ![](https://img.shields.io/badge/pytorch-0.4.1-brightgreen.svg) ![](https://img.shields.io/badge/visdom-0.1.8.5-brightgreen.svg) ![](https://img.shields.io/badge/tqdm-4.28.1-brightgreen.svg) ![](https://img.shields.io/badge/opencv-3.3.1-brightgreen.svg)

This project is an unofficial implementation of [AlphaGAN: Generative adversarial networks for natural image matting](https://arxiv.org/pdf/1807.10088.pdf) published at the BMVC 2018. As for now, the result of my experiment is not as good as the paper's.

**This is a course project of mine and there may exists some mistakes in this project.**

# Dataset

## Adobe Deep Image Matting Dataset

Follow the [instruction](https://sites.google.com/view/deepimagematting) to contact the author for the dataset

You might need to follow the method mentioned in the **Deep Image Matting** to generate the trimap using the alpha mat.

The trimap are generated while the data are loaded.

```python
import numpy as np
import cv2 as cv

def generate_trimap(alpha):
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
```

The Dataset structure in my project

```Bash
Train
  ├── alpha  # the alpha ground-truth
  ├── fg     # the foreground image
  ├── input  # the real image composed by the fg & bg
MSCOCO
  ├── train2014 # the background image

```
# Differences from the original paper

- SyncBatchNorm instead of pytorch original BatchNorm when use multi GPU.

# Records

- Achieved **SAD=78.22** after 21 epoches. The method seems to be right and there is still lots of work to do.

# Acknowledgments

My code is inspired by:

- [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

- [pytorch-book](https://github.com/chenyuntc/pytorch-book) chapter7 generate anime head portrait with GAN

- [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)

- [pytorch-deep-image-matting](https://github.com/huochaitiantang/pytorch-deep-image-matting)

- [indexnet_matting](https://github.com/poppinace/indexnet_matting)
