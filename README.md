# AlphaGAN
>课程项目要求做image matting，于是找到了一篇这样的论文做image matting。刚好也对GAN比较感兴趣

本项目是参考BMVC 2018的一篇论文
[AlphaGAN: Generative adversarial networks for natural image matting](https://www.baidu.com)的复现。目前还没有采用论文中的skip connection。效果也不是很好，或者说效果比较差吧╮(￣▽￣)╭


# Network architecture

## Generator
AlphaGAN matting 很大程度上借鉴了[Deep Image matting]()。几乎可以理解为AlphaGAN matting 将Deep Image matting中的深度网络拿来作为了GAN的generator，只是把encoder中的VGG换成了ResNet50，并把少部分卷积层换成了空洞卷积，以达到不减小feature map也可以增大感受野的目的。

## Discriminator

AlphaGAN matting 的discriminator采用PatchGAN。

# Requirement

- Python 3
- Pytorch 0.4

# Train & Test

等我先重构下代码再说这部分的事情╮(￣▽￣)╭



