import torch as t
from torch import nn
import torchvision as tv
import functools
from torchnet.meter import AverageValueMeter
import tqdm
import numpy as np
from visualize import Visualizer
import math


# ASPP
class _AsppBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate):
        super(_AsppBlock, self).__init__()
        if dilation_rate == 1:
            self.atrous_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        else:
            self.atrous_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    dilation=dilation_rate, padding=dilation_rate)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, _input):

        x = self.atrous_conv(_input)
        x = self.bn(x)

        return self.relu(x)


# input batch x 2048 x 40 x 40
class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        
        self.aspp_1 = _AsppBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1, dilation_rate=1)
        self.aspp_6 = _AsppBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3, dilation_rate=6)
        self.aspp_12 = _AsppBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3, dilation_rate=12)
        self.aspp_18 = _AsppBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3, dilation_rate=18)

        self.image_pooling = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=5*out_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
    
        aspp1 = self.aspp_1(x)    # 256
        aspp6 = self.aspp_6(x)    # 256
        aspp12 = self.aspp_12(x)  # 256
        aspp18 = self.aspp_18(x)  # 256

        im_p = self.image_pooling(x) # 256

        aspp = [aspp1, aspp6, aspp12, aspp18, im_p]
        aspp = t.cat(aspp, dim=1)

        return self.conv1(aspp)


# atrous_ResNet
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if dilation != 1:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                                   padding=dilation, dilation=dilation, bias=False)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv_1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilation=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        downsample_stride = stride if dilation == 1 else 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=downsample_stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        skip_connection1 = x  # 320 x 320

        x = self.conv_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        skip_connection2 = x  # 160 x 160

        x = self.maxpool(x)

        x = self.layer1(x)
        skip_connection3 = x  # 80 x 80

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, skip_connection1, skip_connection2, skip_connection3


def resnet50():

    model = ResNet(Bottleneck, [3, 4, 6, 3])

    return model


# encoder
class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.resnet50 = resnet50()
        self.aspp = ASPP(in_channels=2048, out_channels=256)

        self._initialize_weights()

    def _initialize_weights(self):
        # init atrous_resnet50 with the pretrained resnet
        pretrained_resnet50 = tv.models.resnet50(pretrained=True)
        pretrained_dict = pretrained_resnet50.state_dict()

        atrous_resnet_dict = self.resnet50.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in atrous_resnet_dict}

        atrous_resnet_dict.update(pretrained_dict)

        self.resnet50.load_state_dict(atrous_resnet_dict)
        # init aspp
        for m in self.aspp.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x, skip_connection1, skip_connection2, skip_connection3 = self.resnet50(x)

        x = self.aspp(x)

        return x, skip_connection1, skip_connection2, skip_connection3


# decoder
class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        # bilinear
        self.bilinear = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256)
        )
        # output: 256 x 80 x 80

        self.skip_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=48, kernel_size=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2)

        )

        # deconv1_x
        self.deconv1_x = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256+48, out_channels=256, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        # output: 64 x 80 x 80

        # unpooling
        # well I do not  konw how to get the indices, which is requested by the MaxUnpool2d
        # Therefore, I use the ConvTranspose2d instead of MaxUnpool2d
        # nn.MaxUnpool2d(kernel_size=2, stride=2),
        self.unpooling = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        # output: 64 x 160 x 160

        self.skip_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )

        # deconv2_x
        self.deconv2_x = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64+32, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        # output: 32 x 320 x 320

        # deconv3_x
        self.deconv3_x = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32+4, out_channels=32, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        # output: 32 x 320 x 320

        # deconv4_x
        self.deconv4_x = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        # output: 1 x 320 x 320

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x, skip_connection1, skip_connection2, skip_connection3 = x
        x = self.bilinear(x)

        skip_connection3 = self.skip_3(skip_connection3)

        x = t.cat([x, skip_connection3], dim=1)
        x = self.deconv1_x(x)

        x = self.unpooling(x)
        skip_connection2 = self.skip_2(skip_connection2)
        x = t.cat([x, skip_connection2], dim=1)
        x = self.deconv2_x(x)

        x = t.cat([x, skip_connection1], dim=1)
        x = self.deconv3_x(x)
        x = self.deconv4_x(x)

        return x


# G
class NetG(nn.Module):

    def __init__(self):
        super(NetG, self).__init__()

        self.encoder = Encoder()
        # output 256 x 40 x 40
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=True):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        return self.model(input)


class AlphaGAN(object):
    def __init__(self, args):
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.gpu_mode = args.gpu_mode
        self.device = args.device
        self.lrG = args.lrG
        self.lrD = args.lrD
        self.com_loss = args.com_loss
        self.fine_tune = args.fine_tune
        self.visual = args.visual
        self.env = args.env
        self.d_every = args.d_every
        self.g_every = args.g_every

        if self.fine_tune:
            self.model_G = args.model
            self.model_D = args.model.replace('netG', 'netD')

        # network init
        self.G = NetG()
        if self.com_loss:
            self.D = NLayerDiscriminator(input_nc=4)
        else:
            self.D = NLayerDiscriminator(input_nc=2)

        print(self.G)
        print(self.D)

        if self.fine_tune:
            self.G.load_state_dict(t.load(self.model_G))
            self.D.load_state_dict(t.load(self.model_D))

        self.G_optimizer = t.optim.Adam(self.G.parameters(), lr=self.lrG)
        self.D_optimizer = t.optim.Adam(self.D.parameters(), lr=self.lrD)
        if self.gpu_mode:
            self.G.to(self.device)
            self.D.to(self.device)
            self.G_criterion = t.nn.SmoothL1Loss().to(self.device)
            self.D_criterion = t.nn.MSELoss().to(self.device)

        self.G_error_meter = AverageValueMeter()
        self.Alpha_loss_meter = AverageValueMeter()
        self.Com_loss_meter = AverageValueMeter()
        self.Adv_loss_meter = AverageValueMeter()
        self.D_error_meter = AverageValueMeter()

    def train(self, dataset):
        if self.visual:
            vis = Visualizer(self.env)

        for epoch in range(self.epoch):
            for ii, data in tqdm.tqdm(enumerate(dataset)):
                real_img = data['I']
                tri_img = data['T']

                if self.com_loss:
                    bg_img = data['B'].to(self.device)
                    fg_img = data['F'].to(self.device)

                # input to the G
                input_img = t.tensor(np.append(real_img.numpy(), tri_img.numpy(), axis=1)).to(self.device)

                # real_alpha
                real_alpha = data['A'].to(self.device)

                # vis.images(real_img.numpy()*0.5 + 0.5, win='input_real_img')
                # vis.images(real_alpha.cpu().numpy()*0.5 + 0.5, win='real_alpha')
                # vis.images(tri_img.numpy()*0.5 + 0.5, win='tri_map')

                # train D
                if ii % self.d_every == 0:
                    self.D_optimizer.zero_grad()

                    # real_img_d = input_img[:, 0:3, :, :]
                    tri_img_d = input_img[:, 3:4, :, :]

                    # 真正的alpha 交给判别器判断
                    if self.com_loss:
                        real_d = self.D(input_img)
                    else:
                        real_d = self.D(t.cat([real_alpha, tri_img_d], dim=1))

                    target_real_label = t.tensor(1.0)
                    target_real = target_real_label.expand_as(real_d).to(self.device)

                    loss_d_real = self.D_criterion(real_d, target_real)
                    #loss_d_real.backward()

                    # 生成器生成fake_alpha 交给判别器判断
                    fake_alpha = self.G(input_img)
                    if self.com_loss:
                        fake_img = fake_alpha*fg_img + (1 - fake_alpha) * bg_img
                        fake_d = self.D(t.cat([fake_img, tri_img_d], dim=1))
                    else:
                        fake_d = self.D(t.cat([fake_alpha, tri_img_d], dim=1))
                    target_fake_label = t.tensor(0.0)

                    target_fake = target_fake_label.expand_as(fake_d).to(self.device)

                    loss_d_fake = self.D_criterion(fake_d, target_fake)

                    loss_D = loss_d_real + loss_d_fake
                    loss_D.backward()
                    self.D_optimizer.step()
                    self.D_error_meter.add(loss_D.item())

                # train G
                if ii % self.g_every == 0:
                    self.G_optimizer.zero_grad()

                    real_img_g = input_img[:, 0:3, :, :]
                    tri_img_g = input_img[:, 3:4, :, :]

                    fake_alpha = self.G(input_img)
                    # fake_alpha 与 real_alpha的L1 loss
                    loss_g_alpha = self.G_criterion(fake_alpha, real_alpha)
                    loss_G = loss_g_alpha
                    self.Alpha_loss_meter.add(loss_g_alpha.item())

                    if self.com_loss:
                        fake_img = fake_alpha * fg_img + (1 - fake_alpha) * bg_img
                        loss_g_cmp = self.G_criterion(fake_img, real_img_g)

                        # 迷惑判别器
                        fake_d = self.D(t.cat([fake_img, tri_img_g], dim=1))
                        self.Com_loss_meter.add(loss_g_cmp.item())
                        loss_G = loss_G + loss_g_cmp

                    else:
                        fake_d = self.D(t.cat([fake_alpha, tri_img_g], dim=1))
                    target_fake = t.tensor(1.0).expand_as(fake_d).to(self.device)
                    loss_g_d = self.D_criterion(fake_d, target_fake)

                    self.Adv_loss_meter.add(loss_g_d.item())

                    loss_G = loss_G + loss_g_d

                    loss_G.backward()
                    self.G_optimizer.step()
                    self.G_error_meter.add(loss_G.item())

                if self.visual and ii % 20 == 0:
                    vis.plot('errord', self.D_error_meter.value()[0])
                    #vis.plot('errorg', self.G_error_meter.value()[0])
                    vis.plot('errorg', np.array([self.Adv_loss_meter.value()[0], self.Alpha_loss_meter.value()[0],
                                                 self.Com_loss_meter.value()[0]]), legend=['adv_loss', 'alpha_loss',
                                                                                           'com_loss'])

                    vis.images(tri_img.numpy()*0.5 + 0.5, win='tri_map')
                    vis.images(real_img.cpu().numpy() * 0.5 + 0.5, win='relate_real_input')
                    vis.images(real_alpha.cpu().numpy() * 0.5 + 0.5, win='relate_real_alpha')
                    vis.images(fake_alpha.detach().cpu().numpy(), win='fake_alpha')
                    if self.com_loss:
                        vis.images(fake_img.detach().cpu().numpy()*0.5 + 0.5, win='fake_img')
            self.G_error_meter.reset()
            self.D_error_meter.reset()

            self.Alpha_loss_meter.reset()
            self.Com_loss_meter.reset()
            self.Adv_loss_meter.reset()
            if epoch % 5 == 0:
                t.save(self.D.state_dict(), self.save_dir + '/netD' + '/netD_%s.pth' % epoch)
                t.save(self.G.state_dict(), self.save_dir + '/netG' + '/netG_%s.pth' % epoch)

        return






