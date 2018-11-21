import torch as t
from torch import nn
import torchvision as tv
import functools
from torchnet.meter import AverageValueMeter
import tqdm
import numpy as np
from visualize import Visualizer


class _AsppBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, dilation_rate):
        super(_AsppBlock, self).__init__()

        self.relu1 = nn.LeakyReLU(0.2, True)
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.relu2 = nn.LeakyReLU(0.2, True)
        self.conv_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                dilation=dilation_rate, padding=dilation_rate)

    def forward(self, _input):

        feature = super(_AsppBlock, self).forward(_input)

        return feature


# input batch x 2048 x 40 x 40
class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.aspp_6 = _AsppBlock(in_channels=out_channels, out_channels=out_channels, dilation_rate=6)
        # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=6, padding=6)
        self.aspp_12 = _AsppBlock(in_channels=2*out_channels, out_channels=out_channels, dilation_rate=12)
        # nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, dilation=12, padding=12)
        self.aspp_18 = _AsppBlock(in_channels=3*out_channels, out_channels=out_channels, dilation_rate=18)
        # nn.Conv2d(in_channels=768, out_channels=256, kernel_size=3, dilation=18, padding=18)
        '''
        # self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.aspp_6 = _AsppBlock(in_channels=in_channels, out_channels=out_channels, dilation_rate=6)
        # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=6, padding=6)
        self.aspp_12 = _AsppBlock(in_channels=in_channels, out_channels=out_channels, dilation_rate=12)
        # nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, dilation=12, padding=12)
        self.aspp_18 = _AsppBlock(in_channels=in_channels, out_channels=out_channels, dilation_rate=18)
        # nn.Conv2d(in_channels=768, out_channels=256, kernel_size=3, dilation=18, padding=18)
        '''
        self.image_pooling = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=4*out_channels, out_channels=out_channels, kernel_size=1),
            nn.ReLU(True)
        )

    def forward(self, input):

        x = self.conv1(input)

        aspp6 = self.aspp_6(x)
        x = t.cat((aspp6, x), dim=1)

        aspp12 = self.aspp_12(x)
        x = t.cat((aspp12, x), dim=1)

        aspp18 = self.aspp_18(x)
        x = t.cat((aspp18, x), dim=1)

        x = self.image_pooling(x)
        '''

        aspp6 = self.aspp_6(input)
        aspp12 = self.aspp_12(input)
        aspp18 = self.aspp_18(input)

        x = t.cat((aspp6, aspp12), dim=1)
        x = t.cat((x, aspp18), dim=1)

        x = self.image_pooling(x)
        '''
        return x


# G
class NetG(nn.Module):

    def __init__(self):
        super(NetG, self).__init__()

        resnet50 = tv.models.resnet50(pretrained=True)
        resnet50.conv1.in_channels = 1

        resnet50.layer3[0].conv2.stride = (1, 1)
        resnet50.layer3[0].conv2.dilation = (2, 2)
        resnet50.layer3[0].conv2.padding = (2, 2)
        resnet50.layer3[0].downsample[0].stride = (1, 1)

        resnet50.layer4[0].conv2.stride = (1, 1)
        resnet50.layer4[0].conv2.dilation = (2, 2)
        resnet50.layer4[0].conv2.padding = (2, 2)
        resnet50.layer4[0].downsample[0].stride = (1, 1)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            # resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,
            resnet50.layer2,
            resnet50.layer3,
            resnet50.layer4
        )
        # self.aspp = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)
        self.aspp = ASPP(in_channels=2048, out_channels=256)

        # output 256 x 40 x 40

        self.decoder = nn.Sequential(
            # bilinear
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(256),
            # output: 256 x 80 x 80

            # deconv1_x
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(64),
            # output: 64 x 80 x 80

            # unpooling
            # well I do not  konw how to get the indices, which is requested by the MaxUnpool2d
            # Therefore, I use the ConvTranspose2d instead of MaxUnpool2d
            # nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(64),
            # output: 64 x 160 x 160

            # deconv2_x
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(32),
            # output: 32 x 320 x 320

            # deconv3_x
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(32),
            # output: 32 x 320 x 320

            # deconv4_x
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid()
            # output: 1 x 320 x 320
        )

    def forward(self, input):
        x = self.encoder(input)
        x = self.aspp(x)
        return self.decoder(x)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
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
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

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
        self.model_G = args.model
        self.model_D = args.model.replace('netG', 'netD')

        # network init
        self.G = NetG()
        if self.com_loss:
            self.D = NLayerDiscriminator(input_nc=3)
        else:
            self.D = NLayerDiscriminator(input_nc=1)

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
        self.D_error_meter = AverageValueMeter()

    def train(self, dataset):
        vis = Visualizer('alphaGAN_train')

        for epoch in range(self.epoch):
            for ii, data in tqdm.tqdm(enumerate(dataset)):
                real_img = data['I'].to(self.device)
                tri_img = data['T']

                bg_img = data['B'].to(self.device)
                fg_img = data['F'].to(self.device)

                # input to the G
                input_img = t.tensor(np.append(real_img.cpu().numpy(), tri_img.numpy(), axis=1)).to(self.device)

                # real_alpha
                real_alpha = data['A'].to(self.device)

                # vis.images(real_img.numpy()*0.5 + 0.5, win='input_real_img')
                # vis.images(real_alpha.cpu().numpy()*0.5 + 0.5, win='real_alpha')
                # vis.images(tri_img.numpy()*0.5 + 0.5, win='tri_map')

                if ii % 5 == 0:
                    self.D_optimizer.zero_grad()

                    # 真正的alpha 交给判别器判断
                    if self.com_loss:
                        real_d = self.D(real_img)
                    else:
                        real_d = self.D(real_alpha)

                    target_real_label = t.tensor(1.0)
                    target_real = target_real_label.expand_as(real_d).to(self.device)

                    loss_d_real = self.D_criterion(real_d, target_real)
                    #loss_d_real.backward()

                    # 生成器生成fake_alpha 交给判别器判断
                    fake_alpha = self.G(input_img)
                    if self.com_loss:
                        fake_img = fake_alpha*fg_img + (1 - fake_alpha) * bg_img
                        fake_d = self.D(fake_img)
                    else:
                        fake_d = self.D(fake_alpha)
                    target_fake_label = t.tensor(0.0)

                    target_fake = target_fake_label.expand_as(fake_d).to(self.device)

                    loss_d_fake = self.D_criterion(fake_d, target_fake)

                    loss_D = (loss_d_real + loss_d_fake) * 0.5
                    loss_D.backward()
                    self.D_optimizer.step()
                    self.D_error_meter.add(loss_D.item())

                if ii % 1 == 0:
                    self.G_optimizer.zero_grad()

                    fake_alpha = self.G(input_img)
                    # fake_alpha 与 real_alpha的L1 loss
                    loss_g_alpha = self.G_criterion(fake_alpha, real_alpha)
                    loss_G = 0.8 * loss_g_alpha
                    if self.com_loss:
                        fake_img = fake_alpha * fg_img + (1 - fake_alpha) * bg_img
                        loss_g_cmp = self.G_criterion(fake_img, real_img)

                        # 迷惑判别器
                        fake_d = self.D(fake_img)

                        loss_G = loss_G + 0.2 * loss_g_cmp
                    else:
                        fake_d = self.D(fake_alpha)
                    target_fake = t.tensor(1.0).expand_as(fake_d).to(self.device)
                    loss_g_d = self.D_criterion(fake_d, target_fake)

                    loss_G = loss_G + 0.2 * loss_g_d

                    loss_G.backward()
                    self.G_optimizer.step()
                    self.G_error_meter.add(loss_G.item())

                if ii % 20 == 0:
                    vis.plot('errord', self.D_error_meter.value()[0])
                    vis.plot('errorg', self.G_error_meter.value()[0])
                    vis.images(tri_img.numpy()*0.5 + 0.5, win='tri_map')
                    vis.images(real_img.cpu().numpy() * 0.5 + 0.5, win='relate_real_input')
                    vis.images(real_alpha.cpu().numpy() * 0.5 + 0.5, win='relate_real_alpha')
                    vis.images(fake_alpha.detach().cpu().numpy(), win='fake_alpha')
                    if self.com_loss:
                        vis.images(fake_img.detach().cpu().numpy()*0.5 + 0.5, win='fake_img')
            self.G_error_meter.reset()
            self.D_error_meter.reset()
            if epoch % 5 == 0:
                t.save(self.D.state_dict(), '/home/zzl/model/alphaGAN/netD/' + self.save_dir + '/netD_%s.pth' % epoch)
                t.save(self.G.state_dict(), '/home/zzl/model/alphaGAN/netG/' + self.save_dir + '/netG_%s.pth' % epoch)

        return






