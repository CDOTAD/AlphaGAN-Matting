import torch as t
from torch import nn
from .Encoder import Encoder
from .Decoder import Decoder
from .NLayerDiscriminator import NLayerDiscriminator
from visualize import Visualizer
import numpy as np
from torchnet.meter import AverageValueMeter
from utils.Tester import Tester


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


class AlphaGAN(object):
    def __init__(self, args):
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.save_model = args.save_model
        self.save_dir = args.save_dir
        self.gpu_mode = args.gpu_mode
        self.device = args.device
        self.lrG = args.lrG
        self.lrD = args.lrD
        self.com_loss = args.com_loss
        self.fine_tune = args.fine_tune
        self.visual = args.visual
        self.env = args.env

        if self.fine_tune:
            self.model_G = args.model
            self.model_D = args.model.replace('netG', 'netD')

        # network init
        netG = NetG()
        if self.com_loss:
            netD = NLayerDiscriminator(input_nc=4)
        else:
            netD = NLayerDiscriminator(input_nc=2)

        if self.gpu_mode:
            self.G = nn.DataParallel(netG).cuda()
            self.D = nn.DataParallel(netD).cuda()
            self.G_criterion = t.nn.SmoothL1Loss().cuda()
            self.D_criterion = t.nn.MSELoss().cuda()
        else:
            self.G = netG
            self.D = netD
            self.G_criterion = t.nn.SmoothL1Loss()
            self.D_criterion = t.nn.MSELoss()

        self.G_optimizer = t.optim.Adam(self.G.parameters(), lr=self.lrG, weight_decay=0.0005)
        self.D_optimizer = t.optim.Adam(self.D.parameters(), lr=self.lrD, weight_decay=0.0005)

        self.G_error_meter = AverageValueMeter()
        self.Alpha_loss_meter = AverageValueMeter()
        self.Com_loss_meter = AverageValueMeter()
        self.Adv_loss_meter = AverageValueMeter()
        self.D_error_meter = AverageValueMeter()

        self.SAD_meter = AverageValueMeter()
        self.MSE_meter = AverageValueMeter()

    def train(self, dataset):
        if self.visual:
            vis = Visualizer(self.env)

        for epoch in range(1, self.epoch):

            self.G.train()
            self.D.train()
            for ii, data in enumerate(dataset):

                real_img = data['I']
                tri_img = data['T']

                if self.com_loss:
                    bg_img = data['B'].cuda()
                    fg_img = data['F'].cuda()

                # input to the G
                input_img = t.cat([real_img, tri_img], dim=1).cuda()

                # real_alpha
                real_alpha = data['A'].cuda()

                # train G
                self.set_requires_grad([self.D], False)
                self.G_optimizer.zero_grad()

                real_img_g = input_img[:, 0:3, :, :]
                tri_img_g = input_img[:, 3:4, :, :]

                fake_alpha = self.G(input_img)
                # fake_alpha 与 real_alpha的L1 loss
                loss_g_alpha = self.G_criterion(fake_alpha, real_alpha)
                loss_G = loss_g_alpha / self.batch_size
                self.Alpha_loss_meter.add(loss_g_alpha.item())

                if self.com_loss:
                    fake_img = fake_alpha * fg_img + (1 - fake_alpha) * bg_img
                    loss_g_cmp = self.G_criterion(fake_img, real_img_g) / self.batch_size

                    # 迷惑判别器
                    fake_d = self.D(t.cat([fake_img, tri_img_g], dim=1))
                    self.Com_loss_meter.add(loss_g_cmp.item())
                    loss_G = loss_G + loss_g_cmp
                else:
                    fake_d = self.D(t.cat([fake_alpha, tri_img_g], dim=1))
                target_fake = t.tensor(1.0).expand_as(fake_d).cuda()
                loss_g_d = self.D_criterion(fake_d, target_fake) / self.batch_size

                self.Adv_loss_meter.add(loss_g_d.item())

                loss_G = loss_G + loss_g_d

                loss_G.backward()
                self.G_optimizer.step()
                self.G_error_meter.add(loss_G.item())

                # train D
                self.set_requires_grad([self.D], True)
                self.D_optimizer.zero_grad()

                # real_img_d = input_img[:, 0:3, :, :]
                tri_img_d = input_img[:, 3:4, :, :]

                # 真正的alpha 交给判别器判断
                if self.com_loss:
                    real_d = self.D(input_img)
                else:
                    real_d = self.D(t.cat([real_alpha, tri_img_d], dim=1))

                target_real_label = t.tensor(1.0)
                target_real = target_real_label.expand_as(real_d).cuda()

                loss_d_real = self.D_criterion(real_d, target_real) / self.batch_size

                # 生成器生成fake_alpha 交给判别器判断
                fake_alpha = self.G(input_img)
                if self.com_loss:
                    fake_img = fake_alpha*fg_img + (1 - fake_alpha) * bg_img
                    fake_d = self.D(t.cat([fake_img, tri_img_d], dim=1))
                else:
                    fake_d = self.D(t.cat([fake_alpha, tri_img_d], dim=1))
                target_fake_label = t.tensor(0.0)

                target_fake = target_fake_label.expand_as(fake_d).cuda()
                loss_d_fake = self.D_criterion(fake_d, target_fake) / self.batch_size

                loss_D = loss_d_real + loss_d_fake
                loss_D.backward()
                self.D_optimizer.step()
                self.D_error_meter.add(loss_D.item())

                '''
                if self.visual and ii % 20 == 0:
                    vis.plot('errord', self.D_error_meter.value()[0])
                    vis.plot('errorg', np.array([self.Adv_loss_meter.value()[0], self.Alpha_loss_meter.value()[0],
                                                 self.Com_loss_meter.value()[0]]), legend=['adv_loss', 'alpha_loss',
                                                                                           'com_loss'])

                    vis.images(tri_img.numpy()*0.5 + 0.5, win='tri_map')
                    vis.images(real_img.cpu().numpy() * 0.5 + 0.5, win='relate_real_input')
                    vis.images(real_alpha.cpu().numpy() * 0.5 + 0.5, win='relate_real_alpha')
                    vis.images(fake_alpha.detach().cpu().numpy(), win='fake_alpha')
                    if self.com_loss:
                        vis.images(fake_img.detach().cpu().numpy()*0.5 + 0.5, win='fake_img')
                '''

                vis.plot('errord', self.D_error_meter.value()[0])
                vis.plot('errorg', np.array([self.Adv_loss_meter.value()[0], self.Alpha_loss_meter.value()[0],
                                             self.Com_loss_meter.value()[0]]), legend=['adv_loss', 'alpha_loss',
                                                                                       'com_loss'])
            self.G.eval()
            tester = Tester(net_G=self.G, test_root='/data0/zzl/dataset/matting/Test')
            test_result = tester.test(vis)
            self.SAD_meter.add(test_result['sad'])
            self.MSE_meter.add(test_result['mse'])

            vis.plot('test_result', np.array([self.SAD_meter.value()[0], self.MSE_meter.value()[0]]),
                     legend=['SAD', 'MSE'])
            self.G_error_meter.reset()
            self.D_error_meter.reset()

            self.Alpha_loss_meter.reset()
            self.Com_loss_meter.reset()
            self.Adv_loss_meter.reset()
            if epoch % 5 == 0:
                if self.save_model:
                    t.save(self.D.state_dict(), self.save_dir + '/netD' + '/netD_%s.pth' % epoch)
                    t.save(self.G.state_dict(), self.save_dir + '/netG' + '/netG_%s.pth' % epoch)

        return

    def adjust_learning_rate(self):
        self.lrG = self.lrG / 10
        self.lrD = self.lrD / 10

        for param_group in self.G_optimizer.param_groups:
            param_group['lr'] = self.lrG

        for param_group in self.D_optimizer.param_groups:
            param_group['lr'] = self.lrD

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

