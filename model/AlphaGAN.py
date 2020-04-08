import torch as t
from torch import nn
from .Encoder import Encoder
from .Decoder import Decoder
from .NLayerDiscriminator import NLayerDiscriminator
from visualize import Visualizer
import numpy as np
from utils.Tester import Tester
from .AlphaLoss import AlphaLoss
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from torchnet.meter import AverageValueMeter
import os

def BatchNormGroup(num_features):
    return nn.GroupNorm(num_channels=num_features, num_groups=32)

# G
class NetG(nn.Module):

    def __init__(self, sync_bn=True):
        super(NetG, self).__init__()

        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = BatchNormGroup

        # Encoder
        self.encoder = Encoder(BatchNorm)
        # output 256 x 40 x 40
        self.decoder = Decoder(BatchNorm)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


class AlphaGAN(object):
    def __init__(self, args):
        self.epoch = args.epoch
        self.warmup_step = args.warmup_step
        self.batch_size = args.batch_size
        self.save_model = args.save_model
        self.save_dir = args.save_dir
        self.gpu_mode = args.gpu_mode
        self.device = args.device
        self.lrG = args.lrG
        self.lrD = args.lrD
        self.fine_tune = args.fine_tune
        self.visual = args.visual
        self.env = args.env

        self.best_sad = 1000

        if self.fine_tune:
            self.model_G = args.model
            self.model_D = args.model.replace('netG', 'netD')

        if len(self.device.split(',')) > 1:
            self.sync_bn = False
        else:
            self.sync_bn = False

        # network init
        netG = NetG(self.sync_bn)

        netD = NLayerDiscriminator(input_nc=4, n_layers=2, norm_layer=SynchronizedBatchNorm2d)

        if self.gpu_mode:
            self.G = netG.cuda()
            self.D = netD.cuda()
            self.G_criterion = AlphaLoss().cuda()
            self.D_criterion = t.nn.MSELoss().cuda()
        else:
            self.G = netG
            self.D = netD
            self.G_criterion = AlphaLoss()
            self.D_criterion = t.nn.MSELoss()

        if self.fine_tune:
            self.G.load_state_dict(t.load(self.model_G, map_location=t.device('cpu')))

        self.G_optimizer = t.optim.Adam(self.G.parameters(), lr=self.lrG, weight_decay=0.0005)
        # self.G_optimizer_aspp = t.optim.Adam(self.G.module.aspp.parameters(), lr=1e-4, weight_decay=0.0005)
        # self.G_optimizer_decoder = t.optim.Adam(self.G.module.decoder.parameters(), lr=1e-4, weight_decay=0.0005)
        self.D_optimizer = t.optim.Adam(self.D.parameters(), lr=self.lrD, weight_decay=0.0005)

        self.G_scheduler = lr_scheduler.CosineAnnealingLR(self.G_optimizer, T_max=self.epoch-self.warmup_step)
        self.D_scheduler = lr_scheduler.CosineAnnealingLR(self.D_optimizer, T_max=self.epoch-self.warmup_step)

        self.avg_G_loss = AverageValueMeter()
        self.avg_D_loss = AverageValueMeter()

        self.SAD_meter = AverageValueMeter()
        self.MSE_meter = AverageValueMeter()

    def train(self, dataset):

        print('---------netG------------')
        print(self.G)
        print('---------netD------------')
        print(self.D)

        writer = SummaryWriter('tensorboardlog/AlphaGAN_bs_1')
        niter = 1


        for epoch in range(1, self.epoch+1):

            cur_lr = self.adjust_learning_rate(epoch)
            writer.add_scalar('Train/lr', cur_lr, epoch)

            self.G.train()

            for ii, data in enumerate(dataset):
                #t.cuda.empty_cache()

                real_img = data['I']
                tri_img = data['T']
                bg_img = data['B']
                fg_img = data['F']

                # input to the G
                input_img = t.cat([real_img, tri_img], dim=1).cuda()

                # real_alpha
                real_alpha = data['A'].cuda()

                #####################################
                # train G
                #####################################
                self.set_requires_grad([self.D], False)
                self.G_optimizer.zero_grad()

                real_img_g = input_img[:, 0:3, :, :]
                tri_img_g = input_img[:, 3:4, :, :]

                # tri_img_original = tri_img_g * 0.5 + 0.5

                fake_alpha = self.G(input_img)

                wi = t.zeros(tri_img_g.shape)
                wi[(tri_img_g * 255) == 128] = 1.
                t_wi = wi.cuda()

                unknown_size = t_wi.sum()

                fake_alpha = (1 - t_wi) * tri_img_g + t_wi * fake_alpha

                # alpha loss
                loss_g_alpha = self.G_criterion(fake_alpha, real_alpha, unknown_size)

                # compositional loss
                comp = fake_alpha * fg_img.cuda() + (1. - fake_alpha) * bg_img.cuda()
                loss_g_com = self.G_criterion(comp, real_img_g, unknown_size) / 3.
                # self.Com_loss_meter.add(loss_g_com.item())

                '''
                vis.images(real_img.numpy() * 0.5 + 0.5, win='real_image', opts=dict(title='real_image'))
                vis.images(bg_img.numpy() * 0.5 + 0.5, win='bg_image', opts=dict(title='bg_image'))
                vis.images(fg_img.numpy() * 0.5 + 0.5, win='fg_image', opts=dict(title='fg_image'))
                vis.images(tri_img.numpy() * 0.5 + 0.5, win='trimap', opts=dict(title='trimap'))
                vis.images(real_alpha.detach().cpu().numpy() * 0.5 + 0.5, win='real_alpha', opts=dict(title='real_alpha'))
                vis.images(fake_alpha.detach().cpu().numpy(), win='fake_alpha', opts=dict(title='fake_alpha'))
                '''

                # trick D
                input_d = t.cat([comp, tri_img_g], dim=1)
                fake_d = self.D(input_d)

                target_fake = t.tensor(1.0).expand_as(fake_d).cuda()
                loss_g_d = self.D_criterion(fake_d, target_fake)

                loss_G = 0.5*loss_g_alpha + 0.5*loss_g_com + 0.001 * loss_g_d

                self.avg_G_loss.add(loss_G.item())

                loss_G.backward(retain_graph=True)
                self.G_optimizer.step()
                writer.add_scalar('Train/CompLoss', loss_g_com.item(), niter)
                writer.add_scalar('Train/AlphaLoss', loss_g_alpha.item(), niter)
                writer.add_scalar('Train/AdvLoss', loss_g_d.item(), niter)
                writer.add_scalar('Train/lossG', loss_G.item(), niter)

                #########################################
                # train D
                #########################################
                self.set_requires_grad([self.D], True)
                self.D_optimizer.zero_grad()

                # real [real_img, tri]
                real_d = self.D(input_img)

                target_real_label = t.tensor(1.0)
                target_real = target_real_label.expand_as(real_d).cuda()

                loss_d_real = self.D_criterion(real_d, target_real)

                # fake [fake_img, tri]
                fake_d = self.D(input_d)
                target_fake_label = t.tensor(0.0)

                target_fake = target_fake_label.expand_as(fake_d).cuda()
                loss_d_fake = self.D_criterion(fake_d, target_fake)

                loss_D = 0.5 * (loss_d_real + loss_d_fake)
                loss_D.backward()
                self.D_optimizer.step()
                self.avg_D_loss.add(loss_D.item())
                writer.add_scalar('Train/D_AdvLoss', loss_D.item(), niter)
                niter += 1
            writer.add_scalar('Train/G_avg_loss', self.avg_G_loss.value()[0], epoch)
            writer.add_scalar('Train/D_avg_loss', self.avg_D_loss.value()[0], epoch)
            self.avg_G_loss.reset()
            self.avg_D_loss.reset()
    
            ##############################
            # test
            ##############################
            
            if epoch % 1 == 0:
                print('-----test-------')
                tester = Tester(net_G=self.G.eval(),
                                test_root='/data1/zzl/dataset/Combined_Dataset/Test_set/Adobe-licensed_images')
                test_result = tester.test()
                if test_result['sad']<self.best_sad:
                    self.best_sad = test_result['sad']
                    t.save(self.G.state_dict(), self.save_dir+'/netG'+'/netG_best_sad.pth')
                print('sad : {0}, mse : {1}'.format(test_result['sad'], test_result['mse']))
                writer.add_scalars('Test/metric', test_result, epoch)

        return

    def adjust_learning_rate(self, step):
        if step < self.warmup_step:
            cur_lr = self.warmup_lr(self.lrG, step, self.warmup_step)
            for param_group in self.G_optimizer.param_groups:
                param_group['lr']=cur_lr
            for param_group in self.D_optimizer.param_groups:
                param_group['lr']=cur_lr
        else:
            self.G_scheduler.step()
            self.D_scheduler.step()

            cur_lr = self.G_scheduler.get_lr()[0]
        
        return cur_lr

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def warmup_lr(self, init_lr, step, iter_num):
        return step/iter_num*init_lr