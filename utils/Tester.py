from data import AlphaGANTestDataLoader
import torch as t
import tqdm


class Config(object):
    def __init__(self):
        return


class Tester(object):
    def __init__(self, net_G, test_root, test_bf=4):
        self.net_G = net_G

        opt = Config()
        opt.dataroot = test_root
        opt.batch_size = test_bf

        self.data_loader = AlphaGANTestDataLoader(opt)
        self.mse = t.nn.MSELoss().cuda()
        self.sad = t.nn.L1Loss().cuda()

    @t.no_grad()
    def test(self, vis):

        dataset = self.data_loader.load_data()

        total_sad = 0
        total_mse = 0

        for i, data in tqdm.tqdm(enumerate(dataset)):
            real_img = data['I']
            tri_img = data['T']

            input_img = t.cat([real_img, tri_img], dim=1).cuda()
            real_alpha = data['A'].cuda()
            fake_alpha = self.net_G(input_img)

            vis.images(real_alpha.cpu().numpy(), win='real_alpha')
            vis.images(fake_alpha.cpu().numpy(), win='fake_alpha')

            test_sad = self.sad(fake_alpha, real_alpha).item()
            test_mse = self.mse(fake_alpha, real_alpha).item()

            total_sad += test_sad
            total_mse += test_mse

        average_sad = total_sad
        average_mse = total_mse

        return {'sad': average_sad, 'mse': average_mse}











