import argparse
import os
from data import AlphaGANDataLoader
from model.AlphaGAN import AlphaGAN


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataroot', type=str, default='/home/zzl/dataset/matting/Train', help='Training data root')
    parser.add_argument('--epoch', type=int, default=1000, help='The number of epochs to run (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=4, help='The size of batch (default: 4)')

    parser.add_argument('--save_dir', type=str, default='/home/zzl/model/alphaGAN/full_loss', help='Directory name to save the model')
    parser.add_argument('--gpu_mode', type=str2bool, nargs='?', default=True, help='Use gpu mode (default: True)')
    parser.add_argument('--device', type=str, default='cuda:0', help='The cuda device that to be used (defult: cuda:0)')

    parser.add_argument('--d_every', type=int, default=5, help='the frequency of training D (default: 5)')
    parser.add_argument('--g_every', type=int, default=1, help='the frequency of training G (default: 1)')
    
    parser.add_argument('--lrG', type=float, default=0.0002, help='The learning rate of G (default: 0.0002)')
    parser.add_argument('--lrD', type=float, default=0.0002, help='The learning rate of D (default: 0.0002)')
    parser.add_argument('--com_loss', type=bool, default=True, help='Whether to use com_loss (default: True)')

    parser.add_argument('--fine_tune', type=str2bool, nargs='?', default=False, help='Start fine tune (default: False)')
    parser.add_argument('--model', type=str, help='Directory to get model')

    parser.add_argument('--visual', type=str2bool, nargs='?', default=True, help='Whether to visualize the process (default: True)')
    parser.add_argument('--env', type=str, default='alphaGAN', help='The name of the visdom environment (default: alphaGAN)')

    return check_args(parser.parse_args())


def check_args(args):

    save_D_dir = os.path.join(args.save_dir, 'netD')
    save_G_dir = os.path.join(args.save_dir, 'netG')
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
        os.mkdir(save_D_dir)
        os.mkdir(save_G_dir)

    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args


def main():

    args = parse_args()
    print(args.gpu_mode)
    if args is None:
        exit()

    data_loader = AlphaGANDataLoader(args)
    dataset = data_loader.load_data()

    gan = AlphaGAN(args=args)
    gan.train(dataset)


if __name__ == '__main__':
    main()
    '''
    args = parse_args()
    print(args.gpu_mode)
    print(args.model)
    netG = args.model
    netD = args.model.replace('netG', 'netD')
    print(netG)
    print(netD)
    '''

