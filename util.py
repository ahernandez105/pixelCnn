import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

def init_argparser():
    args = argparse.ArgumentParser()
    args.add_argument('--pickle', type=str, required=False, default='data/mnist.pkl')
    args.add_argument('--epochs', type=int, required=False, default=10)
    args.add_argument('--lr', type=float, required=False, default=0.004)
    args.add_argument('--filters', type=int, required=False, default=64)
    args.add_argument('--dev', type=str, required=False, default='cpu')
    args.add_argument('--workers', type=int, required=False, default=0)
    args.add_argument('--layers', type=int, required=False, default=2)
    args.add_argument('--bsize', type=int, required=False, default=128)
    args.add_argument('--kernel_size', type=int, required=False, default=7)
    args.add_argument('--dist_size', type=int, required=False, default=2)
    args.add_argument('--nll_img_path', type=str, required=False, default='output/binary_nll.png')
    args.add_argument('--samples_img_path', type=str, required=False, default='output/binary_sample.png')
    args.add_argument('--n_samples', type=int, required=False, default=100)
    args.add_argument('--save_path', type=str, required=False, default='')
    args.add_argument(
        '--conv_class', type=str, required=False, 
        choices=['MaskedConv2dBinary', 'MaskedConv2dColor'], default='MaskedConv2dBinary')

    return args.parse_args()

def save_training_plot(train_losses, test_losses, title, fname):
    # borrowed from berkeley deepul sp 2020 git page
    # https://github.com/rll/deepul
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, train_losses, label='train loss')
    plt.plot(x_test, test_losses, label='test loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('NLL')
    plt.tight_layout()
    plt.savefig(fname)

def show_samples(samples, fname=None, nrow=10, title='Samples'):
    # borrowed from berkeley deepul sp 2020 git page
    # https://github.com/rll/deepul
    if samples.shape[1]==1:
        samples = samples.astype('float32') * 255
    else:
        samples = samples.astype('float32') / 3 * 255

    samples = torch.FloatTensor(samples) / 255
    grid_img = make_grid(samples, nrow=nrow)
    plt.figure()
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')

    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()
