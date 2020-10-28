"""
This is for sanity check.

Usage:
    sanity_check.py prepare_data
    sanity_check.py AdaIN
    sanity_check.py Projection
    sanity_check.py mean_std_2d
    sanity_check.py BLClassifier
    sanity_check.py ResBlock2d
    sanity_check.py HoloDiscriminator
    sanity_check.py train_discriminator [--gpu]
    sanity_check.py MLP

--gpu   using GPU
"""
import sys
from datetime import datetime

from docopt import docopt
from barbar import Bar

from torchvision.utils import make_grid
import torch
from torch import nn, optim
from torch.nn.utils import spectral_norm
import numpy as np
import matplotlib.pyplot as plt

from main import prepare_data
from utils.module import AdaIN2d, AdaIN3d, Projection, MLP, BLClassifier, ResBlock2d
from structures.HoloGAN import Discriminator
from utils import functional

from torch.utils.tensorboard import SummaryWriter

# TODO: add sanity check for rigid transformer


def init_layers(model):
    """Reinitialize the layer weights for sanity check."""
    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.5)
            if m.bias is not None:
                m.bias.data.fill_(0.1)
        elif type(m) == nn.Conv2d:
            m.weight.data.fill_(0.5)
            if m.bias is not None:
                m.bias.data.zero_()
        elif type(m) == nn.BatchNorm2d:
            m.reset_parameters()
            m.eval()
            with torch.no_grad():
                m.weight.fill_(1.0)
                m.bias.zero_()
    with torch.no_grad():
        model.apply(init_weights)


def sanity_check_for_prepare_data():
    """Test function prepare_data and print out a sample."""
    print("Running check for function prepare_data()...")
    dataloader = prepare_data(batch_size=1)
    print("Length of dataloader: ", len(dataloader))
    sample = next(iter(dataloader))[0][0]
    print("Shape of the sample image: ", sample.shape)
    plt.figure()
    plt.imshow(np.transpose(sample.detach(), (1, 2, 0)))
    plt.show()
    print("Passed!")


def sanity_check_for_AdaIN():
    """Test AdaIN module."""
    print("Running check for AdaIN...")
    x = torch.Tensor(
        [[[[1, 1], [1, 1]],
          [[0, 1], [2, 3]]],
         [[[0, 0], [0, 0]],
          [[0, -3], [5, 6]]]]
    )
    # print("Shape x: ", x.shape)
    s = torch.Tensor(
        [[[0, 1],
          [1, 0]],
         [[0, 0.5],
          [0.5, 0]]]
    )
    # print("Shape s: ", s.shape)

    adain = AdaIN2d()
    res = adain(x, s)
    # print("Shape of result: ", res.shape)
    truth = torch.Tensor(
        [[[[1.0000, 1.0000],
           [1.0000, 1.0000]],

          [[-1.3416, -0.4472],
           [0.4472, 1.3416]]],


         [[[0.5000, 0.5000],
           [0.5000, 0.5000]],

          [[-0.2722, -0.6804],
           [0.4082, 0.5443]]]])
    if res.shape != truth.shape:
        print("Failed at checking shape of output!")
        print("Expected shape: ", truth.shape)
        print("But got: ", res.shape)
    elif torch.norm(res - truth) > 1e-4:
        print("Failed at checking value of output!")
        print("Expected: \n", truth)
        print("But got: \n", res)
    else:
        print("Passed!")


def sanity_check_for_projection():
    """Test Projection module."""
    # a is a Tensor of shape (bs=2, c=2, h=2, w=2, d=1)
    print("Running check for Projection...")
    a = torch.Tensor([[[[[1],
                         [-2]],

                        [[3],
                         [-4]]],


                       [[[5],
                         [6]],

                        [[7],
                         [8]]]],



                      [[[[9],
                         [10]],

                        [[11],
                         [12]]],


                       [[[-13],
                         [14]],

                        [[-15],
                         [16]]]]])
    a = a.transpose(2, 4)
    a = a.transpose(3, 4)  # (bs=2, c=2, d=1, h=2, w=2)

    inp = a.clone()
    proj = Projection(2, 1, 4)
    init_layers(proj)
    res = proj(inp)

    truth = torch.Tensor([[[[3.1000, 2.1000],
                            [5.1000, 2.1000]],

                           [[3.1000, 2.1000],
                            [5.1000, 2.1000]],

                           [[3.1000, 2.1000],
                            [5.1000, 2.1000]],

                           [[3.1000, 2.1000],
                            [5.1000, 2.1000]]],


                          [[[-0.0190, 12.1000],
                            [-0.0190, 14.1000]],

                           [[-0.0190, 12.1000],
                            [-0.0190, 14.1000]],

                           [[-0.0190, 12.1000],
                            [-0.0190, 14.1000]],

                           [[-0.0190, 12.1000],
                            [-0.0190, 14.1000]]]])

    if not torch.equal(a, inp):
        print("The input has been changed!")
    elif res.shape != truth.shape:
        print("Failed at checking shape of output!")
        print("Expected shape: ", truth.shape)
        print("But got: ", res.shape)
    elif torch.norm(res - truth) > 1e-5:
        print("Failed at checking value of output!")
        print("Expected: \n", truth)
        print("But got: \n", res)
    else:
        print("Passed!")


# def sanity_check_for_rigid_transform():
#     """Test Rigid transformation module."""
#     pass


def sanity_check_for_MLP():
    """Test MLP module."""
    print("Running check for MLP...")
    a = torch.Tensor([1, 2, 3, -1, -5, 0])

    mlp = MLP(list([6, 2]))
    init_layers(mlp)
    res = mlp(a)

    truth = torch.Tensor([0.1, 0.1])
    if res.shape != truth.shape:
        print("Failed at checking shape of output!")
        print("Expected shape: ", truth.shape)
        print("But got: ", res.shape)
    elif torch.norm(res - truth) > 1e-5:
        print("Failed at checking value of output!")
        print("Expected: \n", truth)
        print("But got: \n", res)
    else:
        print("Passed test 1!")

    a = torch.Tensor([1, 3, 3, -1, -5, 0])

    mlp = MLP(list([6, 3, 1]))
    init_layers(mlp)
    res = mlp(a)

    truth = torch.Tensor([1.0])
    if res.shape != truth.shape:
        print("Failed at checking shape of output!")
        print("Expected shape: ", truth.shape)
        print("But got: ", res.shape)
    elif torch.norm(res - truth) > 1e-5:
        print("Failed at checking value of output!")
        print("Expected: \n", truth)
        print("But got: \n", res)
    else:
        print("Passed test 2!")


def sanity_check_for_channel_wise_mean_std_2d():
    """Test function channel_wise_mean_std_2d."""
    print("Running check for function channel_wise_mean_std_2d()...")
    a = torch.arange(24, dtype=torch.float32).view(2, 3, 2, -1)
    res = functional.channel_wise_mean_std_2d(a)
    std = 1.118034
    truth = torch.Tensor([
        [1.5, 5.5, 9.5, std, std, std],
        [13.5, 17.5, 21.5, std, std, std]
    ])
    if res.shape != truth.shape:
        print("Failed at checking shape of output!")
        print("Expected shape: ", truth.shape)
        print("But got: ", res.shape)
    elif torch.norm(res - truth) > 1e-4:
        print("Failed at checking value of output!")
        print("Expected: \n", truth)
        print("But got: \n", res)
    else:
        print("Passed!")


def sanity_check_for_BLClassifier():
    """Test module BLClassifier."""
    print("Running check for BLClassifier...")
    a = torch.arange(16, dtype=torch.float32) / 100
    a = torch.stack((a, a * 2), dim=0)
    cls = BLClassifier(16)
    init_layers(cls)
    res = cls(a)
    truth = torch.Tensor([[0.66819], [0.78583]])
    if res.shape != truth.shape:
        print("Failed at checking shape of output!")
        print("Expected shape: ", truth.shape)
        print("But got: ", res.shape)
    elif torch.norm(res - truth) > 1e-4:
        print("Failed at checking value of output!")
        print("Expected: \n", truth)
        print("But got: \n", res)
    else:
        print("Passed!")


def sanity_check_for_ResBlock2d():
    """Test module ResBlock2d."""
    print("Running check for ResBlock2d...")
    a = torch.stack((torch.zeros(1, 8, 8), torch.ones(1, 8, 8)))
    c = a.clone()
    block = ResBlock2d(1, 2, spec_norm=None, stride=2)
    init_layers(block)
    res = block(c)
    truth = torch.stack(
        (torch.zeros(2, 4, 4),
         torch.stack([torch.Tensor([[13, 23, 23, 20.5],
                                    [23, 41, 41, 36.5],
                                    [23, 41, 41, 36.5],
                                    [20.5, 36.5, 36.5, 32.5]])] * 2))
    )
    if not torch.equal(a, c):
        print("The input has been changed!")
    if res.shape != truth.shape:
        print("Failed at checking shape of output!")
        print("Expected shape: ", truth.shape)
        print("But got: ", res.shape)
    elif torch.norm(res - truth) > 1e-2:
        print("Failed at checking value of output!")
        print("Expected: \n", truth)
        print("But got: \n", res)
    else:
        print("Passed test 1!")

    block = ResBlock2d(1, 1, spec_norm=spectral_norm, stride=2)
    init_layers(block)
    res = block(c)
    res = block.layers[3].weight
    truth = torch.ones(1, 1, 3, 3) * 0.33333
    if res.shape != truth.shape:
        print("Failed at checking shape of output!")
        print("Expected shape: ", truth.shape)
        print("But got: ", res.shape)
    elif torch.norm(res - truth) > 1e-2:
        print("Failed at checking value of output!")
        print("Expected: \n", truth)
        print("But got: \n", res)
    else:
        print("Passed test 2!")


def sanity_check_for_HoloDiscriminator():
    """Test module HoloDiscriminator."""
    print("Running check for HoloDiscriminator...")
    a = torch.zeros(2, 3, 128, 128)
    c = a.clone()
    d = Discriminator(10, (3, 128, 128), spec_norm=spectral_norm, norm_layer=nn.InstanceNorm2d)
    init_layers(d)
    ress = d(c)
    truths = (
        torch.zeros(2, 1) + 0.5250,
        torch.zeros(2, 10) + 0.1,
        torch.zeros(2, 4) + 0.5250
    )
    for (res, truth) in zip(ress, truths):
        if res.shape != truth.shape:
            print("Failed at checking shape of output!")
            print("Expected shape: ", truth.shape)
            print("But got: ", res.shape)
        elif torch.norm(res - truth) > 1e-2:
            print("Failed at checking value of output!")
            print("Expected: \n", truth)
            print("But got: \n", res)
    print("Passed!")


def sanity_check_for_train_discriminator(use_gpu=False):
    """Test training discriminator."""
    now = datetime.now()
    subsample = 80

    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(use_gpu)
    # z = torch.randn(128, device=device=None)

    netD = Discriminator(128, (3, 128, 128), spec_norm=spectral_norm, norm_layer=nn.InstanceNorm2d).to(device)
    criterion = nn.BCELoss()
    real_label = 1
    fake_label = 0
    opt = optim.Adam(netD.parameters())
    dataloader = prepare_data(batch_size=8, subsample=subsample)

    # Setup tensorboard
    logdir = "logs/fit/" + now.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(logdir)
    it = iter(dataloader)
    images, _ = it.next()
    img_grid = make_grid(images)
    writer.add_image("sample_batch", img_grid)
    writer.add_graph(netD, images)
    writer.close()

    running_loss = 0.0

    for epoc in range(10):
        print("\nEpoch: ", epoc + 1)

        for i_batch, data in enumerate(Bar(dataloader)):
            netD.zero_grad()
            real_img = data[0].to(device)
            bs = real_img.shape[0]
            d_gan, d_id, d_style = netD(real_img)
            label = torch.full(d_gan.shape, real_label, dtype=torch.float, device=device)
            errD_real = criterion(d_gan, label)
            label = torch.full(d_style.shape, real_label, dtype=torch.float, device=device)
            errD_real += criterion(d_style, label)
            errD_real.backward()
            opt.step()

            running_loss += errD_real.item()
            if i_batch % 10 == 9:
                writer.add_scalar("training loss", running_loss / 10, epoc * subsample + i_batch)
                writer.add_figure("predictions", functional.plot_classes_preds(netD, real_img))
                running_loss = 0.0


def main():
    """Execute sanity check functions."""
    args = docopt(__doc__)

    seed = 224
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if args['prepare_data']:
        sanity_check_for_prepare_data()
    elif args['AdaIN']:
        sanity_check_for_AdaIN()
    elif args['Projection']:
        sanity_check_for_projection()
    elif args['MLP']:
        sanity_check_for_MLP()
    elif args['mean_std_2d']:
        sanity_check_for_channel_wise_mean_std_2d()
    elif args['BLClassifier']:
        sanity_check_for_BLClassifier()
    elif args['ResBlock2d']:
        sanity_check_for_ResBlock2d()
    elif args['HoloDiscriminator']:
        sanity_check_for_HoloDiscriminator()
    elif args['train_discriminator']:
        if args['--gpu']:
            sanity_check_for_train_discriminator(use_gpu=True)
        else:
            sanity_check_for_train_discriminator()
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
