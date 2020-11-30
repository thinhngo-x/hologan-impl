#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is for sanity check.

Usage:
    sanity_check.py prepare_data
    sanity_check.py AdaIN
    sanity_check.py Projection
    sanity_check.py mean_std_2d
    sanity_check.py BLClassifier
    sanity_check.py ResBlock
    sanity_check.py HoloDiscriminator
    sanity_check.py train_discriminator [--gpu]
    sanity_check.py RigidTransform3d
    sanity_check.py HoloGenerator
    sanity_check.py train_hologan [--gpu]
    sanity_check.py mlp

Options:
    --gpu   using GPU
"""
import sys
from datetime import datetime

from docopt import docopt
from tqdm import tqdm as Bar

from torchvision.utils import make_grid
import torch
from torch import nn, optim
from torch.nn.utils import spectral_norm
import numpy as np
import matplotlib.pyplot as plt

from main import prepare_data, train_one_epoch
from utils.module import AdaIN2d, AdaIN3d, Projection, MLP, BLClassifier, ResBlock
from structures.HoloGAN import Discriminator, Generator
from structures import HoloGAN
from utils import functional

from torch.utils.tensorboard import SummaryWriter


def init_layers(model):
    """Reinitialize the layer weights for sanity check."""
    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.5)
            if m.bias is not None:
                m.bias.data.fill_(0.1)
        elif type(m) in [nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d]:
            m.weight.data.fill_(0.5)
            if m.bias is not None:
                m.bias.data.zero_()
        elif type(m) in [nn.BatchNorm2d, ]:
            m.reset_parameters()
            m.eval()
            with torch.no_grad():
                m.weight.fill_(1.0)
                m.bias.zero_()
        elif type(m) == nn.Parameter:
            torch.nn.init.zeros_(m)
    with torch.no_grad():
        model.apply(init_weights)


def init_layers_serious(model):
    """Reinitialize the layer weights for sanity check."""
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)
        elif type(m) in [nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d]:
            torch.nn.init.kaiming_normal_(m.weight)
        elif type(m) == nn.Parameter:
            torch.nn.init.uniform_(m)
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


def sanity_check_for_ResBlock():
    """Test module ResBlock."""
    print("Running check for ResBlock...")
    a = torch.stack((torch.zeros(1, 8, 8), torch.ones(1, 8, 8)))
    c = a.clone()
    block = ResBlock(1, 2, spec_norm=None, stride=2)
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

    block = ResBlock(1, 1, spec_norm=spectral_norm, stride=2)
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

    a = torch.zeros((1, 512, 4, 4, 4))
    block = ResBlock(512, 128, stride=2, norm_layer=nn.InstanceNorm3d,
                     conv=functional.trans_conv_3d_pad)
    init_layers(block)
    truth = torch.zeros((1, 128, 8, 8, 8))
    res = block(a)
    if res.shape != truth.shape:
        print("Failed at checking shape of output!")
        print("Expected shape: ", truth.shape)
        print("But got: ", res.shape)
    elif torch.norm(res - truth) > 1e-2:
        print("Failed at checking value of output!")
        print("Expected: \n", truth)
        print("But got: \n", res)
    else:
        print("Passed test 3!")

    a = torch.zeros((1, 1024, 16, 16))
    block = ResBlock(1024, 256, stride=2, norm_layer=nn.InstanceNorm2d,
                     conv=functional.trans_conv_2d_pad)
    init_layers(block)
    truth = torch.zeros((1, 256, 32, 32))
    res = block(a)
    if res.shape != truth.shape:
        print("Failed at checking shape of output!")
        print("Expected shape: ", truth.shape)
        print("But got: ", res.shape)
    elif torch.norm(res - truth) > 1e-2:
        print("Failed at checking value of output!")
        print("Expected: \n", truth)
        print("But got: \n", res)
    else:
        print("Passed test 4!")


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

    netD = Discriminator(128, (3, 128, 128), spec_norm=spectral_norm,
                         norm_layer=nn.InstanceNorm2d).to(device)
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
    writer.add_graph(netD, images.to(device))
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


def sanity_check_for_rigid_transform_3d():
    """Test RigidTransform3d."""
    print("Running check for RigidTransform3d...")
    a = torch.Tensor(
        [[1, 1, 1],
         [0, 1, 0],
         [0, 0, 0]]
    )
    # print(a.shape)
    a = a.view(1, 1, 3, 3, 1)
    theta = torch.Tensor([90])
    theta = functional.get_matrix_rot_3d(theta, 'elevation')
    theta = theta.view(1, 3, 4)
    res = functional.rigid_transform_3d_(a, theta, align_corners=False, mode='bilinear')
    truth = torch.Tensor(
        [[1, 0, 0],
         [1, 1, 0],
         [1, 0, 0]]
    )
    truth = truth.view(1, 1, 3, 3, 1)
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

    a = a.view(1, 1, 3, 1, 3)
    theta = torch.Tensor([90])
    theta = functional.get_matrix_rot_3d(theta, 'azimuth')
    theta = theta.view(1, 3, 4)
    res = functional.rigid_transform_3d_(a, theta, align_corners=False, mode='bilinear')
    truth = torch.Tensor(
        [[1, 0, 0],
         [1, 1, 0],
         [1, 0, 0]]
    )
    truth = truth.view(1, 1, 3, 1, 3)
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

    a = a.view(1, 1, 1, 3, 3)
    theta = torch.Tensor([90])
    theta = functional.get_matrix_rot_3d(theta, 'z')
    theta = theta.view(1, 3, 4)
    res = functional.rigid_transform_3d_(a, theta, align_corners=False, mode='bilinear')
    truth = torch.Tensor(
        [[1, 0, 0],
         [1, 1, 0],
         [1, 0, 0]]
    )
    truth = truth.view(1, 1, 1, 3, 3)
    if res.shape != truth.shape:
        print("Failed at checking shape of output!")
        print("Expected shape: ", truth.shape)
        print("But got: ", res.shape)
    elif torch.norm(res - truth) > 1e-2:
        print("Failed at checking value of output!")
        print("Expected: \n", truth)
        print("But got: \n", res)
    else:
        print("Passed test 3!")


def sanity_check_for_HoloGenerator():
    """Test class Generator in HoloGAN.py."""
    print("Running check for HoloGenerator...")
    z = torch.zeros((3, 128))
    thetas = torch.Tensor([90, 15, 45])
    rot_matrix = functional.get_matrix_rot_3d(thetas, 'azimuth')
    g = Generator(128, (3, 128, 128))
    init_layers(g)
    res = g(z, rot_matrix)
    truth = torch.zeros((3, 3, 128, 128))
    if res.shape != truth.shape:
        print("Failed at checking shape of output!")
        print("Expected shape: ", truth.shape)
        print("But got: ", res.shape)
    elif torch.norm(res - truth) / (3 * 3 * 128 * 128) > 1e-3:
        print("Failed at checking value of output!")
        print("Expected: \n", truth)
        print("But got: \n", res)
        print("Norm: \n", torch.norm(res) / (3 * 3 * 128 * 128))
    else:
        print("Passed!")


def sanity_check_for_train_hologan(use_gpu=True):
    """Test training discriminator."""
    now = datetime.now()
    subsample = None

    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(use_gpu)

    hologan = HoloGAN.Net(128, (3, 128, 128)).to(device)
    init_layers_serious(hologan)
    criterion = HoloGAN.compute_loss
    dataloader = prepare_data(batch_size=8, subsample=subsample)
    optim_G = optim.Adam(hologan.G.parameters())
    optim_D = optim.Adam(hologan.D.parameters())

    # Setup tensorboard
    logdir = "logs/fit/" + now.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(logdir)
    it = iter(dataloader)
    images, _ = it.next()
    img_grid = make_grid(images)
    writer.add_image("sample_batch", img_grid)
    writer.close()

    for epoch in range(3):
        train_one_epoch(dataloader, hologan, criterion, optim_G, optim_D, device, writer, epoch=epoch, print_step=20)


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
    elif args['mlp']:
        sanity_check_for_MLP()
    elif args['mean_std_2d']:
        sanity_check_for_channel_wise_mean_std_2d()
    elif args['BLClassifier']:
        sanity_check_for_BLClassifier()
    elif args['ResBlock']:
        sanity_check_for_ResBlock()
    elif args['HoloDiscriminator']:
        sanity_check_for_HoloDiscriminator()
    elif args['train_discriminator']:
        if args['--gpu']:
            sanity_check_for_train_discriminator(use_gpu=True)
        else:
            sanity_check_for_train_discriminator()
    elif args['train_hologan']:
        if args['--gpu']:
            sanity_check_for_train_hologan(use_gpu=True)
        else:
            sanity_check_for_train_hologan()
    elif args['RigidTransform3d']:
        sanity_check_for_rigid_transform_3d()
    elif args['HoloGenerator']:
        sanity_check_for_HoloGenerator()
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
