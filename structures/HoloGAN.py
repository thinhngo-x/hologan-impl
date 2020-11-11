"""Structure of  HoloGAN."""

import sys

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils.module import ResBlock, BLClassifier, Projection, MLP
from utils import functional

from typing import List, Tuple


class Generator(nn.Module):
    def __init__(self, latent_vector_size: int, out_shape: Tuple):
        """Initialize.

        @param latent_vector_size (int) Shape of the latent vector z. Eg: 128
        @param out_shape (Tuple) Shape of the output. Eg: (3, 128, 128)
        """
        super(Generator, self).__init__()
        if out_shape != (3, 128, 128):
            print("HoloGenerator: Output shape should be (3, 128, 128)!")
            sys.exit()
        self.z_dim = latent_vector_size

        self.constant = nn.Parameter(torch.zeros((1, 512, 4, 4, 4)))
        self.trans_conv1 = ResBlock(512, 128, stride=2, norm_layer=nn.InstanceNorm3d, conv=functional.trans_conv_3d_pad)
        self.mlp1 = MLP([self.z_dim, 128 * 2])
        self.trans_conv2 = ResBlock(128, 64, stride=2, norm_layer=nn.InstanceNorm3d, conv=functional.trans_conv_3d_pad)
        self.mlp2 = MLP([self.z_dim, 64 * 2])
        self.conv_3d = ResBlock(64, 64, stride=1, norm_layer=nn.InstanceNorm3d, conv=nn.Conv3d)
        self.projection = Projection(64, 16, 1024)
        self.trans_conv3 = ResBlock(1024, 256, stride=2, norm_layer=nn.InstanceNorm2d, conv=functional.trans_conv_2d_pad)
        self.mlp3 = MLP([self.z_dim, 256 * 2])
        self.trans_conv4 = ResBlock(256, 64, stride=2, norm_layer=nn.InstanceNorm2d, conv=functional.trans_conv_2d_pad)
        self.mlp4 = MLP([self.z_dim, 64 * 2])
        self.trans_conv5 = ResBlock(64, 3, stride=2, norm_layer=nn.InstanceNorm2d, conv=functional.trans_conv_2d_pad)
        self.mlp5 = MLP([self.z_dim, 3 * 2])

    def forward(self, z, rot_matrix):
        """Forward.

        @param z (Tensor) Latent vector of shape (bs, self.z_dim).
        @param rot_matrix (Tensor) Rotation matrix of shape (bs, 3, 4).

        @returns out (Tensor) Generated images of shape (bs, 3, 128, 128).
        """
        bs, _ = z.shape
        x = self.trans_conv1(self.constant)
        style = self.mlp1(z).view(bs, 128, 2)  # Hard-code
        x = functional.adain_3d_(x, style)
        x = F.leaky_relu(x)

        x = self.trans_conv2(x)
        style = self.mlp2(z).view(bs, 64, 2)  # Hard-code
        x = functional.adain_3d_(x, style)
        x = F.leaky_relu(x)

        x = functional.rigid_transform_3d_(x, rot_matrix)
        x = self.conv_3d(x)
        x = F.leaky_relu(x)

        x = self.projection(x)

        x = self.trans_conv3(x)
        style = self.mlp3(z).view(bs, 256, 2)  # Hard-code
        x = functional.adain_2d_(x, style)
        x = F.leaky_relu(x)

        x = self.trans_conv4(x)
        style = self.mlp4(z).view(bs, 64, 2)  # Hard-code
        x = functional.adain_2d_(x, style)
        x = F.leaky_relu(x)

        x = self.trans_conv5(x)
        style = self.mlp5(z).view(bs, 3, 2)  # Hard-code
        x = functional.adain_2d_(x, style)
        out = F.leaky_relu(x)

        return out


class Discriminator(nn.Module):
    def __init__(self, latent_vector_size: int, in_shape: Tuple, spec_norm=None, norm_layer=nn.BatchNorm2d):
        """Initialize.

        @param latent_vector_size (int) Shape of the latent vector z. Eg: 128
        @param in_shape (Tuple) Shape of the input. Eg: (3, 128, 128)
        """
        super(Discriminator, self).__init__()
        if in_shape != (3, 128, 128):
            print("HoloDiscriminator: Input shape should be (3, 128, 128)!")
            sys.exit()
        self.z_dim = latent_vector_size
        if spec_norm is None:
            self.spec_norm = lambda x: x
        else:
            self.spec_norm = spec_norm

        self.conv1 = self.spec_norm(nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False))  # 64x64
        self.conv2 = ResBlock(64, 128, stride=2, spec_norm=spec_norm, norm_layer=norm_layer)  # 32x32
        self.norm2 = norm_layer(128)
        self.conv3 = ResBlock(128, 256, stride=2, spec_norm=spec_norm, norm_layer=norm_layer)  # 16x16
        self.norm3 = norm_layer(256)
        self.conv4 = ResBlock(256, 512, stride=2, spec_norm=spec_norm, norm_layer=norm_layer)  # 8x8
        self.norm4 = norm_layer(512)
        self.conv5 = ResBlock(512, 1024, stride=2, spec_norm=spec_norm, norm_layer=norm_layer)  # 4x4
        self.norm5 = norm_layer(1024)

        self.fc = BLClassifier(1024 * 16)

        self.style_classifier_128 = BLClassifier(128 * 2)
        self.style_classifier_256 = BLClassifier(256 * 2)
        self.style_classifier_512 = BLClassifier(512 * 2)
        self.style_classifier_1024 = BLClassifier(1024 * 2)

        self.reconstruct = nn.Linear(1024 * 16, self.z_dim)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """Forward.

        @param x (Tensor) A tensor of shape in_shape

        @returns d_gan (Tensor) A tensor of shape (bs) indicating the probability whether the image generated is real.
                 d_id (Tensor) A reconstructed latent vector of shape (bs, self.z_dim)
                 d_style (Tensor) A concatenated tensor indicating the probability whether those styles are real.
                                  Shape (bs, 4), which means styles from 4 layers.
        """
        bs = x.shape[0]
        x = self.conv1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        mean_std = functional.channel_wise_mean_std_2d(x)
        d_s1 = self.style_classifier_128(mean_std.view(bs, -1))
        x = F.leaky_relu(self.norm2(x))

        x = self.conv3(x)
        mean_std = functional.channel_wise_mean_std_2d(x)
        d_s2 = self.style_classifier_256(mean_std.view(bs, -1))
        x = F.leaky_relu(self.norm3(x))

        x = self.conv4(x)
        mean_std = functional.channel_wise_mean_std_2d(x)
        d_s3 = self.style_classifier_512(mean_std.view(bs, -1))
        x = F.leaky_relu(self.norm4(x))

        x = self.conv5(x)
        mean_std = functional.channel_wise_mean_std_2d(x)
        d_s4 = self.style_classifier_1024(mean_std.view(bs, -1))
        x = F.leaky_relu(self.norm5(x))

        x = x.view(bs, -1)

        d_gan = self.fc(x)
        d_style = torch.cat((d_s1, d_s2, d_s3, d_s4), dim=1)
        d_id = self.reconstruct(x)

        return d_gan, d_id, d_style
