"""Structure of  HoloGAN."""

import sys

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils.module import ResBlock2d, BLClassifier
from utils import functional

from typing import List, Tuple


class Discriminator(nn.Module):
    def __init__(self, latent_vector_size: int, in_shape: Tuple, spec_norm=None, norm_layer=nn.BatchNorm2d):
        """Initialize.

        @param latent_vector_size (int) Shape of the latent vector z. Eg: 128
        @param in_shape (Tuple) Shape of the input. Eg: (3, 128, 128)
        """
        super(Discriminator, self).__init__()
        if in_shape != (3, 128, 128):
            print("HoloDiscriminator: Input shape does not match!")
            sys.exit()
        self.z_dim = latent_vector_size
        if spec_norm is None:
            self.spec_norm = lambda x: x
        else:
            self.spec_norm = spec_norm

        self.conv1 = self.spec_norm(nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False))  # 64x64
        self.conv2 = ResBlock2d(64, 128, stride=2, spec_norm=spec_norm, norm_layer=norm_layer)  # 32x32
        self.norm2 = norm_layer(128)
        self.conv3 = ResBlock2d(128, 256, stride=2, spec_norm=spec_norm, norm_layer=norm_layer)  # 16x16
        self.norm3 = norm_layer(256)
        self.conv4 = ResBlock2d(256, 512, stride=2, spec_norm=spec_norm, norm_layer=norm_layer)  # 8x8
        self.norm4 = norm_layer(512)
        self.conv5 = ResBlock2d(512, 1024, stride=2, spec_norm=spec_norm, norm_layer=norm_layer)  # 4x4
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
        x = self.norm2(x)

        x = self.conv3(x)
        mean_std = functional.channel_wise_mean_std_2d(x)
        d_s2 = self.style_classifier_256(mean_std.view(bs, -1))
        x = self.norm3(x)

        x = self.conv4(x)
        mean_std = functional.channel_wise_mean_std_2d(x)
        d_s3 = self.style_classifier_512(mean_std.view(bs, -1))
        x = self.norm4(x)

        x = self.conv5(x)
        mean_std = functional.channel_wise_mean_std_2d(x)
        d_s4 = self.style_classifier_1024(mean_std.view(bs, -1))
        x = self.norm5(x)

        x = x.view(bs, -1)

        d_gan = self.fc(x)
        d_style = torch.cat((d_s1, d_s2, d_s3, d_s4), dim=1)
        d_id = self.reconstruct(x)

        return d_gan, d_id, d_style
