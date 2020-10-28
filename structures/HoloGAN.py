"""Structure of  HoloGAN."""

import sys

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils.module import ResBlock2d, BLClassifier
from utils import functional

from typing import List, Tuple


class HoloGenerator(nn.Module):
    def __init__(self, in_shape: Tuple, )


class HoloDiscriminator(nn.Module):
    def __init__(self, latent_vector_size: int, in_shape: Tuple, spec_norm=None):
        """Initialize.

        @param latent_vector_size (int) Shape of the latent vector z. Eg: 128
        @param in_shape (Tuple) Shape of the input. Eg: (3, 128, 128)
        """
        if in_shape != (3, 128, 128):
            print("Input shape does not match!")
            sys.exit()
        self.z_dim = latent_vector_size
        if spec_norm is None:
            self.spec_norm = lambda x: x
        else:
            self.spec_norm = spec_norm

        self.conv1 = self.spec_norm(nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False))  # 64x64
        self.conv2 = ResBlock2d(64, 128, spec_norm=spec_norm)  # 32x32
        self.conv3 = ResBlock2d(128, 256, spec_norm=spec_norm)  # 16x16
        self.conv4 = ResBlock2d(256, 512, spec_norm=spec_norm)  # 8x8
        self.conv5 = ResBlock2d(512, 1024, spec_norm=spec_norm)  # 4x4
        self.conv6 = nn.Conv2d(1024, 1024, kernel_size=4, stride=1, padding=0, bias=False)  # 1

        self.fc = nn.Linear(1024, 1)

        self.style_classifier_128 = BLClassifier(128)
        self.style_classifier_256 = BLClassifier(256)
        self.style_classifier_512 = BLClassifier(512)
        self.style_classifier_1024 = BLClassifier(1024)

        self.reconstruct = nn.Linear(1024, self.z_dim)

    def forward(self, x: torch.Tensor) -> (d_style: List[torch.Tensor], d_gan: torch.Tensor, d_id: torch.Tensor):
        """Forward.

        @param x (Tensor) A tensor of shape in_shape

        @returns d_gan (Tensor) A tensor of shape (bs) indicating the probability whether the image generated is real.
                 d_id (Tensor) A reconstructed latent vector of shape (bs, self.z_dim)
                 d_style (Tensor) A concatenated tensor indicating the probability whether those styles are real.
                                  Shape (bs, 4), which means styles from 4 layers.
        """
        bs = x.shape[0]
        x = self.conv1(x)

        x = self.conv2(x)
        mean_std = functional.channel_wise_mean_std_2d(x)
        d_s1 = self.style_classifier_128(mean_std.view(bs, -1))

        x = self.conv3(x)
        mean_std = functional.channel_wise_mean_std_2d(x)
        d_s2 = self.style_classifier_256(mean_std.view(bs, -1))

        x = self.conv4(x)
        mean_std = functional.channel_wise_mean_std_2d(x)
        d_s3 = self.style_classifier_512(mean_std.view(bs, -1))

        x = self.conv5(x)
        mean_std = functional.channel_wise_mean_std_2d(x)
        d_s4 = self.style_classifier_1024(mean_std.view(bs, -1))

        x = self.conv6(x).view(bs, -1)

        d_gan = self.fc(x)
        d_style = torch.cat((d_s1, d_s2, d_s3, d_s4), dim=1)
        d_id = self.reconstruct(x)

        return d_gan, d_id, d_style
