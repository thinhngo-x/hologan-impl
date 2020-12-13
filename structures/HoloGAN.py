"""Structure of  HoloGAN."""

import sys

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils.module import ResBlock, BLClassifier, Projection, MLP
from utils import functional
from torch.nn.utils import spectral_norm

from typing import List, Tuple


def compute_loss(prediction, label, weights):
    """Compute loss of the model HoloGAN.

    @param prediction (Tuple) A tuple of 3 outputs (d_gan, d_id, d_style)
    @param label (Tuple) A tuple of 3 labels (lb_gan, lb_id, lb_style)
    @param weights (List) A list of 3 weights corresponding to 3 types of loss (loss_gan, loss_id, loss_style)

    @returns loss (List) A list of 3 losses (loss_gan, loss_id, loss_style)
    """
    # print(prediction)
    loss_gan = weights[0] * \
        F.binary_cross_entropy_with_logits(prediction[0], label[0])
    loss_id = weights[1] * F.mse_loss(prediction[1], label[1])
    loss_style = weights[2] * F.binary_cross_entropy_with_logits(
        prediction[2].view(-1), label[2].view(-1))

    return [loss_gan, loss_id, loss_style]


def gen_labels(batch_size: int,
               label: bool, device: torch.device, z: torch.Tensor):
    """Generate labels in training.

    @param batch_size (int)
    @param label (bool) True: Labels Real
                        False: Labels Fake
    @param device
    @param z (Tensor) Latent vector, of shape (bs, z_dim)

    @returns lb_gan (Tensor) GAN label of shape (bs, 1)
             lb_id (Tensor) = z
             lb_style (Tensor) Style label of shape (bs, 4)
    """
    lb = label * 1
    lb_gan = torch.full((batch_size, 1), lb, dtype=torch.float, device=device)
    lb_id = z
    lb_style = torch.full((batch_size, 4), lb,
                          dtype=torch.float, device=device)

    return lb_gan, lb_id, lb_style


class Net(nn.Module):
    def __init__(self, latent_vector_size: int,
                 img_shape: Tuple,
                 spec_norm=spectral_norm,
                 norm_layer=nn.InstanceNorm2d):
        """Initialize.
        """
        super(Net, self).__init__()
        self.G = Generator(latent_vector_size, img_shape)
        self.D = Discriminator(latent_vector_size, img_shape,
                               spec_norm=spec_norm, norm_layer=norm_layer)

    def forward(self, z, rot_matrix):
        """Forward.
        """
        out = self.G(z, rot_matrix)
        d_gan, d_id, d_style = self.D(out)

        return (d_gan, d_id, d_style)


class Generator(nn.Module):
    def __init__(self, latent_vector_size: int, out_shape: Tuple):
        """Initialize.

        @param latent_vector_size (int) Shape of the latent vector z. Eg: 128
        @param out_shape (Tuple) Shape of the output. Eg: (3, 128, 128)
        """
        super(Generator, self).__init__()
        # if out_shape != (3, 128, 128):
        #     print("HoloGenerator: Output shape should be (3, 128, 128)!")
        #     sys.exit()
        self.z_dim = latent_vector_size
        self.out_size = out_shape[-1]

        self.constant = nn.Parameter(torch.rand((1, 512, 4, 4, 4)) * 2 - 1)
        self.trans_conv1 = functional.trans_conv_3d_pad(
            512, 128, stride=2, bias=False)
        self.mlp1 = MLP([self.z_dim, 128 * 2])
        self.trans_conv2 = functional.trans_conv_3d_pad(
            128, 64, stride=2, bias=False)
        self.mlp2 = MLP([self.z_dim, 64 * 2])
        # Rigid-transformation
        self.conv_3d = nn.Sequential(
            nn.Conv3d(64, 64, 3, padding=1, bias=False),
            nn.Conv3d(64, 64, 3, padding=1, bias=False)
        )
        self.projection = Projection(64, 16, 1024)
        self.trans_conv3 = functional.trans_conv_2d_pad(
            1024, 256, stride=2, bias=False)
        self.mlp3 = MLP([self.z_dim, 256 * 2])
        self.trans_conv4 = functional.trans_conv_2d_pad(
            256, 64, stride=2, bias=False)
        self.mlp4 = MLP([self.z_dim, 64 * 2])

        if self.out_size == 64:
            self.conv_2d = nn.Conv2d(64, 3, 3, padding=1, bias=False)

        elif self.out_size == 128:
            self.trans_conv5 = functional.trans_conv_2d_pad(
                64, 32, stride=2, bias=False)
            self.mlp5 = MLP([self.z_dim, 32 * 2])
            self.conv_2d = nn.Conv2d(32, 3, 3, padding=1, bias=False)

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

        if self.out_size == 128:
            x = self.trans_conv5(x)
            style = self.mlp5(z).view(bs, 32, 2)  # Hard-code
            x = functional.adain_2d_(x, style)
            x = F.leaky_relu(x)

        x = self.conv_2d(x)
        out = torch.tanh(x)

        return out


class Discriminator(nn.Module):
    def __init__(self,
                 latent_vector_size: int,
                 in_shape: Tuple,
                 spec_norm=None, norm_layer=nn.InstanceNorm2d):
        """Initialize.

        @param latent_vector_size (int) Shape of the latent vector z. Eg: 128
        @param in_shape (Tuple) Shape of the input. Eg: (3, 128, 128)
        """
        super(Discriminator, self).__init__()
        # if in_shape != (3, 128, 128):
        #     print("HoloDiscriminator: Input shape should be (3, 128, 128)!")
        #     sys.exit()
        self.z_dim = latent_vector_size
        self.in_size = in_shape[-1]

        if spec_norm is None:
            self.spec_norm = lambda x: x
        else:
            self.spec_norm = spec_norm

        self.conv1 = self.spec_norm(
            nn.Conv2d(3, 64, kernel_size=3, stride=2,
                      padding=1, bias=False))  # in_size/2
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2,
                               padding=1, bias=False)  # in_size/4
        self.norm2 = norm_layer(128)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2,
                               padding=1, bias=False)  # in_size/8
        self.norm3 = norm_layer(256)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2,
                               padding=1, bias=False)  # in_size/16
        self.norm4 = norm_layer(512)
        self.conv5 = nn.Conv2d(512, 1024, 3, stride=2,
                               padding=1, bias=False)  # in_size/32
        self.norm5 = norm_layer(1024)

        self.fc = BLClassifier(
            1024 * (self.in_size // 32) * (self.in_size // 32))

        self.style_classifier_128 = BLClassifier(128 * 2)
        self.style_classifier_256 = BLClassifier(256 * 2)
        self.style_classifier_512 = BLClassifier(512 * 2)
        self.style_classifier_1024 = BLClassifier(1024 * 2)

        self.reconstruct = nn.Linear(
            1024 * (self.in_size // 32) * (self.in_size // 32),
            self.z_dim
        )

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """Forward.

        @param x (Tensor) A tensor of shape in_shape

        @returns d_gan (Tensor) A tensor of shape (bs) indicating the probability whether the image generated is real.
                 d_id (Tensor) A reconstructed latent vector of shape (bs, self.z_dim)
                 d_style (Tensor) A concatenated tensor indicating the probability whether those styles are real.
                                  Shape (bs, 5), which means styles from 4 layers and the input.
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

        # print(x.shape)
        d_gan = self.fc(x)
        d_style = torch.cat((d_s1, d_s2, d_s3, d_s4), dim=1)
        d_id = torch.tanh(self.reconstruct(x))

        return d_gan, d_id, d_style
