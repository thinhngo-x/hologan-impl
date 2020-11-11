"""This provides useful layers for the architecture."""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from utils import functional

from typing import List


class AdaIN2d(nn.Module):
    """2D Adaptative instance-normalization."""

    def __init__(self):
        """Initialize."""
        super(AdaIN2d, self).__init__()

    def forward(self, x, s, eps=1e-8):
        """Return a normalized tensor controlled by style s.

        @param x (Tensor) Input of shape (batch_size, c, h, w)
        @param s (Tensor) Style of shape (batch_size, c, 2)
        """
        return functional.adain_2d_(x, s, eps=eps)


class AdaIN3d(nn.Module):
    """3D Adaptative instance-normalization."""

    def __init__(self):
        """Initialize."""
        super(AdaIN3d, self).__init__()

    def forward(self, x, s, eps=1e-8):
        """Return a normalized tensor controlled by style s.

        @param x (Tensor) Input of shape (batch_size, c, d, h, w)
        @param s (Tensor) Style of shape (batch_size, c, 2)
        """
        return functional.adain_3d_(x, s, eps=eps)


class Projection(nn.Module):
    """Projection unit."""

    def __init__(self, c_in: int, d_in: int, c_out: int):
        """Initialize.

        @param c_in (int) Number of channels of input
        @param d_in (int) Depth of input
        @param c_out (int) Number of channels of output
        """
        super(Projection, self).__init__()
        self.linear = nn.Linear(c_in * d_in, c_out)

    def forward(self, x):
        """Return the projection of 4D-tensor x onto 3D-tensor space.

        @param x (Tensor) Input of shape (batch_size, c_in, d, h, w)

        returns out (Tensor) Output of shape (batch_size, c_out, h, w)
        """
        bs, c, d, h, w = x.shape
        x = x.view(bs, c * d, h, w)
        x = x.transpose(1, 3)  # (batch_size, w, h, c_in*d)
        out = F.leaky_relu(self.linear(x))  # (batch_size, w, h, c_out)
        out = torch.transpose(out, 1, 3)  # (batch_size, c_out, h, w)

        return out


class RigidTransform3d(nn.Module):
    """Rigid-body transformer."""

    def __init__(self):
        """Initialization."""
        super(RigidTransform3d, self).__init__()

    def forward(self, x, theta):
        """Rotate a 3D object.

        @param x (Tensor) Input of shape (batch_size, c, d, h, w)
        @param thetas (Tensor) Matrix of rotation, of shape (batch_size, 3, 4)

        returns out (Tensor) Output of shape (batch_size, c, h, w)
        """

        return functional.rigid_transform_3d_(x, theta)


class MLP(nn.Module):
    """Multi-layer perceptron."""

    def __init__(self, n_features: List[int]):
        """Initialization.

        @param n_features (list[int]) A list of integers indicating number of features for
                                      each layer (including input layer)
        """
        super(MLP, self).__init__()
        modules = []
        for i in range(len(n_features) - 1):
            in_f = n_features[i]
            out_f = n_features[i + 1]
            modules.append(nn.Linear(in_f, out_f))
        self.mlp = nn.Sequential(*modules)

    def forward(self, x):
        """Forward.

        @param x (Tensor) Input of shape (batch_size, n_in) with n_in equals to n_features[0]
        """

        return self.mlp(x)


class ResBlock(nn.Module):
    """Residual network block."""

    def __init__(self, inplanes, planes, kernel=3, stride=1,
                 spec_norm=None, norm_layer=None, conv=nn.Conv2d):
        """Initialize.

        @param inplanes (int) Number of channels of input
        @param planes (int) Number of channels of output
        @param stride (int)
        @param spec_norm ()
        """
        super(ResBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if spec_norm is None:
            self.spec_norm = lambda x: x
        else:
            self.spec_norm = spec_norm

        self.layers = nn.Sequential(
            self.spec_norm(conv(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(inplace=True),
            norm_layer(planes),
            self.spec_norm(conv(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)),
            norm_layer(planes)
        )

        if stride != 1 or planes != inplanes:
            self.downsample = nn.Sequential(
                self.spec_norm(conv(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False))
            )
        else:
            self.downsample = nn.Identity()

        # self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        """Forward.

        @param x (Tensor) A tensor of shape (inplanes, h_in, w_in)

        @returns out (Tensor) A tensor of shape (inplanes, h_out, w_out)
        """
        out = self.layers(x)
        identity = self.downsample(x)

        out += identity
        # out = self.relu(out)

        return out


class BLClassifier(nn.Module):
    """Binary logistic classifier."""

    def __init__(self, n_features):
        super(BLClassifier, self).__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        out = self.linear(x)
        out = torch.sigmoid(out)

        return out
