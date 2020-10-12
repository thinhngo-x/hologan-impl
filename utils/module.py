"""This provides useful layers for the architecture."""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def adain_(x, s, eps=1e-8):
    """Return a normalized tensor controlled by style s.

    @param x (Tensor) Input of shape (batch_size, c, h, w, (d))
    @param s (Tensor) Style of shape (batch_size, c, 2)

    @returns out (Tensor) Output of shape (batch_size, c, h, w, (d))
    """
    dims = len(x.shape)
    m_x = torch.mean(x, dim=tuple(range(2, dims)), keepdim=True)
    std_x = torch.std(x, dim=tuple(range(2, dims)), keepdim=True, unbiased=False)  # unbiased or not?

    # print("Mean of x: ", m_x)
    # print("Std of x: ", std_x)

    while(len(s.shape) < len(x.shape) + 1):
        s = s.unsqueeze(3)
        # print(s.shape)
    # print(s[:, :, 1, :].shape)
    out = s[:, :, 0] * (x - m_x) / (std_x + eps) + s[:, :, 1]

    return out


class AdaIN(nn.Module):
    """Adaptative instance-normalization."""

    def __init__(self):
        """Initialize."""
        super(AdaIN, self).__init__()

    def forward(self, x, s, eps=1e-8):
        """Return a normalized tensor controlled by style s.

        @param x (Tensor) Input of shape (batch_size, c, h, w, (d))
        @param s (Tensor) Style of shape (batch_size, c, 2)
        """
        return adain_(x, s, eps=eps)


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

        @param x (Tensor) Input of shape (batch_size, c_in, h, w, d)

        returns out (Tensor) Output of shape (batch_size, c_out, h, w)
        """
        x = torch.transpose(x, 1, 3)  # (batch_size, w, h, c_in, d)
        bs, w, h, c, d = x.shape
        x = x.view(bs, w, h, c * d)
        out = F.leaky_relu(self.linear(x))  # (batch_size, w, h, c_out)
        out = torch.transpose(out, 1, 3)

        return out


def rigid_transform_3d_(x, theta):
    """Rotate a 3D object.

        @param x (Tensor) Input of shape (batch_size, c, d, h, w)
        @param thetas (Tensor) Matrix of rotation, of shape (batch_size, 3, 4)

        returns out (Tensor) Output of shape (batch_size, c, h, w)
    """
    grid = F.affine_grid(theta, x.size())  # Generate a sampling grid given a batch of affine matrices
    out = F.grid_sample(x, grid)  # Compute the output using the sampling grid

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

        return rigid_transform_3d_(x, theta)

# TODO: Add basic blocks for the architecture
