"""This provides useful layers for the architecture."""

import torch
from torch import nn
import numpy as np


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
        dims = len(x.shape)
        m_x = torch.mean(x, dim=tuple(range(2, dims)), keepdim=True)
        std_x = torch.std(x, dim=tuple(range(2, dims)), keepdim=True, unbiased=False)  # unbiased or not?

        # print("Mean of x: ", m_x)
        # print("Std of x: ", std_x)

        while(len(s.shape) < len(x.shape) + 1):
            s = s.unsqueeze(3)
            # print(s.shape)
        # print(s[:, :, 1, :].shape)
        x = s[:, :, 0] * (x - m_x) / (std_x + eps) + s[:, :, 1]

        return x


class Projection(nn.Module):
    """Projection unit."""
