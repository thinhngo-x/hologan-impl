"""Useful functions."""

import sys

import torch


def channel_wise_mean_std_2d(x, unbiased=False):
    """Compute channel-wise mean and standard deviation of a tensor image 2D.

    @param x (Tensor) A tensor of shape (bs, c, h, w)

    @returns out (Tensor) A tensor of shape (bs, c, 2)
    """
    if len(x.shape) != 4:
        print("Expected a tensor 4d!")
        sys.exit()

    std, mean = torch.std_mean(x, dim=(2, 3), unbiased=unbiased)
    out = torch.cat((mean, std), dim=1)

    return out