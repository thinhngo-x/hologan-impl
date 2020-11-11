"""Helper functions."""

import sys

import torch
from torch import nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np


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


def matplotlib_imshow(img, one_channel=False):
    """Plot a torch Tensor as an image."""
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.to('cpu').numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def images_to_probs(net, images):
    """Generate output from a trained network and a list of sampled images."""
    out = net(images)
    return out


def plot_sample_img(img):
    """Plot a sample image."""
    fig = plt.figure()
    matplotlib_imshow(img)
    return fig


def plot_classes_preds(net, images):
    """Plot sampled images and predictions."""
    probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure()
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx])
        ax.set_title(" Real: {0:.2f}\nStyle: {1:.2f}".format(
            probs[0][idx].squeeze(),
            probs[2][idx].mean()
        ))
    return fig


def adain_2d_(x, s, eps=1e-8):
    """Return a normalized tensor controlled by style s.

    @param x (Tensor) Input of shape (batch_size, c, h, w)
    @param s (Tensor) Style of shape (batch_size, c, 2)

    @returns out (Tensor) Output of shape (batch_size, c, h, w)
    """
    m_x = torch.mean(x, dim=(2, 3), keepdim=True)
    std_x = torch.std(x, dim=(2, 3), keepdim=True, unbiased=False)  # unbiased or not?

    # print("Mean of x: ", m_x)
    # print("Std of x: ", std_x)
    s = s.unsqueeze(3)
    s = s.unsqueeze(4)
    # print(s.shape)
    # print(s[:, :, 1, :].shape)
    out = s[:, :, 0] * (x - m_x) / (std_x + eps) + s[:, :, 1]

    return out


def adain_3d_(x, s, eps=1e-8):
    """Return a normalized tensor controlled by style s.

    @param x (Tensor) Input of shape (batch_size, c, d, h, w)
    @param s (Tensor) Style of shape (batch_size, c, 2)

    @returns out (Tensor) Output of shape (batch_size, c, d, h, w)
    """
    dims = len(x.shape)
    m_x = torch.mean(x, dim=tuple(range(2, dims)), keepdim=True)
    std_x = torch.std(x, dim=tuple(range(2, dims)), keepdim=True, unbiased=False)  # unbiased or not?

    # print("Mean of x: ", m_x)
    # print("Std of x: ", std_x)
    s = s.unsqueeze(3)
    s = s.unsqueeze(4)
    s = s.unsqueeze(5)
    # print(s.shape)
    # print(s[:, :, 1, :].shape)
    out = s[:, :, 0] * (x - m_x) / (std_x + eps) + s[:, :, 1]

    return out


def get_matrix_rot_3d(theta, mode):
    """Get a rotation matrix given the angle.

    @param theta (Tensor) The angle degree (/180) tensor of shape (batch_size,)
    @param mode (string) Mode of the rotation
                         'elevation' rotate around x-axis (axis 1)
                         'azimuth' rotate around y-axis (axis 2)

    @returns out (Tensor) Rotation matrix of shape (batch_size, 3, 4)
    """
    bs = len(theta)
    out = torch.zeros((bs, 3, 4))
    theta = torch.deg2rad(theta)
    c = torch.cos(theta)
    s = torch.sin(theta)
    if mode == 'elevation':
        out[:, 0, 0] = 1
        out[:, 1, 1] = c
        out[:, 1, 2] = -s
        out[:, 2, 1] = s
        out[:, 2, 2] = c
    if mode == 'azimuth':
        out[:, 0, 0] = c
        out[:, 0, 2] = -s
        out[:, 2, 0] = s
        out[:, 2, 2] = c
        out[:, 1, 1] = 1
    return out


def rigid_transform_3d_(x, matrix_rot, align_corners=False, mode='bilinear', padding_mode='zeros'):
    """Rotate a 3D object.

    @param x (Tensor) Input of shape (batch_size, c, d, h, w)
    @param matrix_transform (Tensor) Matrix of rotation, of shape (batch_size, 3, 4)

    returns out (Tensor) Output of shape (batch_size, c, h, w)
    """
    # Generate a sampling grid given a batch of affine matrices
    grid = F.affine_grid(matrix_rot, x.size(), align_corners=align_corners)
    # Compute the output using the sampling grid
    out = F.grid_sample(x, grid, align_corners=align_corners, mode=mode, padding_mode=padding_mode)

    return out


def trans_conv_2d_pad(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
    """Construct a transposed convolution 2D with padding.

    The output shape exactly doubles the input shape when stride=2.
    """
    if stride == 1:
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
    elif stride == 2:
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=padding, output_padding=1, bias=bias)


def trans_conv_3d_pad(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
    """Construct a transposed convolution 3D with padding.

    The output shape exactly doubles the input shape when stride=2.
    """
    if stride == 1:
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
    elif stride == 2:
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=2, padding=padding, output_padding=1, bias=bias)
