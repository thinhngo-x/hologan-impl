"""Helper functions."""

import sys

import torch
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
    """Generate output froma trained network and a list of sampled images."""
    out = net(images)
    return out


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
