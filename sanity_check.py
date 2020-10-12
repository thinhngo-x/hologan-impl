"""
This is for sanity check.

Usage:
    sanity_check.py prepare_data
    sanity_check.py AdaIN
    sanity_check.py Projection
"""
import sys

from docopt import docopt

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from main import prepare_data
from utils.module import AdaIN, Projection


def init_layers(model):
    """Reinitialize the layer weights for sanity check."""
    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.5)
            if m.bias is not None:
                m.bias.data.fill_(0.1)
    with torch.no_grad():
        model.apply(init_weights)


def sanity_check_for_prepare_data():
    """Test function prepare_data and print out a sample."""
    dataloader = prepare_data(batch_size=1)
    print("Length of dataloader: ", len(dataloader))
    sample = next(iter(dataloader))[0][0]
    print("Shape of the sample image: ", sample.shape)
    plt.figure()
    plt.imshow(np.transpose(sample.detach(), (1, 2, 0)))
    plt.show()
    print("Prepare data successful")


def sanity_check_for_AdaIN():
    """Test AdaIN module."""
    x = torch.Tensor(
        [[[[1, 1], [1, 1]],
          [[0, 1], [2, 3]]],
         [[[0, 0], [0, 0]],
          [[0, -3], [5, 6]]]]
    )
    # print("Shape x: ", x.shape)
    s = torch.Tensor(
        [[[0, 1],
          [1, 0]],
         [[0, 0.5],
          [0.5, 0]]]
    )
    # print("Shape s: ", s.shape)

    adain = AdaIN()
    res = adain(x, s)
    # print("Shape of result: ", res.shape)
    truth = torch.Tensor(
        [[[[1.0000, 1.0000],
           [1.0000, 1.0000]],

          [[-1.3416, -0.4472],
           [0.4472, 1.3416]]],


         [[[0.5000, 0.5000],
           [0.5000, 0.5000]],

          [[-0.2722, -0.6804],
           [0.4082, 0.5443]]]])
    if res.shape != truth.shape:
        print("Failed at checking shape of output!")
        print("Expected shape: ", truth.shape)
        print("But got: ", res.shape)
    elif torch.norm(res - truth) > 1e-4:
        print("Failed at checking value of output!")
        print("Expected: \n", truth)
        print("But got: \n", res)
    else:
        print("Passed!")


def sanity_check_for_projection():
    """Test Projection module."""
    # a is a Tensor of shape (bs=2, c=2, h=2, w=2, d=1)
    a = torch.Tensor([[[[[1],
                         [-2]],

                        [[3],
                         [-4]]],


                       [[[5],
                         [6]],

                        [[7],
                         [8]]]],



                      [[[[9],
                         [10]],

                        [[11],
                         [12]]],


                       [[[-13],
                         [14]],

                        [[-15],
                         [16]]]]])

    inp = a.clone()
    proj = Projection(2, 1, 4)
    init_layers(proj)
    res = proj(inp)

    truth = torch.Tensor([[[[3.1000, 2.1000],
                            [5.1000, 2.1000]],

                           [[3.1000, 2.1000],
                            [5.1000, 2.1000]],

                           [[3.1000, 2.1000],
                            [5.1000, 2.1000]],

                           [[3.1000, 2.1000],
                            [5.1000, 2.1000]]],


                          [[[-0.0190, 12.1000],
                            [-0.0190, 14.1000]],

                           [[-0.0190, 12.1000],
                            [-0.0190, 14.1000]],

                           [[-0.0190, 12.1000],
                            [-0.0190, 14.1000]],

                           [[-0.0190, 12.1000],
                            [-0.0190, 14.1000]]]])

    if not torch.equal(a, inp):
        print("The input has been changed!")
    elif res.shape != truth.shape:
        print("Failed at checking shape of output!")
        print("Expected shape: ", truth.shape)
        print("But got: ", res.shape)
    elif torch.norm(res - truth) > 1e-5:
        print("Failed at checking value of output!")
        print("Expected: \n", truth)
        print("But got: \n", res)
    else:
        print("Passed!")


# TODO: add sanity check for rigid transformer


def main():
    """Execute sanity check functions."""
    args = docopt(__doc__)

    seed = 224
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if args['prepare_data']:
        sanity_check_for_prepare_data()
    elif args['AdaIN']:
        sanity_check_for_AdaIN()
    elif args['Projection']:
        sanity_check_for_projection()
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
