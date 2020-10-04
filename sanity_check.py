"""
This is for sanity check.

Usage:
    sanity_check.py prepare_data
    sanity_check.py AdaIN
"""
import sys

from docopt import docopt

import torch
import numpy as np
import matplotlib.pyplot as plt

from main import prepare_data
from utils.module import AdaIN


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
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
