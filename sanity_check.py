"""
This is for sanity check.

Usage:
    sanity_check.py prepare_data
"""
import sys

from docopt import docopt

import torch
import numpy as np
import matplotlib.pyplot as plt

from main import prepare_data


def sanity_check_for_prepare_data():
    """Test function prepare_data and print out a sample."""
    dataloader = prepare_data(batch_size=1)
    print("Length of dataloader: ", len(dataloader))
    sample = next(iter(dataloader))[0][0]
    print("Shape of the sample image: ", sample.shape)
    plt.figure()
    plt.imshow(np.transpose(sample.detach(), (1, 2, 0)))
    plt.show()


def main():
    """Execute sanity check functions."""
    args = docopt(__doc__)

    seed = 224
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if args['prepare_data']:
        sanity_check_for_prepare_data()
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
