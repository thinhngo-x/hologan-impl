"""This is for training."""
from pathlib import Path
import os

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


PATH_TO_DATA = Path("data/Cars")
BATCH_SIZE = 8
IMG_SIZE = (128, 128)


def prepare_data(path_to_data=PATH_TO_DATA, batch_size=BATCH_SIZE, img_size=IMG_SIZE, subsample=None):
    """Return a DataLoader of the dataset."""
    if subsample is not None:
        idx = np.arange(5000)
        np.random.shuffle(idx)
        sampler = SubsetRandomSampler(idx[:subsample])
    else:
        sampler = None
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    training_data = torchvision.datasets.ImageFolder(path_to_data, transform=transform)
    print("Length of data: ", len(training_data))

    training_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size,
                                                  sampler=sampler)
    return training_loader