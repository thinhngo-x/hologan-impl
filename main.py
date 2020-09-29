"""This is for training."""
from pathlib import Path
import os

import torch
import torchvision
from torchvision import transforms


PATH_TO_DATA = Path("data/Cars")
BATCH_SIZE = 8
IMG_SIZE = (128, 128)


def prepare_data(path_to_data=PATH_TO_DATA, batch_size=BATCH_SIZE, img_size=IMG_SIZE):
    """Return a DataLoader of the dataset."""
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    training_data = torchvision.datasets.ImageFolder(path_to_data, transform=transform)
    print("Length of data: ", len(training_data))
    training_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    return training_loader