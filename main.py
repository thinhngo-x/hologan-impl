"""This is for training."""
from pathlib import Path
import os
from barbar import Bar

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam
from torch import nn

from utils import functional
from structures import HoloGAN


PATH_TO_DATA = Path("data/Cars")
BATCH_SIZE = 8
IMG_SIZE = (128, 128)


def prepare_data(path_to_data=PATH_TO_DATA, batch_size=BATCH_SIZE,
                 img_size=IMG_SIZE, subsample=None, shuffle=True):
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
                                                  sampler=sampler, shuffle=shuffle)
    return training_loader


def train_one_epoch(dataloader, model: HoloGAN.Net, criterion, optim_G, optim_D, device,
                    writer, epoch, print_step=50, z_norm=200, z_dim=128):
    """Train a model on the dataloader for one epoch."""
    running_loss = [.0, .0, .0]
    num_iter = len(dataloader)
    for i, (imgs, _) in enumerate(Bar(dataloader)):
        model.zero_grad()

        bs, c, h, w = imgs.shape
        z = torch.rand((bs, z_dim), device=device)
        z /= torch.norm(z) * z_norm
        
        thetas_azm = (torch.rand(bs, device=device) - 0.5) * 180
        thetas_elv = (torch.rand(bs, device=device) - 0.5) * 70

        rot_mat_azm = functional.get_matrix_rot_3d(thetas_azm, 'azimuth')
        rot_mat = functional.get_matrix_rot_3d(thetas_elv, 'elevation')
        rot_mat[:, :3, :3] = torch.matmul(rot_mat[:, :3, :3], rot_mat_azm[:, :3, :3])
        rot_mat = rot_mat.to(device)

        imgs = imgs.to(device)

        ###############
        # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###############
        # Train with all-real batch
        model.D.zero_grad()

        real_out = model.D(imgs)
        labels = HoloGAN.gen_labels(bs, True, device, z)

        lossD_real = criterion(real_out, labels)
        sum(lossD_real).backward()
        lossD_real = [ls.mean().item() for ls in lossD_real]

        # Train with all-fake batch
        fake = model.G(z, rot_mat)

        fake_out = model.D(fake.detach())
        labels = HoloGAN.gen_labels(bs, False, device, z)

        lossD_fake = criterion(fake_out, labels)
        sum(lossD_fake).backward()
        lossD_fake = [ls.mean().item() for ls in lossD_fake]

        lossD = [r + f for r, f in zip(lossD_real, lossD_fake)]
        optim_D.step()

        ###############
        # Update G network: maximize log(D(G(z)))
        ###############
        model.G.zero_grad()

        out = model.D(fake)
        labels = HoloGAN.gen_labels(bs, True, device, z)

        lossG = criterion(out, labels)
        sum(lossG).backward()
        lossG = [ls.mean().item() for ls in lossG]
        optim_G.step()

        running_loss = [
            rl + ls for rl, ls in zip(running_loss, [sum(lossD_real), sum(lossD_fake), sum(lossG)])
        ]
        if i % print_step == print_step - 1:
            # print(i)
            writer.add_scalar("lossD_real", running_loss[0] / print_step, epoch * num_iter + i)
            writer.add_scalar("lossD_fake", running_loss[1] / print_step, epoch * num_iter + i)
            writer.add_scalar("lossG", running_loss[2] / print_step, epoch * num_iter + i)
            writer.add_figure("sample_image", functional.plot_sample_img(fake.detach()[0]),
                              global_step=epoch * num_iter + i)
            running_loss = [.0, .0, .0]
