"""This is for training."""
from pathlib import Path
import os
from tqdm import tqdm as Bar
import argparse
from datetime import datetime

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch import optim, nn
from torchvision.utils import make_grid
from torch.nn.utils import spectral_norm
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter


from utils import functional
from structures import HoloGAN


PATH_TO_DATA = Path("data/Cars")
BATCH_SIZE = 8
IMG_SIZE = (128, 128)


def parse_arg():
    now = datetime.now()
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--log_name', type=str, default=now.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--angles', type=str, default='[70, -70, 180, -180, 30, -30]')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr_g', type=float, default=0.0002)
    parser.add_argument('--lr_d', type=float, default=0.0002)
    parser.add_argument('--seed', type=int, default=224)
    parser.add_argument('--gpu', type=bool, default=True)
    parser.add_argument('--norm_generator', type=str, default='InstanceNorm')
    parser.add_argument('--subsample', type=int, default=None)
    parser.add_argument('--print_step', type=int, default=20)
    parser.add_argument('--checkpoint_name', type=str, default='checkpoint.ptn')
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--z_dim', type=int, default=200)
    parser.add_argument('--weights_loss', type=str, default='[1, 1, 1]')
    args = parser.parse_args()
    return args


def prepare_data(path_to_data=PATH_TO_DATA, batch_size=BATCH_SIZE,
                 img_size=IMG_SIZE, subsample=None, shuffle=True):
    """Return a DataLoader of the dataset."""
    if subsample is not None:
        idx = np.arange(5000)
        np.random.shuffle(idx)
        sampler = SubsetRandomSampler(idx[:subsample])
        shuffle = False
    else:
        sampler = None
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    training_data = torchvision.datasets.ImageFolder(path_to_data, transform=transform)
    print("Length of data: ", len(training_data))

    training_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size,
                                                  sampler=sampler, shuffle=shuffle)
    return training_loader


def train_one_epoch(dataloader, model: HoloGAN.Net, criterion, optim_G, optim_D, device,
                    writer, epoch, angles, weights_loss, print_step=50, z_dim=128):
    """Train a model on the dataloader for one epoch."""
    model.train()
    running_loss = [.0, .0, .0]
    running_loss_style = .0
    running_loss_id = .0
    running_loss_gan = .0
    num_iter = len(dataloader)
    for i, (imgs, _) in enumerate(Bar(dataloader)):
        model.zero_grad()

        bs, c, h, w = imgs.shape
        z = torch.rand((bs, z_dim), device=device)
        z = z * 2 - 1

        thetas_azm = torch.rand(bs, device=device) - 0.5
        thetas_azm = thetas_azm * (angles[2] - angles[3]) / 2 + (angles[2] + angles[3]) / 2
        thetas_elv = torch.rand(bs, device=device) - 0.5
        thetas_elv = thetas_elv * (angles[0] - angles[1]) / 2 + (angles[0] + angles[1]) / 2
        thetas_z = torch.rand(bs, device=device) - 0.5
        thetas_z = thetas_z * (angles[4] - angles[5]) / 2 + (angles[4] + angles[5]) / 2

        rot_mat_z = functional.get_matrix_rot_3d(thetas_z, 'z')
        rot_mat_azm = functional.get_matrix_rot_3d(thetas_azm, 'azimuth')
        rot_mat = functional.get_matrix_rot_3d(thetas_elv, 'elevation')
        rot_mat[:, :3, :3] = torch.matmul(rot_mat[:, :3, :3], rot_mat_azm[:, :3, :3])
        rot_mat[:, :3, :3] = torch.matmul(rot_mat_z[:, :3, :3], rot_mat[:, :3, :3])
        rot_mat = rot_mat.to(device)

        imgs = imgs.to(device)

        ###############
        # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###############
        # Train with all-real batch
        model.D.zero_grad()

        real_out = model.D(imgs)
        labels = HoloGAN.gen_labels(bs, True, device, z)

        lossD_real = criterion(real_out, labels, weights_loss)
        sum(lossD_real).backward()
        lossD_real = [ls.mean().item() for ls in lossD_real]

        # Train with all-fake batch
        fake = model.G(z, rot_mat)

        fake_out = model.D(fake.detach())
        labels = HoloGAN.gen_labels(bs, False, device, z)

        lossD_fake = criterion(fake_out, labels, weights_loss)
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

        lossG = criterion(out, labels, weights_loss)
        sum(lossG).backward()
        lossG = [ls.mean().item() for ls in lossG]
        optim_G.step()

        running_loss = [
            rl + ls for rl, ls in zip(running_loss, [sum(lossD_real), sum(lossD_fake), sum(lossG)])
        ]
        running_loss_gan += lossG[0]
        running_loss_id += lossG[1]
        running_loss_style += lossG[2]
        if i % print_step == print_step - 1:
            # print(i)
            step = epoch * num_iter + i + 1
            writer.add_scalar("lossD/real", running_loss[0] / print_step, step)
            writer.add_scalar("lossD/fake", running_loss[1] / print_step, step)
            writer.add_scalar("lossG/lossG", running_loss[2] / print_step, step)
            writer.add_scalar("lossG/GAN", running_loss_gan / print_step, step)
            writer.add_scalar("lossG/Id", running_loss_id / print_step, step)
            writer.add_scalar("lossG/Style", running_loss_style / print_step, step)
            writer.add_figure("sample_image", functional.plot_sample_img(fake.detach()[0]),
                              global_step=epoch * num_iter + i + 1)
            img_grid = make_grid(imgs)
            writer.add_image("sample_batch", img_grid, epoch * num_iter + i + 1)
            running_loss = [.0, .0, .0]
            running_loss_gan = .0
            running_loss_id = .0
            running_loss_style = .0


def save_checkpoint(optim_G, optim_D, model, epoch, name):
    with open(Path('checkpoints/' + name), 'wb') as f:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optim_G': optim_G.state_dict(),
            'optim_D': optim_D.state_dict(),
            'epoch': epoch
        }, f)


def load_checkpoint(optim_G, optim_D, model, checkpoint_path):
    with open(Path(checkpoint_path), 'rb') as f:
        checkpoint = torch.load(f)
    model.load_state_dict(checkpoint['model_state_dict'])

    optim_G.load_state_dict(checkpoint['optim_G'])
    optim_D.load_state_dict(checkpoint['optim_D'])
    if 'epoch' in checkpoint.keys():
        epoch = checkpoint['epoch'] + 1
    else:
        epoch = None

    return optim_G, optim_D, model, epoch


def main():
    args = parse_arg()
    args = vars(args)

    torch.manual_seed(args['seed'])

    if args['gpu'] and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    hologan = HoloGAN.Net(args['z_dim'], (3, 128, 128)).to(device)

    criterion = HoloGAN.compute_loss
    dataloader = prepare_data(batch_size=args['batch_size'], subsample=args['subsample'])
    optim_G = optim.Adam(hologan.G.parameters(), lr=args['lr_g'], betas=(0.5, 0.999))
    optim_D = optim.Adam(hologan.D.parameters(), lr=args['lr_d'], betas=(0.5, 0.999))

    if args['resume'] > 0:
        optim_G, optim_D, hologan, start_epoch = load_checkpoint(optim_G, optim_D,
                                                                 hologan, args['checkpoint_path'])
        if start_epoch is None:
            start_epoch = args['resume']
    else:
        start_epoch = 0

    # Setup tensorboard
    logdir = "logs/" + args['log_name']
    writer = SummaryWriter(Path(logdir))

    angles = eval(args['angles'])
    for epoch in range(start_epoch, args['num_epochs']):
        train_one_epoch(dataloader, hologan, criterion, optim_G, optim_D,
                        device, writer, epoch, angles, eval(args['weights_loss']),
                        z_dim=args['z_dim'], print_step=args['print_step'])
        save_checkpoint(optim_G, optim_D, hologan, epoch, args['checkpoint_name'])


if __name__ == '__main__':
    main()
