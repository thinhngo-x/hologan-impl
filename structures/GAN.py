"""Basic structure of a GAN."""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from typing import List


class GAN(nn.Module):
    def __init__(self, generator: nn.Module, discriminator: nn.Module):
        """Initialize a GAN.

        @param generator (Module) A module to generate results
        @param discriminator (Module) A module to compute loss
        """
        self.G = generator
        self.D = discriminator
