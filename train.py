import torch
from model import Generator, Discriminator
from torch.optim import Adam
from torch import from_numpy
import numpy as np


class Train:
    def __init__(self, n_channels, lr=2e-4):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_channels = n_channels
        self.lr = lr

        self.A_Generator = Generator(self.n_channels).to(self.device)
        self.A_Discriminator = Discriminator(self.n_channels).to(self.device)

        self.B_Generator = Generator(self.n_channels).to(self.device)
        self.B_Discriminator = Discriminator(self.n_channels).to(self.device)

        self.generator_opt = Adam(list(self.A_Generator.parameters()) + list(self.B_Generator.parameters()), self.lr)
        self.discriminator_opt = Adam(
            list(self.A_Discriminator.parameters()) + list(self.B_Discriminator.parameters()), self.lr)

    def forward(self, real_a, real_b):
        real_a = np.expand_dims(real_a, axis=0)
        real_b = np.expand_dims(real_b, axis=0)
        real_a = from_numpy(real_a).float().permute([0, 3, 1, 2]).to(self.device)
        real_b = from_numpy(real_b).float().permute([0, 3, 1, 2]).to(self.device)
        fake_b = self.A_Generator(real_a)
        recycle_a = self.B_Generator(fake_b)
        fake_a = self.B_Generator(real_b)
        recycle_b = self.A_Generator(fake_a)

        return fake_a, recycle_a, fake_b, recycle_b
