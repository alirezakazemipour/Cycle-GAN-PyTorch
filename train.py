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

        self.cycle_loss = torch.nn.L1Loss()

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

    def calculate_generator_loss(self, real_a, fake_a, recycle_a, real_b, fake_b, recycle_b, lam=10):

        for net in [self.A_Discriminator, self.B_Discriminator]:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = False

        a_gan_loss = ((self.A_Discriminator(fake_b) - 1) ** 2).mean()
        b_gan_loss = ((self.A_Discriminator(fake_a) - 1) ** 2).mean()

        a_cycle_loss = self.cycle_loss(recycle_a, real_a)
        b_cycle_loss = self.cycle_loss(recycle_b, real_b)

        full_obj = a_gan_loss + b_gan_loss + lam * (a_cycle_loss + b_cycle_loss)

        return full_obj

    def optimize_generator(self, generator_loss):
        self.generator_opt.zero_grad()
        generator_loss.backward()
        self.generator_opt.step()

    def calculate_discriminator_loss(self, real_a, fake_a, recycle_a, real_b, fake_b, recycle_b, lam=10):

        for net in [self.A_Discriminator, self.B_Discriminator]:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = False

        pass
