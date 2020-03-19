import torch
from model import Generator, Discriminator
from torch.optim import Adam
from torch import from_numpy
import numpy as np
import random
from torch.optim.lr_scheduler import LambdaLR


class Train:
    def __init__(self, n_channels, lr=2e-4):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_channels = n_channels
        self.lr = lr

        self.A_Generator = Generator(self.n_channels).to(self.device)
        self.A_Discriminator = Discriminator(self.n_channels).to(self.device)

        self.B_Generator = Generator(self.n_channels).to(self.device)
        self.B_Discriminator = Discriminator(self.n_channels).to(self.device)

        self.generator_opt = Adam(list(self.A_Generator.parameters()) + list(self.B_Generator.parameters()), self.lr,
                                  betas=(0.5, 0.999))
        self.discriminator_opt = Adam(
            list(self.A_Discriminator.parameters()) + list(self.B_Discriminator.parameters()), self.lr,
            betas=(0.5, 0.999))

        self.cycle_loss = torch.nn.L1Loss()

        self.A_fake_history = []
        self.B_fake_history = []

        self.scheduler = lambda step: 1 if step < 100 else max(1 - 1e-2 * (step - 100), 0)
        self.generator_scheduler = LambdaLR(self.generator_opt, lr_lambda=self.scheduler)
        self.discriminator_scheduler = LambdaLR(self.discriminator_opt, lr_lambda=self.scheduler)

    def forward(self, real_a, real_b):
        real_a = np.expand_dims(real_a, axis=0)
        real_b = np.expand_dims(real_b, axis=0)
        real_a = from_numpy(real_a).float().permute([0, 3, 1, 2]).to(self.device)
        real_b = from_numpy(real_b).float().permute([0, 3, 1, 2]).to(self.device)
        fake_b = self.A_Generator(real_a)
        recycle_a = self.B_Generator(fake_b)
        fake_a = self.B_Generator(real_b)
        recycle_b = self.A_Generator(fake_a)

        self.add_to_history(fake_a.detach().cpu().numpy(), fake_b.detach().cpu().numpy())

        return fake_a, recycle_a, fake_b, recycle_b

    def calculate_generator_loss(self, real_a, fake_a, recycle_a, real_b, fake_b, recycle_b, lam=10):

        real_a = np.expand_dims(real_a, axis=0)
        real_b = np.expand_dims(real_b, axis=0)
        real_a = from_numpy(real_a).float().permute([0, 3, 1, 2]).to(self.device)
        real_b = from_numpy(real_b).float().permute([0, 3, 1, 2]).to(self.device)

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
        self.generator_scheduler.step()

    def calculate_discriminator_loss(self, real_a, history_fake_a, real_b, history_fake_b):

        real_a = np.expand_dims(real_a, axis=0)
        real_b = np.expand_dims(real_b, axis=0)
        real_a = from_numpy(real_a).float().permute([0, 3, 1, 2]).to(self.device)
        real_b = from_numpy(real_b).float().permute([0, 3, 1, 2]).to(self.device)
        history_fake_a = from_numpy(history_fake_a).float().to(self.device)
        history_fake_b = from_numpy(history_fake_b).float().to(self.device)

        for net in [self.A_Discriminator, self.B_Discriminator]:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = True

        a_dis_loss = ((self.A_Discriminator(real_b) - 1) ** 2).mean() + \
                     (self.A_Discriminator(history_fake_b) ** 2).mean()
        b_dis_loss = ((self.B_Discriminator(real_a) - 1) ** 2).mean() + \
                     (self.B_Discriminator(history_fake_a) ** 2).mean()

        return a_dis_loss, b_dis_loss

    def optimize_discriminator(self, a_dis_loss, b_dis_loss):
        self.discriminator_opt.zero_grad()
        a_dis_loss.backward()
        b_dis_loss.backward()
        self.discriminator_opt.step()
        self.discriminator_scheduler.step()

    def get_history(self):
        if len(self.A_fake_history) >= 50:
            return random.sample(self.A_fake_history, 50), random.sample(self.B_fake_history, 50)
        else:
            return random.sample(self.A_fake_history, len(self.A_fake_history)),\
                   random.sample(self.B_fake_history, len(self.B_fake_history))

    def add_to_history(self, fake_a, fake_b):
        if len(self.A_fake_history) < 1000:
            self.A_fake_history.append(fake_a)
            self.B_fake_history.append(fake_b)
        else:
            _ = self.A_fake_history.pop()
            _ = self.B_fake_history.pop()
            self.A_fake_history.append(fake_a)
            self.B_fake_history.append(fake_b)
