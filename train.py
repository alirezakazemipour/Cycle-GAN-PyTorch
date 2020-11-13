import torch
from model import Generator, Discriminator
from torch.optim import Adam
from torch import from_numpy
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from itertools import chain
from copy import deepcopy
from torchsummary import summary


class Train:
    def __init__(self, n_channels, device, lr):
        torch.cuda.empty_cache()
        self.device = device
        self.n_channels = n_channels
        self.lr = lr

        self.A_Generator = Generator(self.n_channels).to(self.device)
        # summary(self.A_Generator, (3, 256, 256))
        # exit(0)
        self.A_Discriminator = Discriminator(self.n_channels).to(self.device)
        # summary(self.A_Discriminator, (3, 256, 256))
        # exit(0)
        self.B_Generator = Generator(self.n_channels).to(self.device)
        self.B_Discriminator = Discriminator(self.n_channels).to(self.device)

        self.generator_opt = Adam(chain(self.A_Generator.parameters(), self.B_Generator.parameters()), self.lr,
                                  betas=(0.5, 0.999))
        self.discriminator_opt = Adam(chain(self.A_Discriminator.parameters(), self.B_Discriminator.parameters()),
                                      self.lr, betas=(0.5, 0.999))

        self.l1_loss = torch.nn.L1Loss()
        self.mse_loss = torch.nn.MSELoss()

        self.A_fake_history = []
        self.B_fake_history = []

        # self.scheduler = lambda step: max(1 - 1e-2 * (step - 100), 0)
        # self.generator_scheduler = LambdaLR(self.generator_opt, lr_lambda=self.scheduler)
        # self.discriminator_scheduler = LambdaLR(self.discriminator_opt, lr_lambda=self.scheduler)

        self.real_labels = torch.ones((1, 1, 30, 30), device=self.device)
        self.fake_labels = torch.zeros((1, 1, 30, 30), device=self.device)

    def forward(self, real_a, real_b):

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
        a_gan_loss = self.mse_loss(self.A_Discriminator(fake_b), self.real_labels)
        b_gan_loss = self.mse_loss(self.B_Discriminator(fake_a), self.real_labels)

        a_cycle_loss = self.l1_loss(recycle_a, real_a)
        b_cycle_loss = self.l1_loss(recycle_b, real_b)

        idt_A = self.A_Generator(real_b)
        loss_idt_A = self.l1_loss(idt_A, real_b) * lam * 0.5
        idt_B = self.B_Generator(real_a)
        loss_idt_B = self.l1_loss(idt_B, real_a) * lam * 0.5

        full_obj = a_gan_loss + b_gan_loss + lam * (a_cycle_loss + b_cycle_loss) + loss_idt_A + loss_idt_B

        return full_obj, a_gan_loss, a_cycle_loss, loss_idt_A, b_gan_loss, b_cycle_loss, loss_idt_B

    def optimize_generator(self, generator_loss):
        self.generator_opt.zero_grad()
        generator_loss.backward()
        self.generator_opt.step()

    def calculate_discriminator_loss(self, real_a, history_fake_a, real_b, history_fake_b):

        for net in [self.A_Discriminator, self.B_Discriminator]:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = True

        a_dis_loss = 0.5 * (
                self.mse_loss(self.A_Discriminator(real_b), self.real_labels) +
                self.mse_loss(self.A_Discriminator(history_fake_b), self.fake_labels))
        b_dis_loss = 0.5 * (
                self.mse_loss(self.B_Discriminator(real_a), self.real_labels) +
                self.mse_loss(self.B_Discriminator(history_fake_a), self.fake_labels))

        return a_dis_loss, b_dis_loss

    def optimize_discriminator(self, a_dis_loss, b_dis_loss):
        self.discriminator_opt.zero_grad()
        a_dis_loss.backward()
        b_dis_loss.backward()
        self.discriminator_opt.step()

    # def schedule_optimizers(self):
    #     self.generator_scheduler.step()
    #     self.discriminator_scheduler.step()

    def get_history(self, fake_a, fake_b):
        if len(self.A_fake_history) < 50:
            self.A_fake_history.append(fake_a)
            self.B_fake_history.append(fake_b)
            return fake_a, fake_b
        else:
            if np.random.uniform() < 0.5:
                rnd_idx = np.random.randint(0, len(self.A_fake_history) - 1)
                a_fake_history = self.A_fake_history[rnd_idx].clone()
                b_fake_history = self.B_fake_history[rnd_idx].clone()
                self.A_fake_history[rnd_idx] = fake_a.clone()
                self.B_fake_history[rnd_idx] = fake_b.clone()
                return a_fake_history, b_fake_history
            else:
                return fake_a, fake_b

    def save_weights(self, epoch, lr):
        torch.save({"A_Generator_dict": self.A_Generator.state_dict(),
                    "B_Generator_dict": self.B_Generator.state_dict(),
                    "A_Discriminator_dict": self.A_Discriminator.state_dict(),
                    "B_Discriminator_dict": self.B_Discriminator.state_dict(),
                    "discriminator_opt_dict": self.discriminator_opt.state_dict(),
                    "generator_opt_dict": self.generator_opt.state_dict(),
                    "epoch": epoch,
                    "lr": lr}, "CycleGan.pth")

    def load_weights(self, path):

        checkpoint = torch.load(path)
        A_Generator_dict = checkpoint["A_Generator_dict"]
        B_Generator_dict = checkpoint["B_Generator_dict"]
        A_Discriminator_dict = checkpoint["A_Discriminator_dict"]
        B_Discriminator_dict = checkpoint["B_Discriminator_dict"]
        discriminator_opt_dict = checkpoint["discriminator_opt_dict"]
        generator_opt_dict = checkpoint["generator_opt_dict"]
        epoch = checkpoint["epoch"]
        lr = checkpoint["lr"]
        self.A_Generator.load_state_dict(A_Generator_dict)
        self.B_Generator.load_state_dict(B_Generator_dict)
        self.A_Discriminator.load_state_dict(A_Discriminator_dict)
        self.B_Discriminator.load_state_dict(B_Discriminator_dict)
        self.discriminator_opt.load_state_dict(discriminator_opt_dict)
        self.generator_opt.load_state_dict(generator_opt_dict)
        return epoch

    def set_to_eval(self):
        self.A_Generator.eval()
        self.B_Generator.eval()