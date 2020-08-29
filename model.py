from torch import nn
import torch.nn.functional as F
import torch


class Generator(nn.Module):
    def __init__(self, n_channels):
        super(Generator, self).__init__()
        self.n_channels = n_channels

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channels=self.n_channels,
                           out_channels=64,
                           kernel_size=7,
                           padding=0),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(True),
                 nn.Conv2d(in_channels=64,
                           out_channels=128,
                           kernel_size=3,
                           stride=2,
                           padding=1),
                 nn.InstanceNorm2d(128),
                 nn.ReLU(True),
                 nn.Conv2d(in_channels=128,
                           out_channels=256,
                           kernel_size=3,
                           stride=2,
                           padding=1),
                 nn.InstanceNorm2d(256),
                 nn.ReLU(True)]

        for _ in range(9):
            model += [ResNet().cuda()]

        model += [nn.ConvTranspose2d(in_channels=256,
                                     out_channels=128,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     output_padding=1),
                  nn.InstanceNorm2d(128),
                  nn.ReLU(True),
                  nn.ConvTranspose2d(in_channels=128,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     output_padding=1),
                  nn.InstanceNorm2d(64),
                  nn.ReLU(True),
                  nn.ReflectionPad2d(3),
                  nn.Conv2d(in_channels=64,
                            out_channels=3,
                            kernel_size=7,
                            padding=0),
                  nn.InstanceNorm2d(3),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

        for layer in self.modules():
            if isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, 0, 0.02)
                layer.bias.data.zero_()

    def forward(self, inputs):
        # x = self.pad1(inputs)
        # x = self.c7s1_64(x)
        # x = F.relu(self.norm1(x))
        #
        # x = self.d128(x)
        # x = F.relu(self.norm2(x))
        #
        # x = self.d256(x)
        # x = F.relu(self.norm3(x))
        #
        # for layer in self.resnet_layers:
        #     x = layer(x)
        #
        # x = self.u128(x)
        # x = F.relu(self.norm4(x))
        #
        # x = self.u64(x)
        # x = F.relu(self.norm5(x))
        #
        # x = self.pad2(x)
        # x = self.output(x)
        # x = self.norm6(x)

        return self.model(inputs)


# region ResNet


class ResNet(nn.Module):
    def __init__(self, in_channels=256, kernel_size=3):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(self.in_channels, 256, 3, padding=0)
        self.norm1 = nn.InstanceNorm2d(256)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=0)
        self.norm2 = nn.InstanceNorm2d(256)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, 0, 0.02)
                layer.bias.data.zero_()

    def forward(self, inputs):
        x = self.pad1(inputs)
        x = self.conv1(x)
        x = F.relu(self.norm1(x))

        x = self.pad2(x)
        x = self.conv2(x)
        x = self.norm2(x)

        return x + inputs  # torch.cat((inputs, x), dim=1)


# endregion


class Discriminator(nn.Module):
    def __init__(self, n_channels):
        super(Discriminator, self).__init__()
        self.n_channels = n_channels

        self.c64 = nn.Conv2d(in_channels=3,
                             out_channels=64,
                             kernel_size=4,
                             stride=2,
                             padding=1)

        self.c128 = nn.Conv2d(in_channels=64,
                              out_channels=128,
                              kernel_size=4,
                              stride=2,
                              padding=1)
        self.norm1 = nn.InstanceNorm2d(128)

        self.c256 = nn.Conv2d(in_channels=128,
                              out_channels=256,
                              kernel_size=4,
                              stride=2,
                              padding=1)
        self.norm2 = nn.InstanceNorm2d(256)

        self.c512 = nn.Conv2d(in_channels=256,
                              out_channels=512,
                              kernel_size=4,
                              stride=2,
                              padding=1)
        self.norm3 = nn.InstanceNorm2d(512)

        self.c512_2 = nn.Conv2d(in_channels=512,
                                out_channels=512,
                                kernel_size=4,
                                stride=1,
                                padding=1)
        self.norm4 = nn.InstanceNorm2d(512)

        self.output = nn.Conv2d(in_channels=512,
                                out_channels=1,
                                kernel_size=4,
                                stride=1,
                                padding=1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, 0, 0.02)
                layer.bias.data.zero_()

    def forward(self, inputs):
        x = F.leaky_relu(self.c64(inputs), 0.2)

        x = self.c128(x)
        x = F.leaky_relu(self.norm1(x), 0.2)

        x = self.c256(x)
        x = F.leaky_relu(self.norm2(x), 0.2)

        x = self.c512(x)
        x = F.leaky_relu(self.norm3(x), 0.2)

        x = self.c512_2(x)
        x = F.leaky_relu(self.norm4(x), 0.2)

        return self.output(x)
