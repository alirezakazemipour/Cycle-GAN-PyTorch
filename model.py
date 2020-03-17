from torch import nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, n_channels):
        super(Generator, self).__init__()
        self.n_channels = n_channels

        self.conv_relu_norm1 = self.conv_norm(self.n_channels, 64)

        self.n_ds_layers = 2
        input_ds_channel = 64
        self.down_samples = []
        for _ in range(self.n_ds_layers):
            down_sample = DownSample(n_channels=input_ds_channel, n_filters=2 * input_ds_channel)
            self.down_samples.append(down_sample)
            input_ds_channel = 2 * input_ds_channel

        self.n_resnet_layers = 9
        self.resnet_layers = [ResNet() for _ in range(self.n_resnet_layers)]

        for _ in range(self.n_ds_layers):
            up_sample = UpSample(n_channels=input_ds_channel, n_filters=input_ds_channel / 2)
            self.down_samples.append(up_sample)
            input_ds_channel = input_ds_channel / 2

        self.conv_relu_norm2 = self.conv_norm(input_ds_channel, 3)

    def forward(self, inputs):
        x = self.conv_relu_norm1(inputs)
        for i in range(self.n_ds_layers):
            x = self.down_samples[i](x)

        for i in range(self.n_resnet_layers):
            x = self.resnet_layers[i](x)

        for i in range(self.n_ds_layers):
            x = self.up_samples[i](x)

        return self.conv_relu_norm2(x)


class ConvNormRelu(nn.Module):
    def __init__(self, n_channels, n_filters, kernel_size=7, stride=1, padding=3):
        super(ConvNormRelu, self).__init__()
        self.padding = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels=n_channels,
                              out_channels=n_filters,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=1)
        self.norm = nn.InstanceNorm2d(n_filters)

        for layer in self.modules():
            nn.init.normal_(layer.weight, 0, 0.02)
            layer.bias.data.zero_()

    def forward(self, x):
        x = self.padding(x)
        x = self.conv(x)
        x = self.norm(x)
        return F.relu(x)


class DownSample(nn.Module):
    def __init__(self, n_channels, n_filters, kernel_size=3, stride=2):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(in_channels=n_channels,
                              out_channels=n_filters,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=1)
        self.norm = nn.InstanceNorm2d(n_filters)

        for layer in self.modules():
            nn.init.normal_(layer.weight, 0, 0.02)
            layer.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return F.relu(x)


class UpSample(nn.Module):
    def __init__(self, n_channels, n_filters, kernel_size=3, stride=2):
        super(UpSample, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=n_channels,
                                         out_channels=n_filters,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=1)
        self.norm = nn.InstanceNorm2d(n_filters)

        for layer in self.modules():
            nn.init.normal_(layer.weight, 0, 0.02)
            layer.bias.data.zero_()

    def forward(self, x):
        x = self.deconv(x)
        x = self.norm(x)
        return F.relu(x)


class ResNet(nn.Module):
    def __init__(self, in_channels=256, kernel_size=3):
        super(ResNet, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.conv_relu_norm1 = ConvNormRelu(self.in_channels, self.in_channels, self.kernel_size, padding=1)
        self.conv_relu_norm2 = ConvNormRelu(self.in_channels, self.in_channels, self.kernel_size, padding=1)

        for layer in self.modules():
            nn.init.normal_(layer.weight, 0, 0.02)
            layer.bias.data.zero_()

    def forward(self, inputs):
        x = self.conv_relu_norm1(inputs)
        x = self.conv_relu_norm2(x)
        return inputs + x
