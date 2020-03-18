from torch import nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, n_channels):
        super(Generator, self).__init__()
        self.n_channels = n_channels

        self.conv_relu_norm = ConvNormRelu(self.n_channels, 64, conv_padding=0).cuda()

        self.n_ds_layers = 2
        input_ds_channel = 64
        self.down_samples = []
        for _ in range(self.n_ds_layers):
            down_sample = DownSample(n_channels=input_ds_channel, n_filters=2 * input_ds_channel).cuda()
            self.down_samples.append(down_sample)
            input_ds_channel = 2 * input_ds_channel

        self.n_resnet_layers = 9
        self.resnet_layers = [ResNet().cuda() for _ in range(self.n_resnet_layers)]

        self.up_samples = []
        for _ in range(self.n_ds_layers):
            up_sample = UpSample(n_channels=int(input_ds_channel), n_filters=int(input_ds_channel / 2)).cuda()
            self.up_samples.append(up_sample)
            input_ds_channel = input_ds_channel / 2

        self.conv_tanh_norm = ConvNormRelu(int(input_ds_channel), 3, activation="tanh", conv_padding=0).cuda()

    def forward(self, inputs):
        x = self.conv_relu_norm(inputs)
        for i in range(self.n_ds_layers):
            x = self.down_samples[i](x)

        for i in range(self.n_resnet_layers):
            x = self.resnet_layers[i](x)

        for i in range(self.n_ds_layers):
            x = self.up_samples[i](x)

        return self.conv_tanh_norm(x)


class ConvNormRelu(nn.Module):
    def __init__(self, n_channels, n_filters, kernel_size=7, stride=1, do_reflect_padding=True,
                 reflect_padding=3, conv_padding=1, activation="relu", do_norm=True):
        super(ConvNormRelu, self).__init__()
        self.activation = activation
        self.do_norm = do_norm
        self.do_reflect_padding = do_reflect_padding

        self.padding = nn.ReflectionPad2d(reflect_padding)
        self.conv = nn.Conv2d(in_channels=n_channels,
                              out_channels=n_filters,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=conv_padding)

        self.norm = nn.InstanceNorm2d(n_filters)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, x):
        if self.do_reflect_padding:
            x = self.padding(x)
        x = self.conv(x)
        if self.do_norm:
            x = self.norm(x)
        if self.activation == "relu":
            return F.relu(x)
        elif self.activation == "leaky relu":
            return F.leaky_relu(x, 0.2)
        elif self.activation == "tanh":
            return F.tanh(x)


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
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight)
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
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.normal_(layer.weight)
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

        self.conv_relu_norm1 = ConvNormRelu(self.in_channels, self.in_channels,
                                            self.kernel_size, conv_padding=0, reflect_padding=1)
        self.conv_relu_norm2 = ConvNormRelu(self.in_channels, self.in_channels,
                                            self.kernel_size, conv_padding=0, reflect_padding=1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, inputs):
        x = self.conv_relu_norm1(inputs)
        x = self.conv_relu_norm2(x)
        return inputs + x


class Discriminator(nn.Module):
    def __init__(self, n_channels):
        super(Discriminator, self).__init__()
        self.n_channels = n_channels
        filters = [64, 128, 256, 512]

        self.C64 = ConvNormRelu(self.n_channels, filters[0], 4, 2, do_reflect_padding=False, activation="leaky relu", do_norm=False)
        self.conv_leakyrelu_norms = []
        for idx, filter in enumerate(filters[:-1]):
            C = self.C64 = ConvNormRelu(filter, filters[idx + 1], 4, 2, do_reflect_padding=False,
                                        activation="leaky relu", do_norm=True)
            self.conv_leakyrelu_norms.append(C)

        self.output = nn.Conv2d(filters[-1], 1, kernel_size=4, stride=1, padding=1)
        nn.init.normal_(self.output.weight, 0, 0.02)
        self.output.bias.data.zero_()

    def forward(self, inputs):
        x = self.C64(inputs)
        for i in range(len(self.conv_relu_norms)):
            x = self.conv_leakyrelu_norms[i](x)
        return self.output(x)
