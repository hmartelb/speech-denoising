import torch
import torch.nn as nn
import torch.nn.functional as F


def convnorm(in_ch, out_ch, filter_size):
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, filter_size, padding=filter_size // 2),
        nn.BatchNorm1d(out_ch),
        nn.LeakyReLU(0.1),
    )

    # n_channels=1,
    # n_class=2,
    # unet_depth=5,
    # unet_scale_factor=32,
    # double_conv_kernel_size=3,
    # double_conv_padding=1,
    # maxpool_kernel_size=2,
    # upsample_scale_factor=2,


class UNetDNP(nn.Module):
    def __init__(self, n_channels=1, n_class=2, unet_depth=6, nefilters=60):
        super(UNetDNP, self).__init__()
        self.unet_depth = unet_depth
        self.nefilters = nefilters
        self.n_class = n_class
        filter_size = 15
        merge_filter_size = 5
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        echannelout = [(i + 1) * nefilters for i in range(unet_depth)]
        echannelin = [n_channels] + [(i + 1) * nefilters for i in range(unet_depth - 1)]
        dchannelout = echannelout[::-1]
        dchannelin = [dchannelout[0] * 2] + [(i) * nefilters + (i - 1) * nefilters for i in range(unet_depth, 1, -1)]

        for i in range(self.unet_depth):
            self.encoder.append(convnorm(echannelin[i], echannelout[i], filter_size))
            self.decoder.append(convnorm(dchannelin[i], dchannelout[i], merge_filter_size))

        self.middle = convnorm(echannelout[-1], echannelout[-1], filter_size)

        self.out = nn.Sequential(nn.Conv1d(nefilters + 1, n_class, 1), nn.Tanh())
        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.downsample = nn.MaxPool1d(2)

    def forward(self, x):
        encoder = list()
        input = x

        for i in range(self.unet_depth):
            x = self.encoder[i](x)
            encoder.append(x)
            x = self.downsample(x)

        x = self.middle(x)
        encoder.reverse()

        for i in range(self.unet_depth):
            x = self.upsample(x)
            x = torch.cat([x, encoder[i]], dim=1)
            x = self.decoder[i](x)
        x = torch.cat([x, input], dim=1)

        x = self.out(x)
        x = nn.Softmax(dim=1)(x)
        x = x * torch.cat([input] * self.n_class, dim=1)

        return x


if __name__ == "__main__":
    from torchsummary import summary
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    model = UNetDNP()
    summary(model, [1, 16000 * 4])
    x = torch.ones([4, 1, 16000 * 4])
    y = model.forward(x)

    print(y.shape)
