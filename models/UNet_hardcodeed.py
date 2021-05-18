import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, in_ch, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(in_ch, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.dconv_down5 = double_conv(512, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.dconv_up4 = double_conv(512, 512)
        self.dconv_up3 = double_conv(512, 256)
        self.dconv_up2 = double_conv(256, 128)
        self.dconv_up1 = double_conv(128, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        x = self.dconv_down5(x)

        x = self.upsample(x)
        x += conv4
        x = self.dconv_up4(x)

        x = self.upsample(x)
        x = self.dconv_up3(x)
        x += conv3

        x = self.upsample(x)
        x = self.dconv_up2(x)
        x += conv2

        x = self.upsample(x)
        x = self.dconv_up1(x)
        x += conv1

        out = self.conv_last(x)

        return out


from torchsummary import summary
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = UNet(1, 2)
summary(model, (1, 256, 256))
x = torch.ones([5, 1, 256, 256])
y = model.forward(x)

print(y.shape)
