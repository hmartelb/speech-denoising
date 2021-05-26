import torch
import torch.nn as nn


def double_conv(in_channels, out_channels, kernel_size, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(
        self,
        n_channels=1,
        n_class=2,
        unet_depth=5,
        unet_scale_factor=32,
        double_conv_kernel_size=3,
        double_conv_padding=1,
        maxpool_kernel_size=2,
        upsample_scale_factor=2,
    ):
        super().__init__()

        self.depth = unet_depth

        self.conv_down = nn.ModuleList(
            [double_conv(n_channels, unet_scale_factor, double_conv_kernel_size, double_conv_padding)]
            + [
                double_conv(
                    unet_scale_factor * (2 ** i),
                    unet_scale_factor * (2 ** (i + 1)),
                    double_conv_kernel_size,
                    double_conv_padding,
                )
                for i in range(unet_depth)
            ]
        )

        self.maxpool = nn.MaxPool2d(maxpool_kernel_size)
        self.upsample = nn.Upsample(scale_factor=upsample_scale_factor, mode="bilinear", align_corners=True)

        self.conv_up = nn.ModuleList(
            [
                double_conv(
                    unet_scale_factor * (2 ** (unet_depth - i - 1)),
                    unet_scale_factor * (2 ** (unet_depth - i - 2)),
                    double_conv_kernel_size,
                    double_conv_padding,
                )
                for i in range(unet_depth - 1)
            ]
            + [nn.Conv2d(unet_scale_factor, n_class, 1)]
        )

    def forward(self, x):
        old = x
        storage = []
        last_item = self.depth - 1

        for i in range(self.depth):
            x = self.conv_down[i](x)
            if i != last_item:
                storage.append(x)
                x = self.maxpool(x)

        storage.reverse()

        for i in range(self.depth):
            if i != last_item:
                x = self.upsample(x)
                x = self.conv_up[i](x)
                x = (
                    x + storage[i]
                )  # Avoid RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
            else:
                x = self.conv_up[i](x)

        # x = nn.ReLU()(x)
        x = nn.Softmax(dim=1)(x)

        x = x * torch.cat([old, old], dim=1)

        return x


if __name__ == "__main__":
    from torchsummary import summary
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    model = UNet(1, 2, 6, 32)
    summary(model, (1, 256, 256))
    x = torch.ones([50, 1, 256, 256])
    y = model.forward(x)

    print(y.shape)
