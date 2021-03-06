import torch
import torch.nn as nn
from self_attention_cv.transunet import TransUnet as transunet_model


class TransUNet(nn.Module):
    def __init__(
        self,
        img_dim=256,
        in_channels=1,
        classes=2,
        vit_blocks=12,
        vit_heads=4,
        vit_dim_linear_mhsa_block=1024,
        apply_masks=False,
    ):
        super(TransUNet, self).__init__()

        self.img_dim = 256
        self.in_channels = 1
        self.classes = 2
        self.vit_blocks = 12
        self.vit_heads = 4
        self.vit_dim_linear_mhsa_block = 1024

        self.apply_masks = apply_masks

        self.model = transunet_model(
            img_dim=self.img_dim,
            in_channels=self.in_channels,
            classes=self.classes,
            vit_blocks=self.vit_blocks,
            vit_heads=self.vit_heads,
            vit_dim_linear_mhsa_block=self.vit_dim_linear_mhsa_block,
        )

    def forward(self, x):
        old = x
        x = self.model(x)

        if self.apply_masks:
            x = nn.Softmax(dim=1)(x)
            x = x * torch.cat([old] * self.classes, dim=1)
        
        return x


if __name__ == "__main__":
    a = torch.rand(5, 1, 256, 256)

    model = TransUNet(img_dim=256, in_channels=1, classes=2, vit_blocks=12, vit_heads=4, vit_dim_linear_mhsa_block=1024)

    y = model(a)

    print(y.shape)
