
import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

from .blocks import DoubleConv, DecoderBlock


class ResNetUNet(nn.Module):

    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        pretrained=True
    ):
        super().__init__()

        if in_channels != 2:
            raise ValueError(
                "ResNetUNet expects two SAR channels."
            )

        # 1. Load ResNet18
        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        encoder = resnet18(weights=weights)

        # 2. Adapt RGB convolution to VV/VH
        new_conv = nn.Conv2d(
            in_channels=2,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        if pretrained:
            with torch.no_grad():

                adapted_weights = (
                    encoder.conv1.weight
                    .mean(dim=1, keepdim=True)
                    .repeat(1, 2, 1, 1)
                    * (3 / 2)
                )

                new_conv.weight.copy_(
                    adapted_weights
                )
        else:
            nn.init.kaiming_normal_(
                new_conv.weight,
                mode="fan_out",
                nonlinearity="relu"
            )

        # 3. Encoder stages
        self.encoder0 = nn.Sequential(
            new_conv,
            encoder.bn1,
            encoder.relu
        )

        self.pool = encoder.maxpool

        self.encoder1 = encoder.layer1
        self.encoder2 = encoder.layer2
        self.encoder3 = encoder.layer3
        self.encoder4 = encoder.layer4

        # 4. Decoder stages
        self.decoder3 = DecoderBlock(
            512, 256, 256
        )

        self.decoder2 = DecoderBlock(
            256, 128, 128
        )

        self.decoder1 = DecoderBlock(
            128, 64, 64
        )

        self.decoder0 = DecoderBlock(
            64, 64, 32
        )

        # 5. Recover original resolution
        self.final_up = nn.ConvTranspose2d(
            32,
            16,
            kernel_size=2,
            stride=2
        )

        self.final_conv = DoubleConv(16, 16)

        # 6. Segmentation output
        self.output = nn.Conv2d(
            16,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e0 = self.encoder0(x)

        e1 = self.encoder1(
            self.pool(e0)
        )

        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with skip connections
        d3 = self.decoder3(e4, e3)
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2, e1)
        d0 = self.decoder0(d1, e0)

        # Final resolution recovery
        x = self.final_up(d0)
        x = self.final_conv(x)

        return self.output(x)
