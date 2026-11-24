import torch
import torch.nn as nn

from floodvision.models.blocks import DoubleConv


class BaselineCNN(nn.Module):

    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        base_channels=16
    ):

        super().__init__()

        c = base_channels

        # Encoder
        self.encoder1 = DoubleConv(
            in_channels, c
        )

        self.encoder2 = DoubleConv(
            c, c * 2
        )

        self.encoder3 = DoubleConv(
            c * 2, c * 4
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Bottleneck
        self.bottleneck = DoubleConv(
            c * 4, c * 8
        )

        # Decoder 3
        self.up3 = nn.ConvTranspose2d(
            c * 8,
            c * 4,
            kernel_size=2,
            stride=2
        )

        self.decoder3 = DoubleConv(
            c * 4, c * 4
        )

        # Decoder 2
        self.up2 = nn.ConvTranspose2d(
            c * 4,
            c * 2,
            kernel_size=2,
            stride=2
        )

        self.decoder2 = DoubleConv(
            c * 2, c * 2
        )

        # Decoder 1
        self.up1 = nn.ConvTranspose2d(
            c * 2,
            c,
            kernel_size=2,
            stride=2
        )

        self.decoder1 = DoubleConv(
            c, c
        )

        # Pixel-wise classification
        self.output = nn.Conv2d(
            c,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        x = self.encoder1(x)
        x = self.pool(x)

        x = self.encoder2(x)
        x = self.pool(x)

        x = self.encoder3(x)
        x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up3(x)
        x = self.decoder3(x)

        x = self.up2(x)
        x = self.decoder2(x)

        x = self.up1(x)
        x = self.decoder1(x)

        # Raw logits
        return self.output(x)