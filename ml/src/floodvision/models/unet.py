import torch
import torch.nn as nn

from .blocks import DoubleConv


class UNet(nn.Module):

    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        base_channels=16
    ):
        super().__init__()

        c = base_channels

        # Encoder
        self.encoder1 = DoubleConv(in_channels, c)
        self.encoder2 = DoubleConv(c, c * 2)
        self.encoder3 = DoubleConv(c * 2, c * 4)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(c * 4, c * 8)

        # Decoder Level 3
        self.up3 = nn.ConvTranspose2d(
            c * 8, c * 4, kernel_size=2, stride=2
        )
        self.decoder3 = DoubleConv(c * 8, c * 4)

        # Decoder Level 2
        self.up2 = nn.ConvTranspose2d(
            c * 4, c * 2, kernel_size=2, stride=2
        )
        self.decoder2 = DoubleConv(c * 4, c * 2)

        # Decoder Level 1
        self.up1 = nn.ConvTranspose2d(
            c * 2, c, kernel_size=2, stride=2
        )
        self.decoder1 = DoubleConv(c * 2, c)

        # Segmentation head
        self.output = nn.Conv2d(
            c, out_channels, kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder + Skip connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)

        return self.output(d1)