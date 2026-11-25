import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

from .blocks import (
    DepthwiseSeparableConv,
    LightweightDecoderBlock
)


class MobileNetUNet(nn.Module):

    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        pretrained=True
    ):

        super().__init__()

        if in_channels != 2:
            raise ValueError(
                "MobileNetUNet expects two SAR channels."
            )

        # 1. Load pretrained backbone

        weights = (
            MobileNet_V3_Small_Weights.DEFAULT
            if pretrained
            else None
        )

        encoder = mobilenet_v3_small(
            weights=weights
        )

        # 2. Adapt RGB input to SAR input

        original_conv = encoder.features[0][0]

        new_conv = nn.Conv2d(
            in_channels=2,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        if pretrained:

            with torch.no_grad():

                adapted_weights = (
                    original_conv.weight
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

        encoder.features[0][0] = new_conv

        # 3. Multi-scale encoder stages

        self.encoder0 = encoder.features[0]

        self.encoder1 = encoder.features[1]

        self.encoder2 = encoder.features[2:4]

        self.encoder3 = encoder.features[4:9]

        self.encoder4 = encoder.features[9:13]

        # 4. Bottleneck channel reduction

        self.bottleneck = nn.Sequential(

            nn.Conv2d(
                576,
                128,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True)
        )

        # 5. Lightweight decoder

        self.decoder3 = LightweightDecoderBlock(
            in_channels=128,
            skip_channels=48,
            out_channels=64
        )

        self.decoder2 = LightweightDecoderBlock(
            in_channels=64,
            skip_channels=24,
            out_channels=32
        )

        self.decoder1 = LightweightDecoderBlock(
            in_channels=32,
            skip_channels=16,
            out_channels=24
        )

        self.decoder0 = LightweightDecoderBlock(
            in_channels=24,
            skip_channels=16,
            out_channels=16
        )

        # 6. Final feature refinement

        self.final_conv = DepthwiseSeparableConv(
            16,
            16
        )

        # 7. Segmentation head

        self.output = nn.Conv2d(
            16,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        input_size = x.shape[-2:]

        # Encoder

        e0 = self.encoder0(x)

        e1 = self.encoder1(e0)

        e2 = self.encoder2(e1)

        e3 = self.encoder3(e2)

        e4 = self.encoder4(e3)

        # Bottleneck

        b = self.bottleneck(e4)

        # Decoder with skip connections

        d3 = self.decoder3(b, e3)

        d2 = self.decoder2(d3, e2)

        d1 = self.decoder1(d2, e1)

        d0 = self.decoder0(d1, e0)

        # Restore original spatial resolution

        x = F.interpolate(
            d0,
            size=input_size,
            mode="bilinear",
            align_corners=False
        )

        x = self.final_conv(x)

        return self.output(x)