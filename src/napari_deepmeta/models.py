import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=False,
        padding="same",
        stride=1,
    ):
        super().__init__()
        padding = 1 if stride > 1 else "same"
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=bias,
            padding=padding,
            stride=stride,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias, padding="same"
        )

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        drop_r,
        conv=SeparableConv2d,
    ):
        super().__init__()
        self.double_conv = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(drop_r),
            conv(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(drop_r),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down_Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        drop=0.1,
        conv_l=SeparableConv2d,
    ):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, drop, conv=conv_l)
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        c = self.conv(x)
        return c, self.down(c)


class Bridge(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        drop=0.1,
        conv_l=SeparableConv2d,
    ):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, drop, conv=conv_l)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_l=SeparableConv2d):
        super().__init__()
        self.conv = conv_l(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Normalize_Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int):
        super().__init__()
        self.down = nn.MaxPool2d(kernel)
        self.norm = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding="same"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.down(x))


class Normalize_Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int):
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=kernel)
        self.norm = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding="same"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.up(x))


class Concat_Block(nn.Module):
    def __init__(
        self,
        kernels_down,
        filters_down,
        kernels_up,
        filters_up,
    ):
        super().__init__()
        self.norm_down = nn.ModuleList(
            [
                Normalize_Down(in_, 32, kernel)
                for (kernel, in_) in zip(kernels_down, filters_down)
            ]
        )
        self.norm_up = nn.ModuleList(
            [
                Normalize_Up(in_, 32, kernel)
                for (kernel, in_) in zip(kernels_up, filters_up)
            ]
        )

    def forward(self, down, up):
        res = [l(d) for d, l in zip(down, self.norm_down)]
        res.extend(l(u) for u, l in zip(up, self.norm_up))
        return torch.cat(res, dim=1)


class Up_Block_3p(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels: int,
        drop=0.1,
        conv_l=SeparableConv2d,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, 32, kernel_size=(2, 2), stride=(2, 2)
        )
        self.conv = DoubleConv(160, 160, drop, conv=conv_l)

    def forward(self, x: torch.Tensor, conc: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x)
        x = torch.cat([conc, x1], dim=1)
        return self.conv(x)


class Unet3plus(nn.Module):
    def __init__(
        self,
        filters=16,
        classes=3,
        drop_r=0.1,
        conv_l=SeparableConv2d,
    ):
        """
        Initialize the U-Net model.
        :param filters: Number of filters in the first convolutional layer.
        :type filters: int
        :param classes: Number of classes to predict.
        :type classes: int
        :param drop_r: Dropout rate.
        :type drop_r: float
        """
        super().__init__()

        self.down1 = Down_Block(1, filters, conv_l=conv_l)
        self.down2 = Down_Block(filters, filters * 2, drop_r, conv_l=conv_l)
        self.down3 = Down_Block(
            filters * 2, filters * 4, drop_r, conv_l=conv_l
        )
        self.down4 = Down_Block(
            filters * 4, filters * 8, drop_r, conv_l=conv_l
        )

        self.bridge = Bridge(filters * 8, filters * 16, drop_r, conv_l=conv_l)

        self.up1 = Up_Block_3p(filters * 16, 160, drop_r, conv_l=conv_l)
        self.up2 = Up_Block_3p(160, 160, drop_r, conv_l=conv_l)
        self.up3 = Up_Block_3p(160, 160, drop_r, conv_l=conv_l)
        self.up4 = Up_Block_3p(160, 160, drop_r, conv_l=conv_l)

        self.concat1 = Concat_Block(
            kernels_down=[8, 4, 2, 1],
            filters_down=[filters, filters * 2, filters * 4, filters * 8],
            kernels_up=[],
            filters_up=[],
        )
        self.concat2 = Concat_Block(
            kernels_down=[4, 2, 1],
            filters_down=[filters, filters * 2, filters * 4],
            kernels_up=[4],
            filters_up=[filters * 16],
        )
        self.concat3 = Concat_Block(
            kernels_down=[2, 1],
            filters_down=[filters, filters * 2],
            kernels_up=[4, 8],
            filters_up=[160, filters * 16],
        )
        self.concat4 = Concat_Block(
            kernels_down=[1],
            filters_down=[filters],
            kernels_up=[4, 8, 16],
            filters_up=[160, 160, filters * 16],
        )
        self.outc = OutConv(160, classes, conv_l=conv_l)

    def forward(self, x):
        c1, x1 = self.down1(x)
        c2, x2 = self.down2(x1)
        c3, x3 = self.down3(x2)
        c4, x4 = self.down4(x3)
        bridge = self.bridge(x4)
        x5 = self.up1(bridge, self.concat1([c1, c2, c3, c4], []))
        x6 = self.up2(x5, self.concat2([c1, c2, c3], [bridge]))
        x7 = self.up3(x6, self.concat3([c1, c2], [x5, bridge]))
        x8 = self.up4(x7, self.concat4([c1], [x6, x5, bridge]))
        return self.outc(x8)
