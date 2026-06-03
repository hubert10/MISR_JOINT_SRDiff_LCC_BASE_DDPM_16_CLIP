import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from einops import rearrange

# BASE classes


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        """
        channel: nb of input channels
        """
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, SE=False):
        super().__init__()
        if SE:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                SELayer(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.double_conv(x)


class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.
    ATTENTION: THIS HAS BEEN GREATLY SIMPLIFIED FOR USAGE WITH SWIN!
    CHECK mmcv\cnn\bricks\conv_module.py FOR ORIGINAL MODULE DEFINITION!
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        inplace=False,
    ):
        super().__init__()

        # build convolution layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        self.bn = nn.BatchNorm2d(out_channels)
        # build activation layer
        self.activate = nn.ReLU(inplace=inplace)

        # Use msra init by default
        self.init_weights()

    def init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, activate=True, norm=True):
        x_conv = self.conv(x)
        x_normed = self.bn(x_conv)
        x_out = self.activate(x_normed)
        return x_out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, volumetric=False, nbts=1, nbots=1):
        super(OutConv, self).__init__()
        if volumetric:
            if nbots == 1:
                self.conv = nn.Conv3d(
                    in_channels, out_channels, kernel_size=(nbts, 1, 1), padding=0
                )
            else:
                self.conv = nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, padding=0
                )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, SE=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )

        self.conv = DoubleConv(in_channels, out_channels, SE)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        if diffX != 0 or diffY != 0:
            x1 = F.pad(
                x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Up3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels,
        out_channels,
        bilinear=True,
        depthwise=False,
        kernels_per_layer=1,
        pseudo=False,
    ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=(1, 2, 2), mode="trilinear", align_corners=True
            )
            self.up_all = nn.Upsample(
                scale_factor=(2, 2, 2), mode="trilinear", align_corners=True
            )
        else:
            self.up = nn.ConvTranspose3d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )

        self.conv = DoubleConv3D(
            in_channels, out_channels, depthwise, kernels_per_layer, pseudo
        )

    def forward(self, x1, x2):
        # x2 ist aus encoder
        # print('size x2 :', x2.size())
        # print('x1 size: ', x1.size())
        if x2.size()[2] == x1.size()[2]:
            x1 = self.up(x1)
        else:
            x1 = self.up_all(x1)
        # input is CHW
        # diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        # diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DoubleConv3D(nn.Module):
    """(3D conv - BN - relu)*2"""

    def __init__(
        self,
        in_channels,
        out_channels,
        depthwise=False,
        kernels_per_layer=1,
        pseudo=False,
    ):
        super().__init__()
        if depthwise:
            self.double_conv = nn.Sequential(
                DepthwiseSeparableConv3D(in_channels, kernels_per_layer, out_channels),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                DepthwiseSeparableConv3D(out_channels, kernels_per_layer, out_channels),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
            )
        elif pseudo:
            self.double_conv = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)
                ),  # spatial conv.
                nn.Conv3d(
                    out_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)
                ),  # temporal conv.
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(
                    out_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)
                ),
                nn.Conv3d(
                    out_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)
                ),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.double_conv(x)


class DepthwiseSeparableConv3D(nn.Module):
    def __init__(self, in_channels, kernels_per_layer, out_channels):
        super(DepthwiseSeparableConv3D, self).__init__()
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=3,
            padding=1,
            groups=in_channels,
        )
        self.pointwise = nn.Conv3d(
            in_channels * kernels_per_layer, out_channels, kernel_size=1
        )

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class LayerNormGeneral(nn.Module):
    r"""General LayerNorm for different situations.
    source: https://github.com/sail-sg/metaformer/blob/main/metaformer_baselines.py#L532

    Args:
        affine_shape (int, list or tuple): The shape of affine weight and bias.
            Usually the affine_shape=C, but in some implementation, like torch.nn.LayerNorm,
            the affine_shape is the same as normalized_dim by default.
            To adapt to different situations, we offer this argument here.
        normalized_dim (tuple or list): Which dims to compute mean and variance.
        scale (bool): Flag indicates whether to use scale or not.
        bias (bool): Flag indicates whether to use scale or not.

        We give several examples to show how to specify the arguments.

        LayerNorm (https://arxiv.org/abs/1607.06450):
            For input shape of (B, *, C) like (B, N, C) or (B, H, W, C),
                affine_shape=C, normalized_dim=(-1, ), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, ), scale=True, bias=True.

        Modified LayerNorm (https://arxiv.org/abs/2111.11418)
            that is idental to partial(torch.nn.GroupNorm, num_groups=1):
            For input shape of (B, N, C),
                affine_shape=C, normalized_dim=(1, 2), scale=True, bias=True;
            For input shape of (B, H, W, C),
                affine_shape=C, normalized_dim=(1, 2, 3), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, 2, 3), scale=True, bias=True.

        For the several metaformer baslines,
            IdentityFormer, RandFormer and PoolFormerV2 utilize Modified LayerNorm without bias (bias=False);
            ConvFormer and CAFormer utilizes LayerNorm without bias (bias=False).
    """

    def __init__(
        self, affine_shape=None, normalized_dim=(-1,), scale=True, bias=True, eps=1e-5
    ):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x
