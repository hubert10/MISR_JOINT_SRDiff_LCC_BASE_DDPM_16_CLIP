import math
import torch
from collections import OrderedDict
from collections.abc import Sequence
from functools import partial
from typing import Any, Callable, Optional
from einops import rearrange
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.utils import _log_api_usage_once
from utils.utils_models import PositionalEncoder


def _get_conv_output_shape(
    input_size: tuple[int, int], kernel_size: int, stride: int, padding: int
) -> tuple[int, int]:
    return (
        (input_size[0] - kernel_size + 2 * padding) // stride + 1,
        (input_size[1] - kernel_size + 2 * padding) // stride + 1,
    )


def _make_block_input_shapes(
    input_size: tuple[int, int], n_blocks: int
) -> list[tuple[int, int]]:
    """Util function to check that the input size is correct for a MaxVit configuration."""
    shapes = []
    block_input_shape = _get_conv_output_shape(input_size, 3, 2, 1)
    for _ in range(n_blocks):
        block_input_shape = _get_conv_output_shape(block_input_shape, 3, 2, 1)
        shapes.append(block_input_shape)
    return shapes


def _get_relative_position_index(height: int, width: int) -> torch.Tensor:
    coords = torch.stack(
        torch.meshgrid([torch.arange(height), torch.arange(width)], indexing="ij")
    )
    coords_flat = torch.flatten(coords, 1)
    relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += height - 1
    relative_coords[:, :, 1] += width - 1
    relative_coords[:, :, 0] *= 2 * width - 1
    return relative_coords.sum(-1)


class MBConv(nn.Module):
    """MBConv: Mobile Inverted Residual Bottleneck.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expansion_ratio (float): Expansion ratio in the bottleneck.
        squeeze_ratio (float): Squeeze ratio in the SE Layer.
        stride (int): Stride of the depthwise convolution.
        activation_layer (Callable[..., nn.Module]): Activation function.
        norm_layer (Callable[..., nn.Module]): Normalization function.
        p_stochastic_dropout (float): Probability of stochastic depth.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: float,
        squeeze_ratio: float,
        stride: int,
        activation_layer: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        p_stochastic_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        proj: Sequence[nn.Module]
        self.proj: nn.Module

        should_proj = stride != 1 or in_channels != out_channels
        if should_proj:
            proj = [
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)
            ]
            if stride == 2:
                proj = [nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)] + proj  # type: ignore
            self.proj = nn.Sequential(*proj)
        else:
            self.proj = nn.Identity()  # type: ignore

        mid_channels = int(out_channels * expansion_ratio)
        sqz_channels = int(out_channels * squeeze_ratio)

        if p_stochastic_dropout:
            self.stochastic_depth = StochasticDepth(p_stochastic_dropout, mode="row")  # type: ignore
        else:
            self.stochastic_depth = nn.Identity()  # type: ignore

        _layers = OrderedDict()
        _layers["pre_norm"] = norm_layer(in_channels)
        _layers["conv_a"] = Conv2dNormActivation(
            in_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation_layer=activation_layer,
            norm_layer=norm_layer,
            inplace=None,
        )
        _layers["conv_b"] = Conv2dNormActivation(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            activation_layer=activation_layer,
            norm_layer=norm_layer,
            groups=mid_channels,
            inplace=None,
        )
        _layers["squeeze_excitation"] = SqueezeExcitation(
            mid_channels, sqz_channels, activation=nn.SiLU
        )
        _layers["conv_c"] = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=True,
        )

        self.layers = nn.Sequential(_layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, C, H, W].
        Returns:
            Tensor: Output tensor with expected layout of [B, C, H / stride, W / stride].
        """
        res = self.proj(x)
        x = self.stochastic_depth(self.layers(x))
        return res + x


class RelativePositionalMultiHeadAttention(nn.Module):
    """Relative Positional Multi-Head Attention.

    Args:
        feat_dim (int): Number of input features.
        head_dim (int): Number of features per head.
        max_seq_len (int): Maximum sequence length.
    """

    def __init__(
        self,
        feat_dim: int,
        head_dim: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()

        if feat_dim % head_dim != 0:
            raise ValueError(
                f"feat_dim: {feat_dim} must be divisible by head_dim: {head_dim}"
            )

        self.n_heads = feat_dim // head_dim
        self.head_dim = head_dim
        self.size = int(math.sqrt(max_seq_len))
        self.max_seq_len = max_seq_len

        self.to_qkv = nn.Linear(feat_dim, self.n_heads * self.head_dim * 3)
        self.scale_factor = feat_dim**-0.5

        self.merge = nn.Linear(self.head_dim * self.n_heads, feat_dim)
        self.relative_position_bias_table = nn.parameter.Parameter(
            torch.empty(
                ((2 * self.size - 1) * (2 * self.size - 1), self.n_heads),
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "relative_position_index",
            _get_relative_position_index(self.size, self.size),
        )
        # initialize with truncated normal the bias
        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def get_relative_positional_bias(self) -> torch.Tensor:
        bias_index = self.relative_position_index.view(-1)  # type: ignore
        relative_bias = self.relative_position_bias_table[bias_index].view(self.max_seq_len, self.max_seq_len, -1)  # type: ignore
        relative_bias = relative_bias.permute(2, 0, 1).contiguous()
        return relative_bias.unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, G, P, D].
        Returns:
            Tensor: Output tensor with expected layout of [B, G, P, D].
        """
        B, G, P, D = x.shape
        H, DH = self.n_heads, self.head_dim

        qkv = self.to_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)
        k = k.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)
        v = v.reshape(B, G, P, H, DH).permute(0, 1, 3, 2, 4)

        k = k * self.scale_factor
        dot_prod = torch.einsum("B G H I D, B G H J D -> B G H I J", q, k)
        pos_bias = self.get_relative_positional_bias()

        dot_prod = F.softmax(dot_prod + pos_bias, dim=-1)

        out = torch.einsum("B G H I J, B G H J D -> B G H I D", dot_prod, v)
        out = out.permute(0, 1, 3, 2, 4).reshape(B, G, P, D)

        out = self.merge(out)
        return out


class SwapAxes(nn.Module):
    """Permute the axes of a tensor."""

    def __init__(self, a: int, b: int) -> None:
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = torch.swapaxes(x, self.a, self.b)
        return res


class WindowPartition(nn.Module):
    """
    Partition the input tensor into non-overlapping windows.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, p: int) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, C, H, W].
            p (int): Number of partitions.
        Returns:
            Tensor: Output tensor with expected layout of [B, H/P, W/P, P*P, C].
        """
        B, C, H, W = x.shape
        P = p
        # chunk up H and W dimensions
        x = x.reshape(B, C, H // P, P, W // P, P)
        x = x.permute(0, 2, 4, 3, 5, 1)
        # colapse P * P dimension
        x = x.reshape(B, (H // P) * (W // P), P * P, C)
        return x


class WindowDepartition(nn.Module):
    """
    Departition the input tensor of non-overlapping windows into a feature volume of layout [B, C, H, W].
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, x: Tensor, p: int, h_partitions: int, w_partitions: int
    ) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, (H/P * W/P), P*P, C].
            p (int): Number of partitions.
            h_partitions (int): Number of vertical partitions.
            w_partitions (int): Number of horizontal partitions.
        Returns:
            Tensor: Output tensor with expected layout of [B, C, H, W].
        """
        B, G, PP, C = x.shape
        P = p
        HP, WP = h_partitions, w_partitions

        # split P * P dimension into 2 P tile dimensionsa
        x = x.reshape(B, HP, WP, P, P, C)
        # permute into B, C, HP, P, WP, P
        x = x.permute(0, 5, 1, 3, 2, 4)
        # reshape into B, C, H, W
        x = x.reshape(B, C, HP * P, WP * P)
        return x


class PartitionAttentionLayer(nn.Module):
    """
    Layer for partitioning the input tensor into non-overlapping windows
      and applying attention to each window.

    Args:
        in_channels (int): Number of input channels.
        head_dim (int): Dimension of each attention head.
        partition_size (int): Size of the partitions.
        partition_type (str): Type of partitioning to use. Can be either "grid" or "window".
        grid_size (Tuple[int, int]): Size of the grid to partition the input tensor into.
        mlp_ratio (int): Ratio of the  feature size expansion in the MLP layer.
        activation_layer (Callable[..., nn.Module]): Activation function to use.
        norm_layer (Callable[..., nn.Module]): Normalization function to use.
        attention_dropout (float): Dropout probability for the attention layer.
        mlp_dropout (float): Dropout probability for the MLP layer.
        p_stochastic_dropout (float): Probability of dropping out a partition.
    """

    def __init__(
        self,
        in_channels: int,
        head_dim: int,
        # partitioning parameters
        partition_size: int,
        partition_type: str,
        # grid size needs to be known at initialization time
        # because we need to know hamy relative offsets there are in the grid
        grid_size: tuple[int, int],
        mlp_ratio: int,
        activation_layer: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        attention_dropout: float,
        mlp_dropout: float,
        p_stochastic_dropout: float,
    ) -> None:
        super().__init__()

        self.n_heads = in_channels // head_dim
        self.head_dim = head_dim
        self.n_partitions = grid_size[0] // partition_size
        self.partition_type = partition_type
        self.grid_size = grid_size

        if partition_type not in ["grid", "window"]:
            raise ValueError("partition_type must be either 'grid' or 'window'")

        if partition_type == "window":
            self.p, self.g = partition_size, self.n_partitions
        else:
            self.p, self.g = self.n_partitions, partition_size

        self.partition_op = WindowPartition()
        self.departition_op = WindowDepartition()
        self.partition_swap = (
            SwapAxes(-2, -3) if partition_type == "grid" else nn.Identity()
        )
        self.departition_swap = (
            SwapAxes(-2, -3) if partition_type == "grid" else nn.Identity()
        )

        self.attn_layer = nn.Sequential(
            norm_layer(in_channels),
            # it's always going to be partition_size ** 2 because
            # of the axis swap in the case of grid partitioning
            RelativePositionalMultiHeadAttention(
                in_channels, head_dim, partition_size**2
            ),
            nn.Dropout(attention_dropout),
        )

        # pre-normalization similar to transformer layers
        self.mlp_layer = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels * mlp_ratio),
            activation_layer(),
            nn.Linear(in_channels * mlp_ratio, in_channels),
            nn.Dropout(mlp_dropout),
        )

        # layer scale factors
        self.stochastic_dropout = StochasticDepth(p_stochastic_dropout, mode="row")

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor with expected layout of [B, C, H, W].
        Returns:
            Tensor: Output tensor with expected layout of [B, C, H, W].
        """

        # Undefined behavior if H or W are not divisible by p
        # https://github.com/google-research/maxvit/blob/da76cf0d8a6ec668cc31b399c4126186da7da944/maxvit/models/maxvit.py#L766
        gh, gw = self.grid_size[0] // self.p, self.grid_size[1] // self.p
        torch._assert(
            self.grid_size[0] % self.p == 0 and self.grid_size[1] % self.p == 0,
            "Grid size must be divisible by partition size. Got grid size of {} and partition size of {}".format(
                self.grid_size, self.p
            ),
        )

        x = self.partition_op(x, self.p)
        x = self.partition_swap(x)
        x = x + self.stochastic_dropout(self.attn_layer(x))
        x = x + self.stochastic_dropout(self.mlp_layer(x))
        x = self.departition_swap(x)
        x = self.departition_op(x, self.p, gh, gw)

        return x


class MaxVitLayer(nn.Module):
    """
    MaxVit layer consisting of a MBConv layer followed by a PartitionAttentionLayer
    with `window` and a PartitionAttentionLayer with `grid`.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expansion_ratio (float): Expansion ratio in the bottleneck.
        squeeze_ratio (float): Squeeze ratio in the SE Layer.
        stride (int): Stride of the depthwise convolution.
        activation_layer (Callable[..., nn.Module]): Activation function.
        norm_layer (Callable[..., nn.Module]): Normalization function.
        head_dim (int): Dimension of the attention heads.
        mlp_ratio (int): Ratio of the MLP layer.
        mlp_dropout (float): Dropout probability for the MLP layer.
        attention_dropout (float): Dropout probability for the attention layer.
        p_stochastic_dropout (float): Probability of stochastic depth.
        partition_size (int): Size of the partitions.
        grid_size (Tuple[int, int]): Size of the input feature grid.
    """

    def __init__(
        self,
        # conv parameters
        in_channels: int,
        out_channels: int,
        squeeze_ratio: float,
        expansion_ratio: float,
        stride: int,
        # conv + transformer parameters
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        # transformer parameters
        head_dim: int,
        mlp_ratio: int,
        mlp_dropout: float,
        attention_dropout: float,
        p_stochastic_dropout: float,
        # partitioning parameters
        partition_size: int,
        grid_size: tuple[int, int],
    ) -> None:
        super().__init__()

        layers: OrderedDict = OrderedDict()

        # convolutional layer
        layers["MBconv"] = MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            expansion_ratio=expansion_ratio,
            squeeze_ratio=squeeze_ratio,
            stride=stride,
            activation_layer=activation_layer,
            norm_layer=norm_layer,
            p_stochastic_dropout=p_stochastic_dropout,
        )
        # attention layers, block -> grid
        layers["window_attention"] = PartitionAttentionLayer(
            in_channels=out_channels,
            head_dim=head_dim,
            partition_size=partition_size,
            partition_type="window",
            grid_size=grid_size,
            mlp_ratio=mlp_ratio,
            activation_layer=activation_layer,
            norm_layer=nn.LayerNorm,
            attention_dropout=attention_dropout,
            mlp_dropout=mlp_dropout,
            p_stochastic_dropout=p_stochastic_dropout,
        )
        layers["grid_attention"] = PartitionAttentionLayer(
            in_channels=out_channels,
            head_dim=head_dim,
            partition_size=partition_size,
            partition_type="grid",
            grid_size=grid_size,
            mlp_ratio=mlp_ratio,
            activation_layer=activation_layer,
            norm_layer=nn.LayerNorm,
            attention_dropout=attention_dropout,
            mlp_dropout=mlp_dropout,
            p_stochastic_dropout=p_stochastic_dropout,
        )
        self.layers = nn.Sequential(layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            Tensor: Output tensor of shape (B, C, H, W).
        """
        x = self.layers(x)
        return x


class MaxVitBlock(nn.Module):
    """
    A MaxVit block consisting of `n_layers` MaxVit layers.

     Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expansion_ratio (float): Expansion ratio in the bottleneck.
        squeeze_ratio (float): Squeeze ratio in the SE Layer.
        activation_layer (Callable[..., nn.Module]): Activation function.
        norm_layer (Callable[..., nn.Module]): Normalization function.
        head_dim (int): Dimension of the attention heads.
        mlp_ratio (int): Ratio of the MLP layer.
        mlp_dropout (float): Dropout probability for the MLP layer.
        attention_dropout (float): Dropout probability for the attention layer.
        p_stochastic_dropout (float): Probability of stochastic depth.
        partition_size (int): Size of the partitions.
        input_grid_size (Tuple[int, int]): Size of the input feature grid.
        n_layers (int): Number of layers in the block.
        p_stochastic (List[float]): List of probabilities for stochastic depth for each layer.
    """

    def __init__(
        self,
        # conv parameters
        in_channels: int,
        out_channels: int,
        squeeze_ratio: float,
        expansion_ratio: float,
        # conv + transformer parameters
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        # transformer parameters
        head_dim: int,
        mlp_ratio: int,
        mlp_dropout: float,
        attention_dropout: float,
        # partitioning parameters
        partition_size: int,
        input_grid_size: tuple[int, int],
        # number of layers
        n_layers: int,
        p_stochastic: list[float],
    ) -> None:
        super().__init__()
        if not len(p_stochastic) == n_layers:
            raise ValueError(
                f"p_stochastic must have length n_layers={n_layers}, got p_stochastic={p_stochastic}."
            )

        self.layers = nn.ModuleList()
        # account for the first stride of the first layer
        self.grid_size = _get_conv_output_shape(
            input_grid_size, kernel_size=3, stride=2, padding=1
        )

        for idx, p in enumerate(p_stochastic):
            stride = 2 if idx == 0 else 1
            self.layers += [
                MaxVitLayer(
                    in_channels=in_channels if idx == 0 else out_channels,
                    out_channels=out_channels,
                    squeeze_ratio=squeeze_ratio,
                    expansion_ratio=expansion_ratio,
                    stride=stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    mlp_dropout=mlp_dropout,
                    attention_dropout=attention_dropout,
                    partition_size=partition_size,
                    grid_size=self.grid_size,
                    p_stochastic_dropout=p,
                ),
            ]

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            Tensor: Output tensor of shape (B, C, H, W).
        """
        for layer in self.layers:
            x = layer(x)
        return x


class WeightedTemporalEncoder(nn.Module):
    """
    Temporal encoder for multi-level spatio-temporal features:
    Input: List of [ (B, T, C, H, W), ... ]
    Output: List of [ (B, C, H, W), ... ] after temporal encoding + pooling
    """

    def __init__(
        self, channels_list, num_heads=4, num_layers=2, mlp_ratio=4.0, dropout=0.1
    ):
        """
        Args:
            channels_list (List[int]): channel dimensions at each feature level
            num_heads (int): number of attention heads
            num_layers (int): number of transformer layers
        """
        super().__init__()

        self.levels = len(channels_list)

        # Build one Transformer encoder per level
        self.encoders = nn.ModuleList()
        self.attn_pools = nn.ModuleList()

        for c in channels_list:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=c,
                nhead=num_heads,
                dim_feedforward=int(c * mlp_ratio),
                dropout=dropout,
                batch_first=True,  # input: (B, T, C)
                norm_first=True,
            )
            self.encoders.append(
                nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            )
            self.attn_pools.append(nn.Linear(c, 1))  # temporal attention pooling

    def forward(self, x_list):
        """
        Args:
            x_list: list of feature tensors [ (B, T, C, H, W), ... ]

        Returns:
            fused_list: list of fused feature maps [ (B, C, H, W), ... ]
        """
        fused_list = []

        for i, x in enumerate(x_list):
            B, T, C, H, W = x.shape  # (B, T, C, H, W)

            # Reshape: (B, H, W, T, C) -> (B*H*W, T, C)
            x = x.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, T, C)

            # Transformer encoder
            x = self.encoders[i](x)  # (B*H*W, T, C)

            # Temporal attention pooling
            attn_scores = torch.softmax(self.attn_pools[i](x), dim=1)  # (B*H*W, T, 1)
            fused = torch.sum(attn_scores * x, dim=1)  # (B*H*W, C)

            # Reshape back: (B, C, H, W)
            fused = fused.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

            fused_list.append(fused)

        return tuple(fused_list)


class TConvFormer(nn.Module):
    """
    Time-series version of MaxViT.
    Input:  [B, T, C, H, W]
    Output: [B, T, C, H, W]
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        stem_channels: int,
        partition_size: int,
        block_channels: list[int],
        block_layers: list[int],
        head_dim: int,
        stochastic_depth_prob: float,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Callable[..., nn.Module] = nn.GELU,
        squeeze_ratio: float = 0.25,
        expansion_ratio: float = 4,
        mlp_ratio: int = 4,
        mlp_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_layers: int = 2,
        # remove classification head for pixel output
        num_classes: Optional[int] = None,
        input_channels: int = 4,  # <-- adjustable number of channels
        d_model=256,
        T=1000,
        n_head=8,
        positional_encoding=True,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)

        self.d_model = d_model

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=2
            )
        else:
            self.positional_encoder = None

        # stem
        self.stem = nn.Sequential(
            Conv2dNormActivation(
                input_channels,
                stem_channels,
                3,
                stride=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                bias=False,
                inplace=None,
            ),
            Conv2dNormActivation(
                stem_channels,
                stem_channels,
                3,
                stride=1,
                norm_layer=None,
                activation_layer=None,
                bias=True,
            ),
        )

        # compute block input sizes
        block_input_sizes = _make_block_input_shapes(input_size, len(block_channels))

        # self.temp_atts = PixelTemporalEncoder(embed_dim=stem_channels, num_heads=16, num_layers=2)

        self.temporal_encoder = WeightedTemporalEncoder(
            [64, 128, 256, 512], num_heads=4, num_layers=2
        )

        input_size = _get_conv_output_shape(
            input_size, kernel_size=3, stride=1, padding=1
        )
        # print("input_size:", input_size)

        self.partition_size = partition_size

        # blocks
        self.blocks = nn.ModuleList()
        in_channels = [stem_channels] + block_channels[:-1]
        out_channels = block_channels

        p_stochastic = np.linspace(0, stochastic_depth_prob, sum(block_layers)).tolist()
        p_idx = 0

        for in_channel, out_channel, num_layers in zip(
            in_channels, out_channels, block_layers
        ):
            self.blocks.append(
                MaxVitBlock(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    squeeze_ratio=squeeze_ratio,
                    expansion_ratio=expansion_ratio,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    mlp_dropout=mlp_dropout,
                    attention_dropout=attention_dropout,
                    partition_size=partition_size,
                    input_grid_size=input_size,
                    n_layers=num_layers,
                    p_stochastic=p_stochastic[p_idx : p_idx + num_layers],
                )
            )
            input_size = self.blocks[-1].grid_size
            p_idx += num_layers

        # optional classifier
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.LayerNorm(block_channels[-1]),
                nn.Linear(block_channels[-1], block_channels[-1]),
                nn.Tanh(),
                nn.Linear(block_channels[-1], num_classes, bias=False),
            )
        else:
            self.classifier = None

        self._init_weights()

    def forward(self, x: Tensor, batch_positions) -> Tensor:
        """
        Input:  [B, T, C, H, W]
        Output: [B, T, C, H, W]
        """
        B, T, C, H, W = x.shape

        # stem
        x_sp = [
            self.stem(x[:, i, :, :, :]) for i in range(T)
        ]  # input conv: B, C_in, H, W
        x_sp = torch.stack(x_sp, 1)  # expected output wieder: B,T,C_out,H,W

        Wh, Ww = x_sp.size(-2), x_sp.size(-1)

        x = x_sp.flatten(-2).transpose(-2, -1)

        # include time embedding
        time_embed = self.positional_encoder(batch_positions)  # torch.Size([5, 12, 64])

        # print("time_embed:", time_embed.shape)  # torch.Size([2, 4, 64])

        time_embed = torch.stack(
            [time_embed[:, i, :].unsqueeze(1).repeat(1, Wh * Ww, 1) for i in range(T)],
            1,
        )
        time_embed = time_embed.to(x.device)
        x = x + time_embed

        x = x_sp.reshape(
            x_sp.shape[0] * T, x_sp.shape[2], x_sp.shape[-2], x_sp.shape[-1]
        )
        out_feats = [x]

        # blocks
        for block in self.blocks:
            x = block(x)
            out_feats.append(x)

        # --- Reshape all block outputs back to [B, T, C, H, W] ---
        temp_feats = [
            f.view(B, T, f.shape[1], f.shape[2], f.shape[3]) for f in out_feats
        ]

        return self.temporal_encoder(temp_feats), temp_feats

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
