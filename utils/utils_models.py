import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers import trunc_normal_, DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights
from einops import rearrange
from abc import ABCMeta, abstractmethod
import warnings


def resize(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    warning=True,
):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is 'x+1' and "
                        f"out size {(output_h, output_w)} is 'nx+1'"
                    )
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def tensor_to_list(x, n_channels, nbts):
    x_new = []
    for i in range(nbts):
        x_cur = x[:, i * n_channels : i * n_channels + n_channels, :, :]
        x_new.append(x_cur)
    return x_new


def list_to_tensor(x, dim=0):
    x_new = torch.cat(x, dim=dim)

    return x_new


def concat(x1, x2):
    diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
    diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
    x = torch.cat([x2, x1], dim=1)
    return x


def padding(im, patch_size, fill_value=0):
    # make the image sizes divisible by patch_size
    H, W = im.size(2), im.size(3)
    pad_h, pad_w = 0, 0
    if H % patch_size > 0:
        pad_h = patch_size - (H % patch_size)
    if W % patch_size > 0:
        pad_w = patch_size - (W % patch_size)
    im_padded = im
    if pad_h > 0 or pad_w > 0:
        im_padded = F.pad(im, (0, pad_w, 0, pad_h), value=fill_value)
    return im_padded


def unpadding(y, target_size):
    H, W = target_size
    H_pad, W_pad = y.size(2), y.size(3)
    # crop predictions on extra pixels coming from padding
    extra_h = H_pad - H
    extra_w = W_pad - W
    if extra_h > 0:
        y = y[:, :, :-extra_h]
    if extra_w > 0:
        y = y[:, :, :, :-extra_w]
    return y


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


# for segmenter
def time_embedding(dates, nbts, out_dim):  # btsz, out_dim, D):
    # February 29th handled as March 1st:
    dates[dates == 229] = 301
    month = torch.div(dates, 100, rounding_mode="trunc")

    day = dates - month * 100

    dpm = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for i in range(13):
        month[month == i] = sum(dpm[:i])

    doy = month + day
    tau = 10000

    B = len(dates)
    te_out = torch.zeros(B, nbts, out_dim)

    for i in range(out_dim):
        te_out[:, :, i] = torch.sin(
            doy / (tau ** (2 * i / out_dim) + math.pi / 2 * (i % 2))
        )

    return te_out

    # for i in range(out_dim):
    #     te = torch.sin(doy[j,i] / (tau**(2*i/nbts) + math.pi/2 * (i % 2)))
    # te_out = torch.ones(btsz, nbts * out_dim, D)
    # for i in range(nbts):
    #     for j in range(btsz):
    #         te_out[j, i*out_dim : (i+1)*out_dim, :] = torch.sin(doy[j,i] / (tau**(2*i/nbts) + math.pi/2 * (i % 2)))
    # return te_out


def resize_pos_embed(posemb, grid_old_shape, grid_new_shape, num_extra_tokens):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    posemb_tok, posemb_grid = (
        posemb[:, :num_extra_tokens],
        posemb[0, num_extra_tokens:],
    )
    if grid_old_shape is None:
        gs_old_h = int(math.sqrt(len(posemb_grid)))
        gs_old_w = gs_old_h
    else:
        gs_old_h, gs_old_w = grid_old_shape

    gs_h, gs_w = grid_new_shape
    posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


class PositionalEncoder(nn.Module):
    def __init__(self, d, T=1000, repeat=None, offset=0):
        super(PositionalEncoder, self).__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(
            T,
            2
            * torch.div(
                torch.arange(offset, offset + d).float(), 2, rounding_mode="floor"
            )
            / d,
        )
        self.updated_location = False

    def forward(self, batch_positions):
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)
            self.updated_location = True
        sinusoid_table = (
            batch_positions[:, :, None] / self.denom[None, None, :]
        )  # B x T x C
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )
        return sinusoid_table
