import torch
import torch.nn as nn
import math
from timm.layers import trunc_normal_


class FFCA(nn.Module):
    """
    What Changed

    Positional encoding (_make_pos_enc)
    Adds 2D sinusoidal embeddings to both aerial & SITS tokens → improves
    spatial alignment.

    Learnable gating (self.gates)
    Instead of always doing aer_flat + fused, we learn a weight alpha ∈ (0,1)
    per level:

    Cross-attention fusion between aerial and SITS features
    - Adds 2D positional encodings
    - Adds learnable gating to balance fusion
    """

    def __init__(
        self, aer_channels_list, sits_channels_list, num_heads=8, pos_enc=True
    ):
        super().__init__()

        self.levels = len(aer_channels_list)
        assert self.levels == len(sits_channels_list), "Feature levels must match"

        self.pos_enc = pos_enc

        # Project SITS features to match aerial feature channels
        self.sits_projs = nn.ModuleList(
            [
                nn.Conv2d(sits_ch, aer_ch, kernel_size=1)
                for aer_ch, sits_ch in zip(aer_channels_list, sits_channels_list)
            ]
        )

        # Cross-attention layers
        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=aer_ch, num_heads=num_heads, batch_first=True
                )
                for aer_ch in aer_channels_list
            ]
        )
        # Add a small Feed-Forward Network (FFN) after the attention (and norm)
        # learn nonlinear combinations of fused channels
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(aer_ch, aer_ch * 4),
                    nn.GELU(),
                    nn.Linear(aer_ch * 4, aer_ch),
                )
                for aer_ch in aer_channels_list
            ]
        )

        # Normalization
        self.norms = nn.ModuleList(
            [nn.LayerNorm(aer_ch) for aer_ch in aer_channels_list]
        )

        # Learnable gates for adaptive fusion
        self.gates = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in aer_channels_list]
        )

    def _make_pos_enc(self, H, W, C, device):
        """2D sinusoidal positional encoding"""
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )
        pos_y = y.flatten().unsqueeze(1)  # (H*W, 1)
        pos_x = x.flatten().unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, C, 2, device=device) * -(math.log(10000.0) / C)
        )
        pe = torch.zeros(H * W, C, device=device)
        pe[:, 0::2] = torch.sin(pos_x * div_term)
        pe[:, 1::2] = torch.cos(pos_y * div_term)
        return pe  # (H*W, C)

    def forward(self, aer_feats, sits_feats):
        fused_feats = []

        for i in range(self.levels):
            B, C_aer, H, W = aer_feats[i].shape

            # Project SITS features
            sits_proj = self.sits_projs[i](sits_feats[i])  # (B, C_aer, H, W)

            # Flatten for attention
            aer_flat = aer_feats[i].flatten(2).permute(0, 2, 1)  # (B, H*W, C_aer)
            sits_flat = sits_proj.flatten(2).permute(0, 2, 1)  # (B, H*W, C_aer)

            # Add positional encoding
            # Flatten aerial and SITS
            aer_flat = (
                aer_feats[i].flatten(2).permute(0, 2, 1)
            )  # (B, H_aer*W_aer, C_aer)
            sits_proj = self.sits_projs[i](sits_feats[i])
            sits_flat = sits_proj.flatten(2).permute(
                0, 2, 1
            )  # (B, H_sits*W_sits, C_aer)

            # Add positional encodings separately
            if self.pos_enc:
                pos_aer = self._make_pos_enc(
                    aer_feats[i].shape[2],
                    aer_feats[i].shape[3],
                    C_aer,
                    aer_feats[i].device,
                )
                pos_sits = self._make_pos_enc(
                    sits_proj.shape[2], sits_proj.shape[3], C_aer, sits_proj.device
                )

                aer_flat = aer_flat + pos_aer.unsqueeze(0).expand(
                    B, -1, -1
                )  # match (B, H_aer*W_aer, C)
                sits_flat = sits_flat + pos_sits.unsqueeze(0).expand(B, -1, -1)

            # Cross-attention
            attn_out, _ = self.attentions[i](
                aer_flat, sits_flat, sits_flat
            )  # (B, H*W, C_aer)

            # Residual + LayerNorm (Post-LN)
            # Adaptive gating: aerial + alpha * fused
            alpha = torch.sigmoid(self.gates[i])

            #  LN + residual connection after attentions
            fs_mhsa = self.norms[i](aer_flat + alpha * attn_out)

            # LN + residual connection after MLPs
            fused = self.norms[i](fs_mhsa + self.mlps[i](fs_mhsa))

            # Reshape back
            fused = fused.permute(0, 2, 1).contiguous().view(B, C_aer, H, W)

            fused_feats.append(fused)

        return fused_feats
