import math

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetBlockType(Enum):
    UP = 0
    DOWN = 1


"""
Swish Activation Function.
"""
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


"""
Adaptive Group Normalization: (AdaGN).
"""
class AdaGN(nn.Module):
    def __init__(self, emb_dim, out_dim, groups=32):
        super().__init__()

        self.y_scale = nn.Linear(emb_dim, out_dim)
        self.y_shift = nn.Linear(emb_dim, out_dim)

        self.group_norm = nn.GroupNorm(groups, out_dim)

    def forward(self, x, emb):
        x_gn = self.group_norm(x)
        
        y_scale = self.y_scale(emb)
        y_scale = y_scale[:, :, None, None]

        y_shift = self.y_scale(emb)
        y_shift = y_shift[:, :, None, None]

        x = y_scale * x_gn + y_shift
        return x


"""
Time Embedding (Positional Sinusodial) + Conditional Info e.g one-hot encoding.
"""
class ConditionalEmbedding(nn.Module):
    def __init__(self, time_dim, cond_dim=None):
        super().__init__()

        # Number of dimensions in the embedding.
        self.time_dim = time_dim
        self.cond_dim = cond_dim

        self.time_layer = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            Swish(),
            nn.Linear(self.time_dim, self.time_dim),
            Swish(),
            nn.Linear(self.time_dim, self.time_dim),
            Swish(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        if self.cond_dim is not None:
            self.cond_layer = nn.Sequential(
                nn.Linear(self.cond_dim, self.time_dim),
                Swish(),
                nn.Linear(self.time_dim, self.time_dim),
                Swish(),
                nn.Linear(self.time_dim, self.time_dim),
                Swish(),
                nn.Linear(self.time_dim, self.time_dim)
            )
        else:
            self.cond_layer = None

    def forward(self, t, cond=None):
        # Sinusoidal Position embeddings.
        half_dim = self.time_dim // 2
        time_emb = math.log(10_000) / (half_dim - 1)
        time_emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=t.device) * -time_emb
        )
        time_emb = t[:, None] * time_emb[None, :]
        time_emb = torch.cat((time_emb.sin(), time_emb.cos()), dim=1)
        
        time_emb = self.time_layer(time_emb)
        
        cond_emb = 0
        if self.cond_layer is not None:
            cond_emb = self.cond_layer(cond)
        emb = time_emb + cond_emb
        return emb


"""
Attention Block. Similar to transformer's mulit-head attention block.
"""
class AttentionBlock(nn.Module):
    def __init__(self, channels, heads=1, d_k=None, groups=32):
        super().__init__()

        # Number of dimensions in each head.
        if d_k is None:
            d_k = channels
        
        # Normalization Layer.
        self.norm = nn.GroupNorm(groups, channels)

        # Projections for query(q), key(k) and values(v).
        self.projection = nn.Linear(channels, heads * d_k * 3)

        # Linear Layer for final transformation.
        self.output = nn.Linear(heads * d_k, channels)

        # Scale for dot-product attention.
        self.scale = d_k ** -0.5

        self.heads = heads
        self.d_k = d_k
    
    def forward(self, x, t=None):
        # Not used, but kept in the arguments because for the attention
        # layer function signature to match with ResidualBlock.
        _ = t
        
        # Batch Size (N). Channel (C), Height (H), Width (W)
        N, C, H, W = x.shape
        # Reshape to (N, seq, channels)
        x = x.view(N, C, -1).permute(0, 2, 1)

        # Get concatenated qkv and reshape to: (N, seq, heads, 3 * d_k)
        qkv = self.projection(x).view(N, -1, self.heads, 3 * self.d_k)
        
        # Split q, k, and v @ (N, seq, heads, d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Calculate scaled dot-product: QK^T / sqrt(d_k)
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        
        # Softmax along sequence dimension.
        attn = attn.softmax(dim=1)

        # Multiply attn by v.
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        
        # Reshape to N, seq, heads * d_k.
        res = res.reshape(N, -1, self.heads * self.d_k)
        
        # Transform to N, seq, channels.
        res = self.output(res)

        # Add skip connection.
        res += x

        # Reshape to: N, C, H, W
        res = res.permute(0, 2, 1).view(N, C, H, W)
        return res


"""
Upsample using ConvTranspose2d.
"""
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1),
            Swish(),)

    def forward(self, x, emb=None):
        _ = emb
        x = self.conv_layer(x)
        return x


"""
Downsample using Conv2d.
"""
class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1),
            Swish())

    def forward(self, x, emb=None):
        _ = emb
        x = self.conv_layer(x)
        return x


"""
Conditional Embedded Conv. Block.
"""
class UNet_ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_activation=True,
            emb_dim=None,
            groups=32):
        super().__init__()
        
        conv_list = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1)
        ]
        if use_activation:
            conv_list.append(Swish())
        self.conv_layer = nn.Sequential(*conv_list)

        if emb_dim is not None:
            self.adagn = AdaGN(
                emb_dim,
                out_channels,
                groups=groups)

    def forward(self, x, emb=None):
        x = self.conv_layer(x)
        
        if emb is not None:
            x = self.adagn(x, emb)
        return x


"""
Residual Conv. Block.
"""
class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_activation=True,
            emb_dim=None,
            groups=32):
        super().__init__()

        self.conv_block_1 = UNet_ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            use_activation=use_activation,
            emb_dim=emb_dim,
            groups=groups)
        self.conv_block_2 = UNet_ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            use_activation=use_activation,
            emb_dim=emb_dim,
            groups=groups)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, emb=None):
        init_x = x
        x = self.conv_block_1(x, emb)
        x = self.conv_block_2(x, emb)
        x = x + self.shortcut(init_x)
        return x


"""
U-Net Block.
"""
class UNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_dim,
        num_resnet_blocks=1,
        use_attn=True,
        num_heads=1,
        dim_per_head=None,
        groups=32,
        block_type=UNetBlockType.DOWN):
        super().__init__()

        hidden_channels = in_channels

        self.res_layers = nn.ModuleList()
        self.attn_layers = nn.ModuleList()
        for layer_count in range(num_resnet_blocks):
            self.res_layers.append(
                ResidualBlock(
                    in_channels=in_channels if layer_count == 0 else hidden_channels,
                    out_channels=hidden_channels,
                    emb_dim=emb_dim))
            if use_attn:
                self.attn_layers.append(
                    AttentionBlock(
                        channels=hidden_channels,
                        heads=num_heads,
                        d_k=dim_per_head,
                        groups=groups))
            else:
                self.attn_layers.append(nn.Identity())

        if block_type == UNetBlockType.DOWN:
            self.out_layer = DownsampleBlock(
                in_channels=hidden_channels,
                out_channels=out_channels)
        elif block_type == UNetBlockType.UP:
            self.out_layer = UpsampleBlock(
                in_channels=hidden_channels,
                out_channels=out_channels)

    def forward(self, x, emb=None):
        for res_layer, attn_layer in zip(self.res_layers, self.attn_layers):
            x = res_layer(x, emb)
            x = attn_layer(x)
        x = self.out_layer(x, emb)
        return x
