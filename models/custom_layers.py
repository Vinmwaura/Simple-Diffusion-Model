import math

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetBlockType(Enum):
    UP = 0
    MIDDLE = 1
    DOWN = 2


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
    def __init__(self, time_channels, out_channels, groups=32):
        
        super().__init__()

        self.y_scale = nn.Linear(time_channels, out_channels)
        self.y_shift = nn.Linear(time_channels, out_channels)
        self.group_norm = nn.GroupNorm(groups, out_channels)

    def forward(self, x, t):
        x_gn = self.group_norm(x)

        y_scale = self.y_scale(t)
        y_scale = y_scale[:, :, None, None]

        y_shift = self.y_scale(t)
        y_shift = y_shift[:, :, None, None]

        x = y_scale * x_gn + y_shift
        return x


"""
Time Embedding (Positional Sinusodial) like in Transformers.
"""
class TimeEmbedding(nn.Module):
    def __init__(self, time_dim, cond_dim=None):
        super().__init__()

        # Number of dimensions in the embedding.
        self.time_dim = time_dim
        self.cond_dim = cond_dim

        self.fc_layer = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            Swish(),
            nn.Linear(self.time_dim, self.time_dim),
            Swish(),
            nn.Linear(self.time_dim, self.time_dim),
            Swish(),
            nn.Linear(self.time_dim, self.time_dim),
            Swish(),
            nn.Linear(self.time_dim, self.time_dim),
            Swish(),
            nn.Linear(self.time_dim, self.time_dim),
            Swish(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        if self.cond_dim is not None:
            self.fc_layer_2 = nn.Linear(self.cond_dim, self.time_dim)
        else:
            self.cond_emb_layer = None

    def forward(self, t, cond=None):
        # Sinusoidal Position embeddings.
        half_dim = self.time_dim // 2
        time_emb = math.log(10_000) / (half_dim - 1)
        time_emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=t.device) * -time_emb
        )
        time_emb = t[:, None] * time_emb[None, :]
        time_emb = torch.cat((time_emb.sin(), time_emb.cos()), dim=1)
        
        time_emb = self.fc_layer(time_emb)
        
        cond_emb = 0
        if self.fc_layer_2 is not None:
            cond_emb = self.fc_layer_2(cond)
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
        res = res.view(N, -1, self.heads * self.d_k)
        
        # Transform to N, seq, channels.
        res = self.output(res)

        # Add skip connection.
        res += x

        # Reshape to: N, C, H, W
        res = res.permute(0, 2, 1).view(N, C, H, W)
        return res


"""
Time Embedded Conv. Block.
"""
class UNet_ConvBlock(nn.Module):
    def __init__(self, time_channels, in_channels, out_channels, groups=32):
        
        super().__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1),
            Swish())
        self.adagn = AdaGN(time_channels, out_channels, groups=groups)

    def forward(self, x, t):
        x = self.conv_layer(x)
        x = self.adagn(x, t)
        return x


"""
Conv. Block.
"""
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_activation=True):
        super().__init__()
        if use_activation:
            self.conv_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1),
                Swish())
        else:
            self.conv_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1))

    def forward(self, x, t=None):
        _ = t
        x = self.conv_layer(x)
        return x


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


    def forward(self, x, t=None):
        _ = t
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

    def forward(self, x, t=None):
        _ = t
        x = self.conv_layer(x)
        return x


"""
Residual Conv. Block.
"""
class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            time_dim=None):
        super().__init__()

        if time_dim is not None:
            self.conv_block_1 = UNet_ConvBlock(
                time_dim,
                in_channels,
                out_channels)
            self.conv_block_2 = UNet_ConvBlock(
                time_dim,
                out_channels,
                out_channels)
        else:
            self.conv_block_1 = ConvBlock(
                in_channels,
                out_channels)
            self.conv_block_2 = ConvBlock(
                out_channels,
                out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()
        
        # self.time_emb = nn.Linear(time_dim, out_channels)
    
    def forward(self, x, t=None):
        init_x = x
        x = self.conv_block_1(x, t)
        x = self.conv_block_2(x, t)
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
            time_dim,
            use_attn=True,
            block_type=UNetBlockType.MIDDLE, # up, middle, down
        ):

        super().__init__()

        hidden_channels = in_channels

        self.use_attn = use_attn
        self.block_type = block_type

        if self.use_attn:
            self.attn_layer = AttentionBlock(hidden_channels)
        else:
            self.attn_layer = nn.Identity()

        self.in_layer = ResidualBlock(
            in_channels=in_channels,
            out_channels=hidden_channels,
            time_dim=time_dim)
        
        if self.block_type == UNetBlockType.DOWN:
            self.out_layer = DownsampleBlock(
                in_channels=hidden_channels,
                out_channels=out_channels)
        elif self.block_type == UNetBlockType.MIDDLE:
            self.out_layer = ResidualBlock(
                in_channels=hidden_channels,
                out_channels=out_channels,
                time_dim=time_dim)
        elif self.block_type == UNetBlockType.UP:
            self.out_layer = UpsampleBlock(
                in_channels=hidden_channels,
                out_channels=out_channels)

    def forward(self, x, t=None):
        x = self.in_layer(x, t)
        x = self.attn_layer(x)
        x = self.out_layer(x, t)
        return x
