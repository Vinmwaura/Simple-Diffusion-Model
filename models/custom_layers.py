import math

from enum import Enum

import torch
import torch.nn as nn


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
Time Embedding (Positional Sinusodial) like in Transformers.
"""
class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        # Number of dimensions in the embedding.
        self.embedding_dim = embedding_dim

        self.fc_layer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            Swish(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        

    def forward(self, t):
        # Sinusoidal Position embeddings.
        half_dim = self.embedding_dim // 2
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb
        )
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = self.fc_layer(emb)
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
Conv. Block.
"""
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_group_norm=False, groups=32):
        super().__init__()
        if use_group_norm:
            self.conv_layer = nn.Sequential(
                nn.GroupNorm(groups, in_channels),
                Swish(),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1),
            )
        else:
            self.conv_layer = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1),
                Swish(),
            )

    def forward(self, in_data):
        out = self.conv_layer(in_data)
        return out


"""
Upsample using ConvTranspose2d.
"""
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_layer = nn.Sequential(
            Swish(),
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1)
        )

    def forward(self, x, t):
        _ = t
        out = self.conv_layer(x)
        return out


"""
Downsample using Conv2d.
"""
class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv_layer = nn.Sequential(
            Swish(),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1)
        )

    def forward(self, x, t):
        _ = t
        out = self.conv_layer(x)
        return out


"""
Residual Conv. Block.
"""
class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            time_channel,
            use_group_norm=True):
        
        super().__init__()

        self.conv_block_1 = ConvBlock(
            in_channels,
            out_channels,
            use_group_norm=use_group_norm)
        
        self.conv_block_2 = ConvBlock(
            out_channels,
            out_channels,
            use_group_norm=True)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()
        
        self.time_emb = nn.Linear(time_channel, out_channels)
    
    def forward(self, x, t):
        init_x = x
        x = self.conv_block_1(x)
        
        # Time Embedding.
        t = self.time_emb(t)[:, :, None, None]
        x = x + t
        
        x = self.conv_block_2(x)
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
            time_channel,
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
            time_channel=time_channel)
        
        if self.block_type == UNetBlockType.DOWN:
            self.out_layer = DownsampleBlock(
                in_channels=hidden_channels,
                out_channels=out_channels)
        elif self.block_type == UNetBlockType.MIDDLE:
            self.out_layer = ResidualBlock(
                in_channels=hidden_channels,
                out_channels=out_channels,
                time_channel=time_channel)
        elif self.block_type == UNetBlockType.UP:
            self.out_layer = UpsampleBlock(
                in_channels=hidden_channels,
                out_channels=out_channels)

    def forward(self, x, t):
        x = self.in_layer(x, t)
        x = self.attn_layer(x)
        x = self.out_layer(x, t)
        return x
