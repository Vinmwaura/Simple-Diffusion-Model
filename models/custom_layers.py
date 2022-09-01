import math

import torch
import torch.nn as nn


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
            nn.Linear(self.embedding_dim//2, self.embedding_dim),
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
Residual Block.
"""
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, n_groups=32):
        super().__init__()

        self.conv_blk_1 = nn.Sequential(
            nn.GroupNorm(n_groups, in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.conv_blk_2 = nn.Sequential(
            nn.GroupNorm(n_groups, out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        """
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        """
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
        
        self.time_emb = nn.Linear(time_channels, out_channels)
        

    def forward(self, x, t):
        init_x = x
        # First Conv Block.
        x = self.conv_blk_1(init_x)
        
        # Time Embedding.
        t = self.time_emb(t)[:, :, None, None]
        x = x + t

        # Second Conv Block. 
        x = self.conv_blk_2(x)
        
        # Residual Operation.
        x = x + self.shortcut(init_x)
        """
        h = self.conv1(self.act1(self.norm1(x)))
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.shortcut(x)
        """
        return x


"""
Upsample using ConvTranspose2d.
"""
class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            channels,
            channels,
            kernel_size=4,
            stride=2,
            padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


"""
Downsample using Conv2d
"""
class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


"""
U-Net DownBlock.
"""
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, has_attn):
        super().__init__()
        
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()
        
    def forward(self, x, t):
        x = self.res(x, t)
        x = self.attn(x)
        return x


"""
U-Net MiddleBlock.
"""
class MiddleBlock(nn.Module):
    def __init__(self, n_channels, time_channels):
        super().__init__()
        
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)        
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


"""
U-Net UpBlock.
"""
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, has_attn):
        super().__init__()
        
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()
        
    def forward(self, x, t):
        x = self.res(x, t)
        x = self.attn(x)
        return x
