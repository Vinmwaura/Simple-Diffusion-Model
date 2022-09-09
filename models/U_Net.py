import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .custom_layers import *


"""
Simpler U-Net Architecture.
"""
class U_Net(nn.Module):
    def __init__(self, img_dim=64):
        super().__init__()

        self.min_dim = 4
        self.max_dim = 1024

        if img_dim <= self.min_dim or img_dim >= self.max_dim or img_dim % 4 != 0:
            raise Exception(f"Image dimension must be between {self.min_dim:,} and {self.max_dim:,} and be multiple of 4")

        self.min_channel = 64
        self.max_channel = 512
        
        # Time Embedding Layer.
        self.time_emb = TimeEmbedding(self.min_channel)
        
        current_dim = img_dim
        current_channel = self.min_channel

        prev_channel_list = []
        
        self.init_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=current_channel,
                kernel_size=3,
                stride=1,
                padding=1),
            Swish(),
            ConvBlock(
                in_channels=current_channel,
                out_channels=current_channel,
                use_group_norm=True)
        )

        # Down Section.
        self.down_layers = nn.ModuleList()
        
        while current_dim > self.min_dim:
            self.down_layers.append(
                UNetBlock(
                    in_channels=current_channel,
                    out_channels=self.max_channel if current_channel * 2 > self.max_channel else current_channel * 2,
                    time_channel=self.min_channel,
                    use_attn=current_dim <= 16,
                    block_type=UNetBlockType.DOWN))

            prev_channel_list.append(current_channel)

            current_channel = self.max_channel if current_channel * 2 > self.max_channel else current_channel * 2
            current_dim /= 2
        
        prev_channel_list.append(current_channel)
        
        # Middle Section.
        # 4 -> 4.
        self.middle_layer = UNetBlock(
            in_channels=current_channel,
            out_channels=current_channel,
            time_channel=self.min_channel,
            use_attn=True,
            block_type=UNetBlockType.MIDDLE)

        # Up Section.
        index = 0
        prev_channel_list.reverse()
        
        self.up_layers = nn.ModuleList()
        while current_dim < img_dim:
            self.up_layers.append(
                UNetBlock(
                    in_channels=prev_channel_list[index] * 2,
                    out_channels=prev_channel_list[index + 1],
                    time_channel=self.min_channel,
                    use_attn=current_dim <= 16,
                    block_type=UNetBlockType.UP),
            )
            index += 1
            current_dim *= 2

        self.out_layers = nn.Sequential(
            ConvBlock(
                in_channels=prev_channel_list[index],
                out_channels=prev_channel_list[index],
                use_group_norm=True),
            Swish(),
            nn.Conv2d(
                in_channels=prev_channel_list[index],
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1),
        )

    def forward(self, x, t=0):
        prev_out = []

        # Init Layer
        x = self.init_layer(x)

        # Time Embedding (x + t).
        t_emb = self.time_emb(t)

        # Down Section.
        for down_layer in self.down_layers:
            x = down_layer(x, t_emb)
            prev_out.append(x)

        # Middle Section.
        x = self.middle_layer(x, t_emb)
        
        # Up Section.
        for up_layer in self.up_layers:
            prev_in = prev_out.pop()
            x = torch.cat((x, prev_in), dim=1)
            x = up_layer(x, t_emb)

        # Approximate noise added to image.
        eps_approx = self.out_layers(x)
        return eps_approx
