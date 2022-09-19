import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .custom_layers import *


"""
U-Net Architecture.
"""
class U_Net(nn.Module):
    def __init__(
            self,
            num_layers=5,
            attn_layers=[3, 4],
            time_channel=64,
            min_channel=64,
            max_channel=512):
        super().__init__()
        
        # Asserts to ensure params are valid to prevent wierd issues.
        assert isinstance(num_layers, int)
        assert isinstance(attn_layers, list)
        assert num_layers > 1
        for attn_layer in attn_layers:
            assert isinstance(attn_layer, int)
            assert attn_layer > 0
            assert attn_layer < num_layers
        
        self.time_channel = time_channel
        
        self.min_channel = min_channel
        self.max_channel = max_channel

        # Channels to use in each layer, doubles until max_channel is reached.
        channel_layers = [self.min_channel]
        channel = self.min_channel
        for _ in range(num_layers):
            channel = channel * 2
            channel_layers.append(self.max_channel if channel > self.max_channel else channel)

        # Time Embedding Layer.
        self.time_emb = TimeEmbedding(self.time_channel)

        # Images to featch maps.
        self.init_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=channel_layers[0],
                kernel_size=3,
                stride=1,
                padding=1),
            Swish(),
            ConvBlock(
                in_channels=channel_layers[0],
                out_channels=channel_layers[0],
                use_group_norm=True))
        
        # Down Section.
        self.down_layers = nn.ModuleList()
        for layer_count in range(0, num_layers, 1):
            self.down_layers.append(
                UNetBlock(
                    in_channels=channel_layers[layer_count],
                    out_channels=channel_layers[layer_count + 1],
                    time_channel=self.time_channel,
                    use_attn=layer_count in attn_layers,
                    block_type=UNetBlockType.DOWN
                )
            )

        # Middle Section.
        self.middle_layer = UNetBlock(
            in_channels=channel_layers[-1],
            out_channels=channel_layers[-1],
            time_channel=self.time_channel,
            use_attn=True,
            block_type=UNetBlockType.MIDDLE)

        # Hack to get older models to work, comment out when training from scratch.
        attn_layers.append(2)

        # Up Section.
        self.up_layers = nn.ModuleList()
        for layer_count in range(num_layers - 1, -1, -1):
            self.up_layers.append(
                UNetBlock(
                    in_channels=channel_layers[layer_count + 1] * 2,   # Doubles channels
                    out_channels=channel_layers[layer_count],
                    time_channel=self.time_channel,
                    use_attn=layer_count in attn_layers,
                    block_type=UNetBlockType.UP
                )
            )
        
        # Output Section.
        self.out_layers = nn.Sequential(
            ConvBlock(
                in_channels=channel_layers[0],
                out_channels=channel_layers[0],
                use_group_norm=True),
            Swish(),
            nn.Conv2d(
                in_channels=channel_layers[0],
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
