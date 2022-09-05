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

        self.init_layer = UNetResidualBlock(
            in_channels=3,
            hidden_channels=current_channel,
            out_channels=current_channel,
            time_channel=self.min_channel,
            use_group_norm=False)

        # Down Section.
        self.down_layers = nn.ModuleList()
        
        while current_dim > self.min_dim:
            self.down_layers.append(
                UNetResidualBlock(
                    in_channels=current_channel,
                    hidden_channels=current_channel,
                    out_channels=self.max_channel if current_channel * 2 > self.max_channel else current_channel * 2,
                    time_channel=self.min_channel))

            prev_channel_list.append(current_channel)

            current_channel = self.max_channel if current_channel * 2 > self.max_channel else current_channel * 2
            current_dim /= 2
        
        prev_channel_list.append(current_channel)
        
        # Middle Section.
        # 4 -> 4.
        self.middle_layer = MiddleBlock(
            channels=current_channel,
            time_channel=self.min_channel)

        # Up Section.
        index = 0
        prev_channel_list.reverse()
        
        self.up_layers = nn.ModuleList()
        while current_dim < img_dim:
            self.up_layers.append(
                UNetResidualBlock(
                    in_channels=prev_channel_list[index] * 2,
                    hidden_channels=prev_channel_list[index],
                    out_channels=prev_channel_list[index + 1],
                    time_channel=self.min_channel),
            )
            index += 1
            current_dim *= 2
        
        self.out_to_rgb = nn.Sequential(
            ConvBlock(
                in_channels=prev_channel_list[index],
                out_channels=prev_channel_list[index]),
            ConvBlock_toRGB(prev_channel_list[index])
        )
    def forward(self, x, t=0):
        # Time Embedding (x + t).
        t = self.time_emb(t)
        # t = t[:, :, None, None]

        # Init Layer
        x = self.init_layer(x, t)

        prev_out = []

        # Down Section.
        for down_layer in self.down_layers:
            x = down_layer(x, t)
            prev_out.append(x)
            x = F.max_pool2d(x, kernel_size=2)
        
        # Middle Section.
        x = self.middle_layer(x, t)
        
        # Up Section.
        for up_layer in self.up_layers:
            x = F.interpolate(x, scale_factor=2)
            prev_in = prev_out.pop()
            
            x = torch.cat((x, prev_in), dim=1)
            x = up_layer(x, t)

        x = self.out_to_rgb(x)
        return x

