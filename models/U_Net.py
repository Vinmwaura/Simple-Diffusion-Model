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
            in_channel=3,
            out_channel=3,
            num_layers=5,
            attn_layers=[2, 3, 4],
            time_channel=64,
            min_channel=128,
            max_channel=512,
            image_recon=False):

        super().__init__()
        
        # Asserts to ensure params are valid to prevent wierd issues.
        assert isinstance(num_layers, int)
        assert isinstance(attn_layers, list)
        assert num_layers > 1
        for attn_layer in attn_layers:
            assert isinstance(attn_layer, int)
            assert attn_layer > 0
            assert attn_layer < num_layers
        
        self.image_recon = image_recon
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
        if time_channel is not None:
            self.time_emb = TimeEmbedding(self.time_channel)
        else:
            self.time_emb = None
        
        # Down Section.
        self.down_layers = nn.ModuleList()
        self.down_layers.append(
            ConvBlock(
                in_channels=in_channel,
                out_channels=min_channel,
                use_activation=False))

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
        self.middle_layer_pre = UNetBlock(
            in_channels=channel_layers[-1],
            out_channels=channel_layers[-1],
            time_channel=self.time_channel,
            use_attn=True,
            block_type=UNetBlockType.MIDDLE)
        self.middle_layer = nn.Sequential(
            ConvBlock(
                in_channels=channel_layers[-1],
                out_channels=channel_layers[-1]),
            ConvBlock(
                in_channels=channel_layers[-1],
                out_channels=channel_layers[-1]))
        self.middle_layer_post = UNetBlock(
            in_channels=channel_layers[-1],
            out_channels=channel_layers[-1],
            time_channel=self.time_channel,
            use_attn=True,
            block_type=UNetBlockType.MIDDLE)

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
        out_layers = []
        out_layers.append(
            ConvBlock(
                in_channels=channel_layers[0],
                out_channels=channel_layers[0],
                use_activation=True))
        out_layers.append(
            ConvBlock(
                in_channels=channel_layers[0],
                out_channels=out_channel,
                use_activation=False))
        if image_recon:
            out_layers.append(nn.Tanh())
        
        self.out_layers = nn.Sequential(*out_layers)

    def custom_load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            # Skip loading mismatched weights, in cases of weight changes.
            if (own_state[name].shape != param.data.shape):
                print(f"Skipped: {name}")
                continue
            if isinstance(param, torch.nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def forward(self, x, t=None):
        prev_out = []

        if self.time_emb is not None:
            # Time Embedding (t).
            t_emb = self.time_emb(t)
        else:
            t_emb = None

        # Down Section.
        x = self.down_layers[0](x)
        for down_layer in self.down_layers[1:]:
            x = down_layer(x, t_emb)
            prev_out.append(x)

        # Middle Section.
        x = self.middle_layer_pre(x, t_emb)
        z = self.middle_layer(x)
        x = self.middle_layer_post(z, t_emb)

        # Up Section.
        for up_layer in self.up_layers:
            prev_in = prev_out.pop()
            x = torch.cat((x, prev_in), dim=1)
            x = up_layer(x, t_emb)
            
        x = self.out_layers(x)
        return x
