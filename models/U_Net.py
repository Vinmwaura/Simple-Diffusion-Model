import torch
import torch.nn as nn

from .custom_layers import *


"""
U-Net Architecture.
"""
class U_Net(nn.Module):
    def __init__(
            self,
            num_resnet_blocks=5,
            in_channel=3,
            out_channel=3,
            time_dim=64,
            cond_dim=None,
            num_layers=5,
            attn_layers=[2, 3, 4],
            num_heads=1,
            dim_per_head=None,
            groups=32,
            min_channel=128,
            max_channel=512,
            image_recon=False):
        super().__init__()
        
        # Checks to ensure params are valid to prevent issues.
        if not isinstance(num_layers, int) or not isinstance(attn_layers, list):
            raise TypeError("Invalid type!")
        
        if num_layers < 1:
            raise ValueError("Invalid num layer value!")
        for attn_layer in attn_layers:
            if not isinstance(attn_layer, int):
                raise ValueError("Invalid type in attention layer!")
            if attn_layer < 0 or attn_layer >= num_layers:
                raise ValueError("Invalid Attention Layer values!")

        # Channels to use in each layer, doubles until max_channel is reached.
        channel_layers = [min_channel]
        channel = min_channel
        for _ in range(num_layers):
            channel = channel * 2
            channel_layers.append(
                max_channel if channel > max_channel else channel)

        # Conditional Embedding Layer.
        if time_dim is not None:
            # Adds Conditional to Time Embedding if any (Seems to work).
            self.cond_emb = ConditionalEmbedding(time_dim, cond_dim)
        else:
            self.cond_emb = None
        
        self.in_layer = nn.Sequential(
            UNet_ConvBlock(
                in_channel,
                channel_layers[0],
                use_activation=True,
                emb_dim=None,),
            UNet_ConvBlock(
                in_channels=channel_layers[0],
                out_channels=channel_layers[0],
                use_activation=True,
                emb_dim=None,)
        )

        # Down Section.
        self.down_layers = nn.ModuleList()
        for layer_count in range(0, num_layers, 1):
            self.down_layers.append(
                UNetBlock(
                    in_channels=channel_layers[layer_count],
                    out_channels=channel_layers[layer_count + 1],
                    emb_dim=time_dim,
                    num_resnet_blocks=num_resnet_blocks,
                    use_attn=layer_count in attn_layers,
                    num_heads=num_heads,
                    dim_per_head=dim_per_head,
                    groups=groups,
                    block_type=UNetBlockType.DOWN)
            )

        # Middle Section.
        self.middle_layer = nn.Sequential(
            UNet_ConvBlock(
                in_channels=channel_layers[-1],
                out_channels=channel_layers[-1],
                use_activation=True,
                emb_dim=None,),
            UNet_ConvBlock(
                in_channels=channel_layers[-1],
                out_channels=channel_layers[-1],
                use_activation=True,
                emb_dim=None,))

        # Up Section.
        self.up_layers = nn.ModuleList()
        for layer_count in range(num_layers - 1, -1, -1):
            self.up_layers.append(
                UNetBlock(
                    in_channels=channel_layers[layer_count + 1] * 2,   # Doubles channels
                    out_channels=channel_layers[layer_count],
                    emb_dim=time_dim,
                    num_resnet_blocks=num_resnet_blocks,
                    use_attn=layer_count in attn_layers,
                    num_heads=num_heads,
                    dim_per_head=dim_per_head,
                    groups=groups,
                    block_type=UNetBlockType.UP)
            )

        # Output Section.
        out_layers_list = []
        out_layers_list.append(
            UNet_ConvBlock(
                in_channels=channel_layers[0],
                out_channels=channel_layers[0],
                use_activation=True,
                emb_dim=None,))
        out_layers_list.append(
            UNet_ConvBlock(
                in_channels=channel_layers[0],
                out_channels=out_channel,
                use_activation=False,
                emb_dim=None,))
        if image_recon:
            out_layers_list.append(nn.Tanh())
        
        self.out_layers = nn.Sequential(*out_layers_list)

    def custom_load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print(f"No Layer found: {name}, skipping")
                continue
            # Skip loading mismatched weights, in cases of weight changes.
            if (own_state[name].shape != param.data.shape):
                print(f"Skipped: {name}")
                continue
            if isinstance(param, torch.nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def forward(self, x, t=None, cond=None):
        prev_out = []

        if self.cond_emb is not None:
            # Time + Cond Embedding.
            cond_emb = self.cond_emb(t, cond)
        else:
            cond_emb = None
        
        # Down Section.
        x = self.in_layer(x)
        for down_layer in self.down_layers:
            x = down_layer(x, cond_emb)
            prev_out.append(x)

        # Middle Section.
        x = self.middle_layer(x)

        # Up Section.
        for up_layer in self.up_layers:
            prev_in = prev_out.pop()
            x = torch.cat((x, prev_in), dim=1)
            x = up_layer(x, cond_emb)

        # Returns duffusion output.
        x_out = self.out_layers(x)
        return x_out
