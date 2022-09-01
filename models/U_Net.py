import torch
import torch.nn as nn

from custom_layers import *


"""
U-Net based model as defined in DDPM.
"""
class U_Net(nn.Module):
    def __init__(
            self,
            img_channel=3,
            init_channel=64,
            channel_mults=[1, 2, 2, 4],
            is_attn=[False, False, True, True],
            num_blocks=2):

        super().__init__()

        # Number of layers in down and up layers in UNet.
        num_resolutions = len(channel_mults)

        # Project image into feature map.
        self.image_proj = nn.Conv2d(
            img_channel,
            init_channel,
            kernel_size=3,
            padding=1)
        
        # Time Embedding Layer.
        self.time_emb = TimeEmbedding(init_channel)

        in_channel = init_channel
        
        # Down Section.
        down_list = []
        
        for index in range(num_resolutions):
            # Multiply channel size with multiplers for each layer.
            out_channel = in_channel * channel_mults[index]

            # Adds multiple block in each layer.
            for _ in range(num_blocks):
                down_list.append(
                    DownBlock(
                        in_channel,
                        out_channel,
                        init_channel,
                        is_attn[index])
                )

                in_channel = out_channel

            # Downsample all resolutions except the last.
            if index < num_resolutions - 1:
                down_list.append(DownBlock(in_channel))

        self.down_layers = nn.ModuleList(down_list)

        # Middle Section.
        self.middle_layer = MiddleBlock(out_channel, init_channel)

        # Up Section.
        up_list = []

        in_channel = out_channel

        for index_reversed in reversed(range(num_resolutions)):
            out_channel = in_channel
            for _ in range(num_blocks):
                up_list.append(
                    UpBlock(
                        in_channel,
                        out_channel,
                        init_channel,
                        is_attn[index_reversed])
                )
            out_channel = in_channel // channel_mults[index_reversed]
            
            up_list.append(
                UpBlock(
                    in_channel,
                    out_channel,
                    init_channel,
                    is_attn[index_reversed]))
            in_channels = out_channel

            if index_reversed > 0:
                self.up_layers.append(Upsample(in_channel))
        
        self.up_layers = nn.ModuleList(up_list)
        
        # Final Normalization and Conv Layer.
        self.final_layer = nn.Sequential(
            nn.GroupNorm(8, init_channel),
            Swish(),
            nn.Conv2d(in_channels, img_channel, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        prev_out = []
        
        # Initial Layer: Image Projection and time embedding.
        x = self.image_proj(x)
        t = self.time_emb(t)

        prev_out.append(x)

        # Down Section.
        for down_layer in self.down_layers:
            x = down_layer(x, t)
            prev_out.append(x)
        
        # Middle Section.
        x = self.middle_layer(x)

        # Up Section.
        for up_layer in self.up_layers:
            if isinstance(up_layer, Upsample):
                x = up_layer(x, t)
            else:
                prev_down_out = prev_out.pop()
                x = torch.cat((x, prev_down_out), dim=1)
            
            x = up_layer(x, t)

        # Final Layer.
        x = self.final_layer(x)
        return x
