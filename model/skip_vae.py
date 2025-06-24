'''
    This file add skipped connect between encoder and decoder
'''

from typing import Optional, Tuple
import torch
import torch.nn as nn
from diffusers.models.autoencoders.vae import Decoder, Encoder,is_torch_version
from diffusers.models.unet_2d_blocks import DownEncoderBlock2D,UpDecoderBlock2D

class Skip_DownEncoderBlock2d(DownEncoderBlock2D):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0, num_layers: int = 1, resnet_eps: float = 0.000001, resnet_time_scale_shift: str = "default", resnet_act_fn: str = "swish", resnet_groups: int = 32, resnet_pre_norm: bool = True, output_scale_factor: float = 1, add_downsample: bool = True, downsample_padding: int = 1):
        super().__init__(in_channels, out_channels, dropout, num_layers, resnet_eps, resnet_time_scale_shift, resnet_act_fn, resnet_groups, resnet_pre_norm, output_scale_factor, add_downsample, downsample_padding)
        
    def forward(self, hidden_states: torch.FloatTensor, scale: float = 1.0) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None, scale=scale)
        # print(hidden_states.shape)
        hidden_states_bfsample = hidden_states.detach().clone()
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, scale)
        # print(hidden_states.shape)
        
        return hidden_states, hidden_states_bfsample

class Skip_UpDecoderBlock2D(UpDecoderBlock2D):
    def __init__(self, in_channels: int, out_channels: int, resolution_idx: int | None = None, dropout: float = 0, num_layers: int = 1, resnet_eps: float = 0.000001, resnet_time_scale_shift: str = "default", resnet_act_fn: str = "swish", resnet_groups: int = 32, resnet_pre_norm: bool = True, output_scale_factor: float = 1, add_upsample: bool = True, temb_channels: int | None = None):
        super().__init__(in_channels, out_channels, resolution_idx, dropout, num_layers, resnet_eps, resnet_time_scale_shift, resnet_act_fn, resnet_groups, resnet_pre_norm, output_scale_factor, add_upsample, temb_channels)
        self.out_channels = out_channels
        
    def add_skip_connect(self):
        # self.skip_conv = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1,bias=False)
        # # self.concat_conv = nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1)
        # self.skip_concat_conv = nn.Sequential(
        #                                 nn.Conv2d(self.out_channels*2, self.out_channels, kernel_size=3, padding=1, bias=False),
        #                                 nn.GroupNorm(32, self.out_channels),
        #                                 nn.SiLU()
        #                             )
        
        self.skip_conv = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1,bias=False)
        self.skip_concat_conv = nn.Sequential(
                                        nn.Conv2d(self.out_channels*2, self.out_channels//4, kernel_size=1, bias=False),
                                        nn.GroupNorm(8, self.out_channels//4),
                                        nn.SiLU(),
                                        nn.Conv2d(self.out_channels//4, self.out_channels, kernel_size=1, bias=False),
                                        nn.GroupNorm(16, self.out_channels),
                                        nn.SiLU()
                                    )
    def forward(
        self, hidden_states: torch.FloatTensor, skip_feature = None, temb: Optional[torch.FloatTensor] = None, scale: float = 1.0
    ) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb, scale=scale)
        # print(hidden_states.shape, skip_feature.shape)
        fused = torch.cat((hidden_states, self.skip_conv(skip_feature)), dim=1)
        residual = self.skip_concat_conv(fused)
        hidden_states = hidden_states + residual  # 残差连接
        # hidden_states = self.skip_concat_conv(torch.cat((hidden_states,self.skip_conv(skip_feature)), dim=1))
        
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states

class Skip_Encoder(Encoder):
    def __init__(self, 
                 in_channels: int = 3, 
                 out_channels: int = 3, 
                 down_block_types: Tuple[str, ...] = ..., 
                 block_out_channels: Tuple[int, ...] = ..., 
                 layers_per_block: int = 2, 
                 norm_num_groups: int = 32, 
                 act_fn: str = "silu", 
                 double_z: bool = True, 
                 mid_block_add_attention=True):
        super().__init__(in_channels, 
                         out_channels,
                         down_block_types,
                         block_out_channels, 
                         layers_per_block, 
                         norm_num_groups, 
                         act_fn, 
                         double_z, 
                         mid_block_add_attention)
        # down
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = Skip_DownEncoderBlock2d(
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                dropout=0.0,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                downsample_padding=0,
            )
            self.down_blocks.append(down_block)
        
    def forward(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        r"""The forward method of the `Encoder` class."""
        temp = []
        sample = self.conv_in(sample)

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            if is_torch_version(">=", "1.11.0"):
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(down_block), sample, use_reentrant=False
                    )
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, use_reentrant=False
                )
            else:
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(down_block), sample)
                # middle
                sample = torch.utils.checkpoint.checkpoint(create_custom_forward(self.mid_block), sample)

        else:
            # down
            for down_block in self.down_blocks:
                # temp.append(sample)
                sample, skip_feature = down_block(sample)
                temp.append(skip_feature)
                

            # middle
            # temp.append(sample)
            sample = self.mid_block(sample)
            temp.append(sample)
            
        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample,temp
        
class Skip_Decoder(Decoder):
    def __init__(self, 
                 in_channels: int = 3, 
                 out_channels: int = 3, 
                 up_block_types: Tuple[str, ...] = ..., 
                 block_out_channels: Tuple[int, ...] = ..., 
                 layers_per_block: int = 2, 
                 norm_num_groups: int = 32, 
                 act_fn: str = "silu", 
                 norm_type: str = "group", 
                 mid_block_add_attention=True):
        super().__init__(in_channels, 
                         out_channels, 
                         up_block_types, 
                         block_out_channels, 
                         layers_per_block, 
                         norm_num_groups, 
                         act_fn, 
                         norm_type, 
                         mid_block_add_attention)
        
        temb_channels = in_channels if norm_type == "spatial" else None
        # up
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = Skip_UpDecoderBlock2D(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                temb_channels=temb_channels,
                resnet_time_scale_shift=norm_type,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel
        
    def add_skip_connect(self):
        for up_block in self.up_blocks:
            up_block.add_skip_connect()
            
    def forward(
        self,
        sample: torch.FloatTensor,
        skip_feature_list = None,
        latent_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        r"""The forward method of the `Decoder` class."""
        sample = self.conv_in(sample)
        
        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if self.training and self.gradient_checkpointing:
            raise RuntimeError("当前处于训练模式且启用了 gradient_checkpointing，但该模式不被支持，请关闭 gradient_checkpointing 或切换到 eval 模式。")
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    latent_embeds,
                    use_reentrant=False,
                )
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        latent_embeds,
                        use_reentrant=False,
                    )
            else:
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, latent_embeds
                )
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample, latent_embeds)
        else:
            # middle
            sample = self.mid_block(sample, latent_embeds)
            sample = sample.to(upscale_dtype)
            skip_feature = skip_feature_list.pop()
            # print(sample.shape, skip_feature.shape)
            # up
            for up_block in self.up_blocks:
                skip_feature = skip_feature_list.pop()
                # print("block", sample.shape, skip_feature.shape)
                sample = up_block(sample, skip_feature, latent_embeds)

        # skip_feature = skip_feature_list.pop()
        # print("final_block", sample.shape, skip_feature.shape)
        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample
    
if __name__ == "__main__":
    encoder = Skip_Encoder(down_block_types= ["DownEncoderBlock2D",
                                          "DownEncoderBlock2D",
                                          "DownEncoderBlock2D",
                                          "DownEncoderBlock2D"],
                            block_out_channels=[128, 256, 512, 512], 
                            out_channels= 4
                            )
    input_img = torch.randn(1,3,256,256)
    sample,temp = encoder(input_img)
    for feature in temp:
        if isinstance(feature, str):
            print(feature)
        else:
            print(feature.shape)
    print(len(temp))
    decoder = Skip_Decoder(up_block_types= ["UpDecoderBlock2D",
                                          "UpDecoderBlock2D",
                                          "UpDecoderBlock2D",
                                          "UpDecoderBlock2D"],
                       block_out_channels=[128, 256, 512, 512], 
                       in_channels=4,
                       out_channels= 3
                       )
    decoder.add_skip_connect()
    sample_input = torch.randn(1,4,32,32)
    out = decoder(sample_input, temp)
    # for feature in in_temp:
    #     if isinstance(feature, str):
    #         print(feature)
    #     else:
    #         print(feature.shape)