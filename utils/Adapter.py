import torch
import torch.nn.functional as F
import math
import torch
import torch.nn as nn

from typing import Any, Optional, Union

class Adapter(nn.Module):
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = d_model if d_model is None else d_model
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, 64)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):

        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in': #  none
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out': #  none
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up
        return output


class Linear_Lora(nn.Module):
    def __init__(self,
                 r,
                 in_feature,
                 out_feature, 
                 lora_dropout=0.0,
                 lora_alpha=8,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = r
        self.lora_A = nn.Linear(in_features=in_feature, out_features=r, bias=False)
        self.lora_B = nn.Linear(in_features=r, out_features=out_feature, bias=False)
        self.dropout = nn.Dropout(p=lora_dropout)
        self.scaling = lora_alpha / r
        self.initialize()
        
    def initialize(self):
        with torch.no_grad():
            # nn.init.normal_(self.lora_A, std=1 / self.r)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
    

class Conv2d_Lora(nn.Module):
    def __init__(self,
                 r,
                 in_features,
                 out_features,
                 kernel_size,
                 stride,
                 padding,
                 lora_dropout=0.0,
                 lora_alpha=8,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_A = nn.Conv2d(in_features, r, kernel_size, stride, padding, bias=False)
        self.lora_B = nn.Conv2d(r, out_features, (1, 1), (1, 1), bias=False)
        self.dropout = nn.Dropout(p=lora_dropout)
        self.scaling = lora_alpha / r
        self.initialize()
        
    def initialize(self):
        with torch.no_grad():
            # nn.init.normal_(self.lora_A, std=1 / self.r)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling



# # TODO 这个是具体修改UNet的具体的Attention逻辑的函数，具体可以看https://zhouyifan.net/2024/01/27/20240123-SD-Attn/
# class MoeAttnProcessor2_0(AttnProcessor2_0):
#     def __init__(self, dispacher:SparseDispatcher, ):
#         super().__init__()
#         self.dispacher = dispacher
        
#     def __call__(self, attn, hidden_states, encoder_hidden_states = None, attention_mask = None, temb = None, scale = 1):
#         residual = hidden_states
#         if attn.spatial_norm is not None:
#             hidden_states = attn.spatial_norm(hidden_states, temb)

#         input_ndim = hidden_states.ndim

#         if input_ndim == 4:
#             batch_size, channel, height, width = hidden_states.shape
#             hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

#         batch_size, sequence_length, _ = (
#             hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
#         )

#         if attention_mask is not None:
#             attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
#             # scaled_dot_product_attention expects attention_mask shape to be
#             # (batch, heads, source_length, target_length)
#             attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

#         if attn.group_norm is not None:
#             hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

#         args = ()
#         query = attn.to_q(hidden_states, *args)

#         if encoder_hidden_states is None:
#             encoder_hidden_states = hidden_states
#         elif attn.norm_cross:
#             encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

#         key = attn.to_k(encoder_hidden_states, *args)
#         value = attn.to_v(encoder_hidden_states, *args)

#         inner_dim = key.shape[-1]
#         head_dim = inner_dim // attn.heads

#         query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

#         key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#         value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

#         # the output of sdp = (batch, num_heads, seq_len, head_dim)
#         # TODO: add support for attn.scale when we move to Torch 2.1
#         hidden_states = F.scaled_dot_product_attention(
#             query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
#         )

#         hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
#         hidden_states = hidden_states.to(query.dtype)

#         # linear proj
#         hidden_states = attn.to_out[0](hidden_states, *args)
#         # dropout
#         hidden_states = attn.to_out[1](hidden_states)

#         if input_ndim == 4:
#             hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

#         if attn.residual_connection:
#             hidden_states = hidden_states + residual

#         hidden_states = hidden_states / attn.rescale_output_factor

#         return hidden_states
