import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig

from utils.moe_utils import SparseDispatcher, Moe_layer, replace_layers_byname
from model.autoencoder_kl import AutoencoderKL,Skip_AutoencoderKL
from model.unet_2d_condition import UNet2DConditionModel

def initialize_vae(pretrained_model, lora_rank, num_experts, dispatcher:SparseDispatcher, add_place="encoder"):
    vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae")
    vae.requires_grad_(False)
    vae.train()
    
    l_target_modules_encoder = []
    l_grep = ["conv1","conv2","conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n: 
            continue
        for pattern in l_grep:
            if pattern in n and (add_place in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
            elif ("encoder" in add_place) and ('quant_conv' in n) and ('post_quant_conv' not in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
    
    for name in l_target_modules_encoder:
        replace_layers_byname(vae, "{}".format(name), [nn.Conv2d, nn.Linear], Moe_layer, dispatcher=dispatcher, r=lora_rank, expert_nums=num_experts,lora_alpha=8,name=name)
        
    return vae, l_target_modules_encoder

def initialize_skip_vae(pretrained_model, lora_rank):
    vae = Skip_AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae")
    vae.requires_grad_(False)
    vae.train()
    
    l_target_modules_encoder = []
    l_grep = ["conv1","conv2","conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n: 
            continue
        for pattern in l_grep:
            if pattern in n:
                l_target_modules_encoder.append(n.replace(".weight",""))
            elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
    print(len(l_target_modules_encoder))
    lora_conf_encoder = LoraConfig(r=lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    vae.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    vae.decoder.add_skip_connect()
    
    return vae, l_target_modules_encoder

def initialize_vae_lora(pretrained_model, lora_rank):
    vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae")
    vae.requires_grad_(False)
    vae.train()
    
    l_target_modules_encoder = []
    l_grep = ["conv1","conv2","conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n: 
            continue
        for pattern in l_grep:
            if pattern in n and ("encoder" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
            elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
    
    lora_conf_encoder = LoraConfig(r=lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    vae.add_adapter(lora_conf_encoder, adapter_name="default_encoder")

    return vae, l_target_modules_encoder

def initialize_unet(pretrained_model, lora_rank, num_experts, dispatcher:SparseDispatcher, return_lora_module_names=False, pretrained_model_name_or_path=None):
    unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break

    for name in l_target_modules_encoder:
        replace_layers_byname(unet, "{}".format(name), [nn.Conv2d, nn.Linear], Moe_layer, dispatcher=dispatcher, r=lora_rank, expert_nums=num_experts,lora_alpha=8,name=name)
    for name in l_target_modules_decoder:
        replace_layers_byname(unet, "{}".format(name), [nn.Conv2d, nn.Linear], Moe_layer, dispatcher=dispatcher, r=lora_rank, expert_nums=num_experts,lora_alpha=8,name=name)
    for name in l_modules_others:
        replace_layers_byname(unet, "{}".format(name), [nn.Conv2d, nn.Linear], Moe_layer, dispatcher=dispatcher, r=lora_rank, expert_nums=num_experts,lora_alpha=8,name=name)        
    
    return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others

def initialize_unet_only(pretrained_model, lora_rank, return_lora_module_names=False, pretrained_model_name_or_path=None):
    unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break

    lora_conf_encoder = LoraConfig(r=lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    lora_conf_decoder = LoraConfig(r=lora_rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder)
    lora_conf_others = LoraConfig(r=lora_rank, init_lora_weights="gaussian",target_modules=l_modules_others)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")

    return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others
