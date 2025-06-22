import gc
from typing import Optional, Tuple
import torch
import torch.nn as nn
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from model.autoencoder_kl import Skip_AutoencoderKL,AutoencoderKL
from peft import LoraConfig


class VAEEncoderOnly(nn.Module):
    def __init__(self, pretrained_model, lora_rank, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  
        vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae")
        vae.requires_grad_(False)
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
        
        
        self.encoder = vae.encoder
        self.config = vae.config
        self.quant_conv = vae.quant_conv
        # self.post_quant_conv = vae.post_quant_conv
        
        del vae               # 删除整个 vae 对象
        torch.cuda.empty_cache()  # 清理显存缓存（可选）
        gc.collect()              # 触发 Python 垃圾回收（可选）
    
    def forward(self, c_t):
        h = self.encoder(c_t)
        moments = self.quant_conv(h.to(dtype=self.quant_conv.weight.dtype))
        posterior = DiagonalGaussianDistribution(moments)
        latent = AutoencoderKLOutput(latent_dist=posterior).latent_dist.sample() * self.config.scaling_factor
        
        return latent
    

if __name__ == "__main__":
    encoder = VAEEncoderOnly(pretrained_model="/home/gongzihang/workspace/Final_mix/pretrained/sd", lora_rank=4)
    for k,v in encoder.named_parameters():
        v.requires_grad_(True)
    input_img = torch.randn(1,3,256,256) 
    output = torch.randn(1,4,32,32) 
    latent =  encoder(input_img)
    import torch.nn.functional as F
    loss = F.mse_loss(latent.float(), output.float(), reduction="mean")
    loss.backward()
    for k,v in encoder.named_parameters():
        print(k,"--", v.grad is not None)
    