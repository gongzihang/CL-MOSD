import os
import sys
sys.path.append(os.getcwd())
import types
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from torchvision import transforms
from peft import LoraConfig
import copy

# init vlm model
from ram.models.ram_lora import ram
from ram import inference_ram as inference
from model.add_Moe import initialize_skip_vae
from model.unet_2d_condition import UNet2DConditionModel
from model.promptembed import Promept_embed
from model.router import Router
from model.diffusion import Diffusion_EVAE, Diffusion_EVAE_All, Diffusion_EVAE_Decoder, Diffusion_EVAE_Encoder, Infer_Diffusion_EVAE

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
        
    def to(self, device):  
        for k in self.shadow:
            self.shadow[k] = self.shadow[k].to(device)

class MoeDiffusion(nn.Module):
    def __init__(self, pretrained_model_path, lora_rank=4, num_experts=5, train_vae=True, task_num=5, router_embed_dim=32, top_k=2, noise_epsilon=0.01, degradation_channels=3,
                 vae_path=None, train_vae_part=None, use_ema=True, ema_decay=0.999, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.router_list = nn.ModuleList()
        for _ in range(task_num):
            self.router_list.append(Router(expert_nums=num_experts,
                                           embed_dim=router_embed_dim,
                                           top_k=top_k,
                                           noise_epsilon=noise_epsilon,
                                           degradation_channels=degradation_channels))
        # print(f"[DEBUG] the train_vae_part is {train_vae_part} || Type : {type(train_vae_part)} ")
        if train_vae_part == None:
            print(f"[INFO] the train_vae_part is {train_vae_part} || Load diffusion Diffusion_EVAE")
            self.diffusion = Diffusion_EVAE(pretrained_model_path=pretrained_model_path,
                                            lora_rank=lora_rank,
                                            num_experts=num_experts,
                                            vae_path=vae_path,
                                            train_vae=train_vae)  
        elif train_vae_part == "Decoder":
            print(f"[INFO] the train_vae_part is {train_vae_part} || Load diffusion Diffusion_EVAE_Decoder")
            self.diffusion = Diffusion_EVAE_Decoder(pretrained_model_path=pretrained_model_path,
                                                    lora_rank=lora_rank,
                                                    num_experts=num_experts,
                                                    vae_path=vae_path,
                                                    train_vae=train_vae)
        elif train_vae_part == "Encoder":
            print(f"[INFO] the train_vae_part is {train_vae_part} || Load diffusion Diffusion_EVAE_Encoder")
            self.diffusion = Diffusion_EVAE_Encoder(pretrained_model_path=pretrained_model_path,
                                                    lora_rank=lora_rank,
                                                    num_experts=num_experts,
                                                    vae_path=vae_path,
                                                    train_vae=train_vae)
            self.train_vae_encoder = True
        elif train_vae_part == "All":
            print(f"[INFO] the train_vae_part is {train_vae_part} || Load diffusion Diffusion_EVAE_All")
            self.diffusion = Diffusion_EVAE_All(pretrained_model_path=pretrained_model_path,
                                                    lora_rank=lora_rank,
                                                    num_experts=num_experts,
                                                    vae_path=vae_path,
                                                    train_vae=train_vae)
            self.train_vae_encoder = True
        
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        # self.diffusion.load_evae(load_path=vae_path)
        
    def add_noise(self,latents_pred, noise, timesteps):
        return self.diffusion.noise_scheduler.add_noise(latents_pred, noise, timesteps)
    
    def set_train(self, task_id=0, frozen=[]):
        self.router_list.requires_grad_(False)
        self.diffusion.requires_grad_(False)
        # self.vae_encoder.requires_grad_(False)
        # self.vae.requires_grad_(False)
        
        self.router_list[task_id].requires_grad_(True)
        self.diffusion.set_train(frozen=frozen)
        
        if self.use_ema:
            self.ema = EMA(self, decay=self.ema_decay)
            self.ema.register()
    
    def forward(self, generate_turn=False,lq=None, prompt_embeds=None, task_id=None, 
                    kl_turn=False, latents_pred=None, noisy_latents=None, timesteps=None,noise_pred_fix=None,
                    diff_turn=False):
        if (generate_turn):
            if self.train_vae_encoder:
                clean, encoded_control, skip_feature_list = self.diffusion.evae_prepare(lq)
            else:
                with torch.no_grad():
                    clean, encoded_control, skip_feature_list = self.diffusion.evae_prepare(lq)
            torch_task_id = torch.tensor([task_id]*encoded_control.shape[0]).to(encoded_control.device)
            gates, load = self.router_list[task_id](task_id=torch_task_id, degradation_info=skip_feature_list[-1])
            
            output_image, latents_pred = self.diffusion.generate_forward(encoded_control, skip_feature_list, prompt_embeds, gates)
            output_image = output_image*0.5 + 0.5
            
            return clean, load, output_image, latents_pred
        
        if (kl_turn):
            with torch.no_grad():
                x0_pred_fix,x0_pred_update = self.diffusion.kl_forward(noisy_latents, timesteps, prompt_embeds, noise_pred_fix)
                
            weighting_factor = torch.abs(latents_pred - x0_pred_fix).mean(dim=[1, 2, 3], keepdim=True)
            grad = (x0_pred_update - x0_pred_fix) / weighting_factor
            loss = F.mse_loss(latents_pred, (latents_pred - grad).detach())

            return loss
        
        if (diff_turn):
            loss_diff = self.diffusion.diff_forward(latents=latents_pred,prompt_embeds=prompt_embeds)
            return loss_diff
        
    def save_model(self, outf):
        sd = {}
        sd["router"] = self.router_list.state_dict()
        sd["diffusion"] = self.diffusion.save_lora()
        if self.use_ema:
            sd["ema"] = self.ema.shadow  # ← 保存 EMA 参数
        # sd["state_dict_vae_encoder"] = {k: v for k, v in self.vae_encoder.state_dict().items() if "lora" in k}
        torch.save(sd, outf)
    
    def load_model(self, load_path):
        sd = torch.load(load_path, weights_only=True, map_location="cpu")
        self.router_list.load_state_dict(sd["router"], strict=True)
        self.diffusion.load_model(sd["diffusion"])
        if self.use_ema and "ema" in sd:
            self.ema.shadow = sd["ema"]
            
    def load_ema_model(self, load_path):
        sd = torch.load(load_path, weights_only=True, map_location="cpu")
        for name, param in self.named_parameters():
            if name in sd["ema"].keys():
                param.data.copy_(sd["ema"][name])

class AUXModel(nn.Module):
    def __init__(self,ram_path, prompt_embed_path, prompt_embed_revision, 
                 task_num, img_channel, width, enc_blks, middle_blk_num, dec_blks, GCE_CONVS_nums,CGNet_load_path=None,
                 pretrained_model_path=None,
                 lora_rank = 4,vae_pretrained_model_path=None, vae_path=None,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vlm = ram(pretrained=ram_path,
            pretrained_condition=None,
            image_size=384,
            vit='swin_l')
        self.prompt_embed = Promept_embed(path=prompt_embed_path, revision=prompt_embed_revision)
        self.ram_transforms = transforms.Compose([
                                                    transforms.Resize((384, 384)),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                ])
        
        if CGNet_load_path is not None:
            self.pre_model_list = nn.ModuleList()
            for _ in range(task_num):
                self.pre_model_list.append(CascadedGaze(img_channel=img_channel,
                                                        width=width, 
                                                        enc_blk_nums=enc_blks, 
                                                        middle_blk_num=middle_blk_num,
                                                        dec_blk_nums=dec_blks,
                                                        GCE_CONVS_nums=GCE_CONVS_nums))
            sd = torch.load(CGNet_load_path, weights_only=True, map_location="cpu")
            self.pre_model_list.load_state_dict(sd, strict=True)
            print("[INFO] load pre_model weight successfully !!")
        
        if pretrained_model_path is not None:
            self.unet_fix = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
            self.unet_fix.requires_grad_(False)
            print("[INFO] load unet_fix weight successfully !!")
            
            
        if vae_path is not None:
            self.vae = eVAE(pretrained_model_path=vae_pretrained_model_path,lora_rank=lora_rank)
            # self.vae.requires_grad_(False)
            self.vae.eval()
            self.vae.load_model_path(vae_path)
            print("[INFO] load VAE successfully")
        
    def prepare_model(self, device, weight_dtype):
        self.eval()
        self.device = device
        self.weight_dtype = weight_dtype
        self.vlm.to(device=device,dtype=torch.float16)
        self.prompt_embed.to(device=device, dtype=weight_dtype)
        
        try:
            self.pre_model_list.to(device=device, dtype=weight_dtype)
        except AttributeError as e:
            pass
        
        try:
            self.unet_fix.to(device=device, dtype=weight_dtype)
        except AttributeError as e:
            pass
        
        try:
            self.vae.to(device=device, dtype=weight_dtype)
        except AttributeError as e:
            pass
    
    @torch.no_grad()
    def get_neg_prompt_embeds(self, B):
        neg_promt = ["" for _ in range(B)]
        prompt_embeds = self.prompt_embed(neg_promt)
        return prompt_embeds
    
    @torch.no_grad()
    def get_prompt_embeds(self, x_tgt:torch.Tensor):
        cur_device = x_tgt.device
        x_tgt_ram = self.ram_transforms(x_tgt)
        caption = inference(x_tgt_ram.to(self.device, dtype=torch.float16), self.vlm)
        prompt = [f'{each_caption}, A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting' 
                            for each_caption in caption]
        prompt_embeds = self.prompt_embed(prompt).to(cur_device)
        return prompt_embeds
    
    @torch.no_grad()
    def get_clean(self, task_id, lq:torch.Tensor):
        cur_device = lq.device
        clean = self.pre_model_list[task_id](lq.to(self.device)).to(device=cur_device)
        return clean
    
    @torch.no_grad()
    def get_fixed_pred(self, noisy_latents, timesteps, neg_prompt_embeds, prompt_embeds):
        cur_device = noisy_latents.device
        noisy_latents_input = torch.cat([noisy_latents] * 2).to(self.device)
        timesteps_input = torch.cat([timesteps] * 2).to(self.device)
        prompt_embeds = torch.cat([neg_prompt_embeds.to(self.device), prompt_embeds.to(self.device)], dim=0)
        noise_pred_fix,_ = self.unet_fix(
                noisy_latents_input.to(dtype=self.weight_dtype),
                timestep=timesteps_input,
                encoder_hidden_states=prompt_embeds.to(dtype=self.weight_dtype),
                )
        noise_pred_fix = noise_pred_fix.sample
        noise_pred_fix.to(cur_device)
        return noise_pred_fix
    
    @torch.no_grad()
    def get_latent(self, x_src):
        cur_device = x_src.device
        c_src = x_src*2-1
        lq_latent, skip_feature = self.vae.encoder(c_src.to(self.device))
        skip_feature = [feature.to(cur_device) for feature in skip_feature]
        return lq_latent.to(cur_device), skip_feature
        
    @torch.no_grad()
    def get_high_latent(self, x_tgt):
        cur_device = x_tgt.device
        c_tgt = x_tgt*2-1
        high_latnt = self.vae.get_high_latent(c_tgt.to(self.device))
        return high_latnt.to(cur_device)
    
    def decoder(self,pre_latent, skip_feature):
        cur_device = pre_latent.device
        new_skip_feature = []
        for feature in skip_feature:
            new_skip_feature.append(feature.to(self.device))
        output_img = self.vae.decoder(pre_latent.to(self.device),new_skip_feature).clamp(-1, 1)
        return output_img.to(cur_device)
    
    
class Inference_MoeDiffusion(nn.Module):
    def __init__(self, pretrained_model_path, lora_rank=4, num_experts=5, train_vae=True, task_num=5, router_embed_dim=32, top_k=2, noise_epsilon=0.01, degradation_channels=3,
                 vae_path=None, train_vae_part=None, use_ema=True, ema_decay=0.999, 
                 ram_path=None, prompt_embed_revision=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.router_list = nn.ModuleList()
        for _ in range(task_num):
            self.router_list.append(Router(expert_nums=num_experts,
                                           embed_dim=router_embed_dim,
                                           top_k=top_k,
                                           noise_epsilon=noise_epsilon,
                                           degradation_channels=degradation_channels))
        self.diffusion = Infer_Diffusion_EVAE(pretrained_model_path=pretrained_model_path,
                                                lora_rank=lora_rank,
                                                num_experts=num_experts,
                                                vae_path=vae_path,
                                                train_vae=train_vae)
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        
        self.vlm = ram(pretrained=ram_path,
            pretrained_condition=None,
            image_size=384,
            vit='swin_l')
        self.prompt_embed = Promept_embed(path=pretrained_model_path, revision=prompt_embed_revision)
        self.ram_transforms = transforms.Compose([
                                                    transforms.Resize((384, 384)),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                ])
        
    @torch.no_grad()
    def get_neg_prompt_embeds(self, B):
        neg_promt = ["" for _ in range(B)]
        prompt_embeds = self.prompt_embed(neg_promt)
        return prompt_embeds
    
    @torch.no_grad()
    def get_prompt_embeds(self, x_tgt:torch.Tensor):
        cur_device = x_tgt.device
        x_tgt_ram = self.ram_transforms(x_tgt)
        caption = inference(x_tgt_ram.to(self.device, dtype=torch.float16), self.vlm)
        prompt = [f'{each_caption}, A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting' 
                            for each_caption in caption]
        prompt_embeds = self.prompt_embed(prompt).to(cur_device)
        return prompt_embeds
    
    @torch.no_grad()
    def forward(self, lq=None, task_id=None):
            clean, encoded_control, skip_feature_list = self.diffusion.evae_prepare(lq)
            prompt_embeds = self.get_prompt_embeds(lq)
            
            torch_task_id = torch.tensor([task_id]*encoded_control.shape[0]).to(encoded_control.device)
            gates, load = self.router_list[task_id](task_id=torch_task_id, degradation_info=skip_feature_list[-1])
            
            output_image, latents_pred = self.diffusion.generate_forward(encoded_control, skip_feature_list, prompt_embeds, gates)
            output_image = output_image*0.5 + 0.5
            
            return clean, load, output_image, latents_pred
        
    def load_ema_model(self, load_path):
        sd = torch.load(load_path, weights_only=True, map_location="cpu")
        for name, param in self.named_parameters():
            if name in sd["ema"].keys():
                param.data.copy_(sd["ema"][name])
    
        
        