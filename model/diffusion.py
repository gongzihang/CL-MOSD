import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from diffusers import DDPMScheduler


from model.add_Moe import initialize_skip_vae, initialize_unet_only, initialize_unet
from utils.moe_utils import SparseDispatcher, Moe_layer, replace_layers_byname

class eVAE(nn.Module):
    def __init__(self, pretrained_model_path, lora_rank=4,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vae, self.lora_vae_modules_encoder = initialize_skip_vae(pretrained_model=pretrained_model_path, lora_rank=lora_rank)
        
    def set_train(self):
        for n, _p in self.vae.named_parameters():
            if "lora" in n or "skip" in n:
                _p.requires_grad = True
    
    def get_pre_info(self, c_t):
        """
            c_t = [-1,1]
        """
        latent, skip_feature_list = self.vae.encode(c_t)
        latent = latent.latent_dist.sample() * self.vae.config.scaling_factor
        return latent, skip_feature_list
    
    def encoder(self, c_t):
        """
            c_t = [-1,1]
        """
        sample, skip_feature_list = self.vae.encode(c_t)
        encoded_control = sample.latent_dist.sample() * self.vae.config.scaling_factor
        return encoded_control, skip_feature_list
    
    def decoder(self, high_latent, pre_info):
        """
            return C_t (-1,1)
        """
        output_image = (self.vae.decode(high_latent /self.vae.config.scaling_factor, pre_info).sample).clamp(-1, 1)
        return output_image
    
    def prepare(self, x_src):
        c_t_src = x_src*2 -1 
        # c_t_tgt = x_tgt*2 -1 
        latent, pre_info = self.get_pre_info(c_t_src)
        skip_feature_list = copy.deepcopy(pre_info)
        output_img = self.decoder(high_latent=latent, pre_info=pre_info)
        output_img = (output_img+1)/2
        return output_img, latent, skip_feature_list
    
    def forward(self, x_src, x_tgt):
        """
            for train convenient, img = [0,1] , return [0,1]
        """
        c_t_src = x_src*2 -1 
        # c_t_tgt = x_tgt*2 -1 
        latent, pre_info = self.get_pre_info(c_t_src)
        output_img = self.decoder(high_latent=latent, pre_info=pre_info)
        output_img = (output_img+1)/2
        return output_img
    
    def save_lora(self, path):
        sd = {}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k }
        torch.save(sd, path)
    
    def load_model(self, path):
        sd = torch.load(path, map_location="cpu")
        for n, p in self.vae.named_parameters():
            if "lora" in n or "skip" in n:
                if n in sd['state_dict_vae'].keys():
                    p.data.copy_(sd["state_dict_vae"][n])
                    
                    
    def load_model_path(self, path):
        sd = torch.load(path, map_location="cpu")
        for n, p in self.vae.named_parameters():
            if "lora" in n or "skip" in n:
                if n in sd['state_dict_vae'].keys():
                    p.data.copy_(sd["state_dict_vae"][n])

class Diffusion_EVAE(nn.Module):
    def __init__(self, pretrained_model_path, lora_rank=4, num_experts=5, vae_path=None, train_vae=True, *args, **kwargs) -> None:
        super().__init__(pretrained_model_path, lora_rank, num_experts, train_vae, *args, **kwargs)
        self.dispatcher = SparseDispatcher()
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device="cuda")
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.cuda()
        
        self.vae = eVAE(pretrained_model_path=pretrained_model_path, lora_rank=lora_rank)
        self.unet, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = initialize_unet(pretrained_model_path, lora_rank, num_experts, self.dispatcher)
        self.reg_unet,_,_,_ = initialize_unet_only(pretrained_model_path, lora_rank)
        
        self.unet.to("cuda")
        self.reg_unet.to("cuda")
        self.vae.to("cuda")
        self.vae.load_model(path=vae_path)
        self.timesteps = torch.tensor([999], device="cuda").long()
    
        self.num_experts = num_experts
        
    def set_train(self, frozen=[]):
        self.unet.train()
        self.vae.eval()
        self.vae.requires_grad_(False)
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        for frozen_id in frozen:
            for n, _p in self.named_parameters():
                if f"expert_linear_lists.{frozen_id}." in n:
                    _p.requires_grad = False
                    
        self.reg_unet.train()
        for n, _p in self.reg_unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        
    def evae_prepare(self, x_src):
        with torch.no_grad():
            clean, _, _ = self.vae.prepare(x_src=x_src)
        encoded_control, skip_feature_list = self.vae.encoder(c_t=x_src*2-1)
        return clean, encoded_control, skip_feature_list
    
    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        return pred_original_sample
    
    def generate_forward(self, encoded_control, skip_feature_list, prompt_embeds, gates):
        """
        c_t: [-1,1] 输入RGB图片
        return :
            origin_output, middle_out, VAE_latent
        """
        self.dispatcher.updata(num_experts=self.num_experts, gates=gates)
        
        model_pred, midlle_output = self.unet(encoded_control, self.timesteps, encoder_hidden_states=prompt_embeds.to(torch.float32),)
        model_pred = model_pred.sample
        # midlle_output = midlle_output[-1].float()
        
        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
        output_image = self.vae.decoder(high_latent=x_denoised, pre_info=skip_feature_list)
        # return output_image, x_denoised, encoded_control, model_pred
        return output_image, x_denoised
    
    @torch.no_grad()
    def kl_forward(self,noisy_latents, timesteps, prompt_embeds, noise_pred_fix):
        noise_pred_update,_ = self.reg_unet(
                noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds.float(),
                )
        noise_pred_update = noise_pred_update.sample
        x0_pred_update = self.eps_to_mu(self.noise_scheduler, noise_pred_update, noisy_latents, timesteps)
        x0_pred_fix = self.eps_to_mu(self.noise_scheduler, noise_pred_fix, noisy_latents, timesteps)
        return x0_pred_fix,x0_pred_update
    
    def diff_forward(self, latents, prompt_embeds):

        latents, prompt_embeds = latents.detach(), prompt_embeds.detach()
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = self.reg_unet(
        noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        )[0].sample

        loss_d = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        
        return loss_d
    
    def save_lora(self):
        sd = {}
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_reg_unet"] = {k: v for k, v in self.reg_unet.state_dict().items() if "lora" in k or "conv_in" in k}
        return sd
    
    def load_model(self, sd):
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(sd["state_dict_unet"][n])
                
        for n, p in self.reg_unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(sd["state_dict_reg_unet"][n])
    
class Diffusion_EVAE_Decoder(Diffusion_EVAE):
    def __init__(self, pretrained_model_path, lora_rank=4, num_experts=5, vae_path=None, train_vae=True, *args, **kwargs) -> None:
        super().__init__(pretrained_model_path, lora_rank, num_experts, vae_path, train_vae, *args, **kwargs)
        
    def set_train(self, frozen=[]):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        for frozen_id in frozen:
            for n, _p in self.named_parameters():
                if f"expert_linear_lists.{frozen_id}." in n:
                    _p.requires_grad = False
                    
        self.reg_unet.train()
        for n, _p in self.reg_unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        
        self.vae_train_list = []
        self.vae.train()
        self.vae.requires_grad_(False)
        for n, _p in self.vae.named_parameters():
            if "lora" in n and "decoder" in n:
                _p.requires_grad = True
                self.vae_train_list.append(n)
        
    
    def save_lora(self):
        sd = {}
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if k in self.vae_train_list}
        sd["state_dict_reg_unet"] = {k: v for k, v in self.reg_unet.state_dict().items() if "lora" in k or "conv_in" in k}
        return sd
    
    def load_model(self, sd):
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(sd["state_dict_unet"][n])
                
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                if n in sd['state_dict_vae'].keys():
                    p.data.copy_(sd["state_dict_vae"][n])
                    
        for n, p in self.reg_unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(sd["state_dict_reg_unet"][n])
    
class Diffusion_EVAE_Encoder(Diffusion_EVAE):
    def __init__(self, pretrained_model_path, lora_rank=4, num_experts=5, vae_path=None, train_vae=True, *args, **kwargs) -> None:
        super().__init__(pretrained_model_path, lora_rank, num_experts, vae_path, train_vae, *args, **kwargs)
        
    def set_train(self, frozen=[]):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        for frozen_id in frozen:
            for n, _p in self.named_parameters():
                if f"expert_linear_lists.{frozen_id}." in n:
                    _p.requires_grad = False
                    
        self.reg_unet.train()
        for n, _p in self.reg_unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        
        self.vae_train_list = []
        self.vae.train()
        self.vae.requires_grad_(False)
        for n, _p in self.vae.named_parameters():
            if "lora" in n and "encoder" in n:
                _p.requires_grad = True
                self.vae_train_list.append(n)
    
    def save_lora(self):
        sd = {}
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if k in self.vae_train_list}
        sd["state_dict_reg_unet"] = {k: v for k, v in self.reg_unet.state_dict().items() if "lora" in k or "conv_in" in k}
        return sd
    
    def load_model(self, sd):
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(sd["state_dict_unet"][n])
                
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                if n in sd['state_dict_vae'].keys():
                    p.data.copy_(sd["state_dict_vae"][n])
                    
        for n, p in self.reg_unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(sd["state_dict_reg_unet"][n])
                    
class Diffusion_EVAE_All(Diffusion_EVAE):
    def __init__(self, pretrained_model_path, lora_rank=4, num_experts=5, vae_path=None, train_vae=True, *args, **kwargs) -> None:
        super().__init__(pretrained_model_path, lora_rank, num_experts, vae_path, train_vae, *args, **kwargs)

    def set_train(self, frozen=[]):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        for frozen_id in frozen:
            for n, _p in self.named_parameters():
                if f"expert_linear_lists.{frozen_id}." in n:
                    _p.requires_grad = False
                    
        self.reg_unet.train()
        for n, _p in self.reg_unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        
        self.vae_train_list = []
        self.vae.train()
        self.vae.requires_grad_(False)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
                self.vae_train_list.append(n)
    
    def save_lora(self):
        sd = {}
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if k in self.vae_train_list}
        sd["state_dict_reg_unet"] = {k: v for k, v in self.reg_unet.state_dict().items() if "lora" in k or "conv_in" in k}
        return sd
    
    def load_model(self, sd):
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(sd["state_dict_unet"][n])
                
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                if n in sd['state_dict_vae'].keys():
                    p.data.copy_(sd["state_dict_vae"][n])
                    
        for n, p in self.reg_unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(sd["state_dict_reg_unet"][n])
