import os
import gc
import sys
sys.path.append(os.getcwd())
import lpips
import clip
import argparse
import numpy as np
import pyiqa
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
import torchvision.utils as vutils
from tqdm.auto import tqdm
import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
import logging
from pathlib import Path
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf


from model.dists_loss import DISTS
from model.MoeDiffusion import MoeDiffusion, AUXModel
from model.diffusion import eVAE
from utils.utils_logger import logger_info
from utils.common import instantiate_from_config


def cycle(dl):
    while True:
        for data in dl:
            yield data


def main(args):
# ========================  Prepare ============================
    cfg = OmegaConf.load(args.config)
    set_seed(cfg.seed)
    logging_dir = Path(args.output_dir, cfg.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )
    print(f"当前进程rank: {accelerator.process_index} || {torch.cuda.device_count()//2+accelerator.process_index}")
    # 辅助显卡
    aux_device = torch.device(f"cuda:{torch.cuda.device_count()//2+accelerator.process_index}")
    # aux_device = torch.device(f"cuda:{torch.cuda.device_count()-1}")
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True


    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
        
# ========================  Df model ============================
    if args.train_type == "Diffusion":
        AUX_model:AUXModel = instantiate_from_config(cfg.model.AUXModel)
        AUX_model.prepare_model(device=aux_device, weight_dtype=weight_dtype)
        
        frozen = []
        model:MoeDiffusion = instantiate_from_config(cfg.model.MoeDiffusion)
        model.set_train(task_id=args.task_id, frozen=frozen)
        
        # optimizer
        if cfg.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                model.diffusion.unet.enable_xformers_memory_efficient_attention()
                model.diffusion.vae.vae.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available, please install it by running `pip install xformers`")
        if cfg.gradient_checkpointing:
            model.diffusion.unet.enable_gradient_checkpointing()
            # model.diffusion.vae.enable_gradient_checkpointing()
            
    elif args.train_type == "Evae":
        model:eVAE = instantiate_from_config(cfg.model.EVAE)
        model.set_train()
    
    
    if cfg.loss_type == "LPIPS":
        feat_loss_fn = lpips.LPIPS(net='vgg').cuda()
        feat_loss_fn.requires_grad_(False)
    else:
        feat_loss_fn = DISTS(device=accelerator.device, as_loss=True)
        feat_loss_fn.requires_grad_(False)
        


# ========================  Optimizer ============================

    # make the optimizer
    if args.train_type == "Evae":
        layers_to_opt_vae = []
        for n, _p in model.named_parameters():
            if( _p.requires_grad):
                layers_to_opt_vae.append(_p)
        layers_to_opt = [
            {
                "params":layers_to_opt_vae, 
                "lr": cfg.evae_lr,                  # Unet学习率
                "weight_decay": cfg.adam_weight_decay,
            },
        ]
    elif args.train_type == "Diffusion":
        layers_to_opt_unet = []
        layers_to_opt_router = []
        layers_to_opt_vae = []
        layers_to_opt_reg = []
        layers_to_opt = []
        for n, _p in model.named_parameters():
            if( _p.requires_grad):
                if "reg_unet" in n:
                    layers_to_opt_reg.append(_p)
                elif "vae" in n:
                    layers_to_opt_vae.append(_p)
                elif "router_list" in n:
                    print(n)
                    layers_to_opt_router.append(_p)
                else:
                    layers_to_opt_unet.append(_p)
                    
        if len(layers_to_opt_unet) > 0:
            layers_to_opt.append({
                "params":layers_to_opt_unet, 
                "lr": cfg.unet_lr,                  # Unet学习率
                "weight_decay": cfg.adam_weight_decay,
            })
        if len(layers_to_opt_router) > 0:
            layers_to_opt.append({
                "params":layers_to_opt_router, 
                "lr": cfg.router_lr,                  # router学习率
                "weight_decay": cfg.adam_weight_decay,
            })
        if len(layers_to_opt_router) > 0:
            layers_to_opt.append({
                "params":layers_to_opt_vae, 
                "lr": cfg.vae_lr,                  # decoder学习率（已预训练过）
                "weight_decay": cfg.adam_weight_decay,
            })
        
        optimizer_reg = torch.optim.AdamW(layers_to_opt_reg, lr=cfg.reg_lr,
            betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.adam_weight_decay,
            eps=cfg.adam_epsilon,)
        lr_scheduler_reg = get_scheduler(cfg.lr_scheduler, optimizer=optimizer_reg,
            num_warmup_steps=cfg.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=cfg.lr_num_cycles, power=cfg.lr_power,)
                
    optimizer = torch.optim.AdamW(layers_to_opt,
        betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,)
    lr_scheduler = get_scheduler(cfg.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=cfg.lr_num_cycles, power=cfg.lr_power,)
    
# ========================  Load ============================
    # Get the most recent checkpoint
    dirs = os.listdir(args.output_dir)
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    path = dirs[-1] if len(dirs) > 0 else None
    
    if path == None:
        accelerator.print("[WARNNING] Checkpoint does not exist. Starting a new training run.")
        initial_global_step = 0
    else:
        accelerator.print(f"[INFO] Resuming from checkpoint {path}")
        initial_global_step = int(path.split("-")[1])
        load_dir = os.path.join(args.output_dir, path)
        print("load_dir",load_dir)
        model_load_path = os.path.join(load_dir, "model.pkl")  
        model.load_model(load_path=model_load_path)      
        outf_opt = os.path.join(load_dir, "optimizer.pkl")
        if os.path.exists(outf_opt):
            opt_sd = torch.load(outf_opt, map_location='cpu')
            optimizer.load_state_dict(opt_sd)
        else:
            accelerator.print("[WARNNING] optimizer_state does not exist. Starting a new training run.")

# ========================  Dataset ============================
    dataset_train = instantiate_from_config(cfg.datasets.train)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=cfg.train_batch_size, shuffle=True,pin_memory=True, num_workers=cfg.dataloader_num_workers)
    dataset_test = instantiate_from_config(cfg.datasets.test)
    dl_test = torch.utils.data.DataLoader(dataset_test, batch_size=cfg.test_batch_size, shuffle=True,pin_memory=True, num_workers=cfg.dataloader_num_workers)
    
# ========================  Prepare ============================
    # 清理未使用的缓存
    torch.cuda.empty_cache()
    if args.train_type == "Evae":
        model,  optimizer, lr_scheduler, dl_train, dl_test = accelerator.prepare(
            model,  optimizer, lr_scheduler, dl_train,dl_test
        )
    else:
        model,  optimizer, lr_scheduler, optimizer_reg, lr_scheduler_reg, dl_train = accelerator.prepare(
            model,  optimizer, lr_scheduler, optimizer_reg, lr_scheduler_reg, dl_train
        )
        try:
            accelerator.unwrap_model(model).ema.to(accelerator.device)
        except AttributeError as e:
            pass
        
    dl_train = cycle(dl_train)
    feat_loss_fn = accelerator.prepare(feat_loss_fn)

    if accelerator.is_main_process:
        accelerator.init_trackers(cfg.tracker_project_name)
   
    global_step = initial_global_step
    progress_bar = tqdm(range(0, args.max_train_steps), initial=initial_global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,)
    
    if accelerator.is_main_process:     
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "image"))
        logger_name = "val"
        logger_info(logger_name, os.path.join(os.path.join(args.output_dir, "eval"), logger_name+'.log'))# type: ignore
        logger = logging.getLogger(logger_name)
        if args.train_type == "Evae":
            musiq_metric = pyiqa.create_metric('musiq', device=accelerator.device)
            clipiqa_metric = pyiqa.create_metric('clipiqa', device=accelerator.device)
            psnr_metric = pyiqa.create_metric('psnr', device=accelerator.device)
            dists_metric = pyiqa.create_metric('dists', device=accelerator.device)
            ssim_metric = pyiqa.create_metric('ssim', device=accelerator.device)
        load_sum = np.array([0,0,0,0,0])
    
    with torch.no_grad():
        neg_prompt_embeds = AUX_model.get_neg_prompt_embeds(cfg.train_batch_size)

# ========================  Start Training ============================
    for _ in range(args.max_train_steps):
        batch = next(dl_train)
        with accelerator.accumulate(model):
            generator_loss = 0.0
            x_tgt = batch['gt'].to(accelerator.device, dtype=weight_dtype)
            x_src = batch['lq'].to(accelerator.device, dtype=weight_dtype)
            B, C, H, W = x_src.shape

            if args.train_type == "Evae":
                output_image = model(x_src,x_tgt)
                loss_l2 = F.mse_loss(output_image.float(), x_tgt.float(), reduction="mean") * cfg.loss_l2
                loss_feat = feat_loss_fn(output_image.float(), x_tgt.float()).mean() * cfg.loss_feat

                generator_loss = loss_l2 + loss_feat
                
                accelerator.backward(generator_loss)
                if accelerator.sync_gradients:
                    generator_grad_norm = accelerator.clip_grad_norm_(layers_to_opt, cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad() 
            else:
                with torch.no_grad():
                    prompt_embeds = AUX_model.get_prompt_embeds(x_tgt)
                
                clean, load, output_image, latents_pred = model(generate_turn=True, lq=x_src, prompt_embeds=prompt_embeds, task_id=args.task_id)
                
                timesteps = torch.randint(20, 980, (B,), device=latents_pred.device).long()
                noise = torch.randn_like(latents_pred)
                noisy_latents = accelerator.unwrap_model(model).add_noise(latents_pred, noise, timesteps)
                
                with torch.no_grad():
                    noise_pred_fix = AUX_model.get_fixed_pred(noisy_latents,timesteps, neg_prompt_embeds, prompt_embeds)
                    noise_pred_uncond, noise_pred_text = noise_pred_fix.chunk(2)
                    noise_pred_fix = noise_pred_uncond + cfg.cfg_vsd * (noise_pred_text - noise_pred_uncond)
                    noise_pred_fix.to(dtype=torch.float32)
                    
                loss_kl = model(kl_turn=True, latents_pred=latents_pred, noisy_latents=noisy_latents, timesteps=timesteps,noise_pred_fix=noise_pred_fix, prompt_embeds=prompt_embeds)*cfg.loss_kl
                # load: tensor([0.0000, 0.4055, 0.0000, 0.0000, 0.5945], device='cuda:0',grad_fn=<_DDPSinkBackward>)
                loss_l2 = F.mse_loss(output_image.float(), x_tgt.float(), reduction="mean") * cfg.loss_l2
                loss_feat = feat_loss_fn(output_image.float(), x_tgt.float()).mean() * cfg.loss_feat

                generator_loss = loss_l2 + loss_feat + loss_kl
                
                accelerator.backward(generator_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt_unet, cfg.max_grad_norm)
                    accelerator.clip_grad_norm_(layers_to_opt_router, cfg.max_grad_norm)
                    accelerator.clip_grad_norm_(layers_to_opt_vae, cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad() 
                
                """
                diff loss: let lora model closed to generator 
                """
                loss_d = model(diff_turn=True, latents_pred=latents_pred, prompt_embeds=prompt_embeds)
                accelerator.backward(loss_d)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt_reg, cfg.max_grad_norm)
                optimizer_reg.step()
                lr_scheduler_reg.step()
                optimizer_reg.zero_grad()

                try:
                    accelerator.unwrap_model(model).ema.update()
                except AttributeError as e:
                    pass
                
        ############## log #########
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            
            if accelerator.is_main_process:
               
                logs = {}
                # log all the losses
                if args.train_type == "Diffusion":
                    load_detach = load.detach()
                    if load_detach.ndimension() == 1:
                        load_detach.unsqueeze_(0)   
                    load_detach = load_detach.cpu().numpy()
                    load_sum = load_sum + load_detach.sum(axis=0)
                    
                    logs["loss_d"] = loss_d.detach().item()
                    logs["loss_kl"] = loss_kl.detach().item()
                    
                logs["loss_feat"] = loss_feat.detach().item()
                logs["loss_l2"] = loss_l2.detach().item()
                logs["generator_loss"] = generator_loss.detach().item()
                
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step % cfg.log.image_every == 0:
                    combined = torch.cat((x_tgt[0], output_image[0], x_src[0]), dim=2)  # (B, C, 2*W, H)

                    # 将拼接后的 combined 图像转换为网格，并记录到 TensorBoard
                    combined_grid = vutils.make_grid(combined, nrow=3)  # 将每4张图片排列为一行
                    writer.add_image("gt || out || lq", combined_grid, global_step=global_step//cfg.log.image_every)  
                
                # checkpoint the model
                if global_step % cfg.log.checkpointing_steps == 0:
                    load_dir = os.path.join(args.output_dir, f"checkpoints-{global_step}")
                    os.makedirs(load_dir, exist_ok=True)
                    outf = os.path.join(load_dir, "model.pkl")
                    outf_opt = os.path.join(load_dir, "optimizer.pkl")
                    accelerator.unwrap_model(model).save_model(outf)
                    torch.save(accelerator.unwrap_model(optimizer).state_dict(),outf_opt)
                    if args.train_type == "Diffusion":
                        # accelerator.unwrap_model(model_ir).save_model(outf_ir)
                        with open(os.path.join(args.output_dir,'load.txt'), 'a') as f:  # 使用 'a' 模式打开文件进行追加写入
                            np.savetxt(f, [load_sum], fmt='%.4f', delimiter=', ')
                    print(f"Save to {load_dir} successfully !!")
                    
                ### Eval
                if (args.train_type == "Evae") and (global_step % cfg.log.val_every == 0):
                    with torch.no_grad():
                        model.eval()
                        avg_dict = {'psnr':{ 'd_psnr':0,},
                                    'ssim':{'d_ssim':0, }, 
                                    'dists':{'d_dists':0, }, 
                                    # 'lpips':{'d_lpips':0},
                                    'dists':{'d_dists':0},
                                    'musiq':{'d_musiq':0, 'gt_musiq':0},
                                    'clipiqa':{ 'd_clipiqa':0, 'gt_clipiqa':0}}
                        logger.info(f'========= START TESTING : length of test_dataset is : {len(dl_test)}')
                        for count, batch in enumerate(dl_test):
                            x_tgt = batch['gt'].to(accelerator.device, dtype=weight_dtype)
                            x_src = batch['lq'].to(accelerator.device, dtype=weight_dtype)
                            output_image = model(x_src,x_tgt)
                            
                            image_name = os.path.splitext(os.path.basename(batch['gt_path'][0]))[0]
                            
                            d_psnr = psnr_metric(output_image, x_tgt).item()
                            logger.info('[PSNR] || {:->4d}--> img:{:>10s}  | D:{:<4.2f}dB '.format(count+1, image_name, d_psnr))
                            avg_dict['psnr']['d_psnr'] += d_psnr
                            
                            d_ssim = ssim_metric(output_image, x_tgt).item()
                            logger.info('[SSIM] || {:->4d}--> img:{:>10s}  | D:{:<4.2f}dB '.format(count+1, image_name,  d_ssim))
                            avg_dict['ssim']['d_ssim'] += d_ssim
                            
                            d_dists = dists_metric(output_image, x_tgt).item()
                            logger.info('[DISTS] || {:->4d}--> img:{:>10s}  | D:{:<4.2f}dB '.format(count+1, image_name, d_dists))
                            avg_dict['dists']['d_dists'] += d_dists
                            
                            # d_lpips = net_lpips(output_image.float(), x_tgt.float()).mean().item()
                            # logger.info('[LPIPS] || {:->4d}--> img:{:>10s}  | D:{:<4.2f}dB '.format(count+1, image_name,  d_lpips))
                            # avg_dict['lpips']['d_lpips'] += d_lpips
                            
                            d_dists = feat_loss_fn(output_image.float(), x_tgt.float()).mean().item()
                            logger.info('[dists] || {:->4d}--> img:{:>10s}  | D:{:<4.2f}dB '.format(count+1, image_name,  d_dists))
                            avg_dict['dists']['d_dists'] += d_dists
                            
                            d_musiq = musiq_metric(output_image).item()
                            gt_musiq = musiq_metric(x_tgt).item()
                            logger.info('[MUSIQ] || {:->4d}--> img:{:>10s}  | D:{:<4.2f} | GT:{:<4.2f}'.format(count+1, image_name, d_musiq, gt_musiq))
                            avg_dict['musiq']['d_musiq'] += d_musiq
                            avg_dict['musiq']['gt_musiq'] += gt_musiq
                            
                            d_clipiqa = clipiqa_metric(output_image).item()
                            gt_clipiqa = clipiqa_metric(x_tgt).item()
                            logger.info('[CLIPIQA] || {:->4d}--> img:{:>10s}  | D:{:<4.2f} | GT:{:<4.2f}'.format(count+1, image_name, d_clipiqa,gt_clipiqa))
                            avg_dict['clipiqa']['d_clipiqa'] += d_clipiqa
                            avg_dict['clipiqa']['gt_clipiqa'] += gt_clipiqa

                            combined = torch.cat(( x_tgt[0], output_image[0], x_src[0]), dim=2)  # (B, C, 2*W, H)
                            # 将拼接后的 combined 图像转换为网格，并记录到 TensorBoard
                            combined_grid = vutils.make_grid(combined, nrow=3)  # 将每5张图片排列为一行
                            writer.add_image(f"test_img", combined_grid, global_step=count) 
                            print('+'*50)
                        for k, v in avg_dict.items():
                            for n,p in v.items():
                                avg_dict[k][n] = p / (count + 1)
                                
                        writer.add_scalar("PSNR",  avg_dict['psnr']['d_psnr'], global_step // cfg.log.val_every)
                        writer.add_scalar("SSIM",  avg_dict['ssim']['d_ssim'], global_step // cfg.log.val_every)
                        writer.add_scalar("DISTS",  avg_dict['dists']['d_dists'], global_step // cfg.log.val_every)
                        # writer.add_scalar("LPIPS",  avg_dict['lpips']['d_lpips'], global_step // cfg.log.val_every)
                        writer.add_scalar("MUSIQ",  avg_dict['musiq']['d_musiq'], global_step // cfg.log.val_every)
                        writer.add_scalar("CLIPIQA",  avg_dict['clipiqa']['d_clipiqa'], global_step // cfg.log.val_every)
                        logger.info('================== END TESTING =======================')
                        logger.info('<iter:Testing, Average PSNR : D:{:<4.2f}dB '.format(avg_dict['psnr']['d_psnr']))
                        logger.info('<iter:Testing, Average SSIM : D:{:<4.2f} '.format(avg_dict['ssim']['d_ssim']))
                        logger.info('<iter:Testing, Average DISTS : D:{:<4.2f} '.format(avg_dict['dists']['d_dists']))
                        # logger.info('<iter:Testing, Average LPIPS : D:{:<4.2f} '.format(avg_dict['lpips']['d_lpips']))
                        logger.info('<iter:Testing, Average MUSIQ : D:{:<4.2f} | GT:{:<4.2f} '.format(avg_dict['musiq']['d_musiq'], avg_dict['musiq']['gt_musiq']))
                        logger.info('<iter:Testing, Average CLIPIQA : D:{:<4.2f} | GT:{:<4.2f}'.format(avg_dict['clipiqa']['d_clipiqa'], avg_dict['clipiqa']['gt_clipiqa']))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train_type", type=str, choices=["Evae", "Diffusion"])
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_train_steps", type=int, required=True)
    args = parser.parse_args()
    main(args)