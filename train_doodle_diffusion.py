import os
import glob
import json
import pathlib
import logging
import argparse

import torch

import torchvision
import torch.nn.functional as F

# U Net Model.
from models.U_Net import U_Net

# Degradation Operators.
from degraders import *

from utils.utils import *

from diffusion_enums import *

from custom_dataset.doodle_dataset import DoodleImgDataset

from diffusion_sampling_algorithms import (
    ddpm_sampling,
    ddim_sampling)

def main():
    project_name = "Doodle-Diffusion"

    parser = argparse.ArgumentParser(
        description="Train Doodle Diffusion models.")
    
    parser.add_argument(
        "-c",
        "--config-path",
        help="File path to load json config file.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--device",
        help="Which hardware device will model run on (default='cpu')?",
        choices=['cpu', 'cuda'],
        type=str,
        default="cpu")
    
    args = vars(parser.parse_args())

    # Device to run model on.
    device = args["device"]

    # Load and Parse config JSON.
    config_json = args["config_path"]
    with open(config_json, 'r') as json_file:
        json_data = json_file.read()
    config_dict = json.loads(json_data)

    # File path to json dataset.
    dataset_path = config_dict["dataset_path"]
    if dataset_path is None:
        raise ValueError("No dataset_path entered.")

    # Output Directory for model's checkpoint, logs and sample output.
    out_dir = config_dict["out_dir"]
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        raise e

    # Training Params.
    starting_epoch = 0
    global_steps = 0
    checkpoint_steps = config_dict["checkpoint_steps"]  # Global steps in between checkpoints
    lr_steps = config_dict["lr_steps"]  # Global steps in between halving learning rate.
    max_epoch = config_dict["max_epoch"]
    plot_img_count = config_dict["plot_img_count"]

    min_noise_step = config_dict["min_noise_step"]  # t_1
    max_noise_step = config_dict["max_noise_step"]  # T
    max_actual_noise_step = config_dict["max_actual_noise_step"]  # Max timestep used in training step (For ensemble models training).
    skip_step = config_dict["skip_step"]  # Step to be skipped when sampling.
    if max_actual_noise_step < min_noise_step \
        or max_noise_step < min_noise_step \
            or skip_step > max_actual_noise_step \
                or skip_step < 0 \
                    or min_noise_step < 0:
        raise ValueError("Invalid step values entered!")

    # File path to checkpoints.
    diffusion_checkpoint = config_dict["model_checkpoint"]
    config_checkpoint = config_dict["config_checkpoint"]

    # Model Params.
    diffusion_lr = config_dict["diffusion_lr"]
    batch_size = config_dict["batch_size"]

    # Diffusion Params.
    # Noise Schedulers (LINEAR, COSINE).
    if config_dict["noise_scheduler"] == "LINEAR":
        noise_scheduling = NoiseScheduler.LINEAR

        # Noise Scheduler Params.
        beta_1 = config_dict["beta1"]
        beta_T = config_dict["betaT"]
    elif config_dict["noise_scheduler"] == "COSINE":
        noise_scheduling = NoiseScheduler.COSINE
    else:
        raise ValueError("Invalid noise scheduler type.")

    if config_dict["diffusion_alg"] == "DDIM":
        diffusion_alg = DiffusionAlg.DDIM
    elif config_dict["diffusion_alg"] == "DDPM":
        diffusion_alg = DiffusionAlg.DDPM
    else:
        raise ValueError("Invalid diffusion algorithm type.")

    log_path = os.path.join(out_dir, f"{project_name}.log")
    logging.basicConfig(
        # filename=log_path,
        format="%(asctime)s %(message)s",
        encoding='utf-8',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ],
        level=logging.DEBUG)

    # Initialize gradient scaler.
    scaler = torch.cuda.amp.GradScaler()
    
    # Dataset and DataLoader.
    dataset = DoodleImgDataset(dataset_path=dataset_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True)
    plot_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=plot_img_count,
        num_workers=1,
        shuffle=True)

    _, plot_labels = next(iter(plot_dataloader))

    plot_sampled_images(
        sampled_imgs=plot_labels,
        file_name=f"label_plot",
        dest_path=out_dir)

    # Model Params.
    in_channel = config_dict["in_channel"]
    out_channel = config_dict["out_channel"]
    num_layers = config_dict["num_layers"]
    num_resnet_block = config_dict["num_resnet_block"]
    attn_layers = config_dict["attn_layers"]
    attn_heads = config_dict["attn_heads"]
    attn_dim_per_head = config_dict["attn_dim_per_head"]
    time_dim = config_dict["time_dim"]
    cond_dim = config_dict["cond_dim"]
    min_channel = config_dict["min_channel"]
    max_channel = config_dict["max_channel"]
    img_recon = config_dict["img_recon"]

    # Model.
    diffusion_net = U_Net(
        in_channel=in_channel,
        out_channel=out_channel,
        num_layers=num_layers,
        num_resnet_blocks=num_resnet_block,
        attn_layers=attn_layers,
        num_heads=attn_heads,
        dim_per_head=attn_dim_per_head,
        time_dim=time_dim,
        cond_dim=cond_dim,
        min_channel=min_channel,
        max_channel=max_channel,
        image_recon=img_recon)

    # Load Pre-trained optimization configs, ignored if no checkpoint is passed.
    load_diffusion_optim = config_dict["load_diffusion_optim"]

    # Load Diffusion Model Checkpoints.
    if diffusion_checkpoint is not None:
        diffusion_status, diffusion_dict= load_checkpoint(diffusion_checkpoint)
        if not diffusion_status:
            raise Exception("An error occured while loading model checkpoint!")

        diffusion_net.custom_load_state_dict(diffusion_dict["model"])
        diffusion_net = diffusion_net.to(device)
        
        diffusion_optim = torch.optim.Adam(
            diffusion_net.parameters(),
            lr=diffusion_lr,
            betas=(0.5, 0.999))

        if load_diffusion_optim:
            diffusion_optim.load_state_dict(diffusion_dict["optimizer"])
    else:
        diffusion_net = diffusion_net.to(device)
        
        diffusion_optim = torch.optim.Adam(
            diffusion_net.parameters(),
            lr=diffusion_lr,
            betas=(0.5, 0.999))

    # Load Config Checkpoints.
    if config_checkpoint is not None:
        config_ckpt_status, config_ckpt_dict = load_checkpoint(config_checkpoint)
        if not config_ckpt_status:
            raise Exception("An error occured while loading config checkpoint!")

        if noise_scheduling == NoiseScheduler.LINEAR:
            beta_1 = config_ckpt_dict["beta_1"]
            beta_T = config_ckpt_dict["beta_T"]
        starting_epoch = config_ckpt_dict["starting_epoch"]
        global_steps = config_ckpt_dict["global_steps"]

    # x_t(X_0, eps) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.
    if noise_scheduling == NoiseScheduler.LINEAR:
        noise_degradation = NoiseDegradation(
            beta_1,
            beta_T,
            max_noise_step,
            device)
    elif noise_scheduling == NoiseScheduler.COSINE:
        noise_degradation = CosineNoiseDegradation(max_noise_step)

    logging.info("#" * 100)
    logging.info(f"Train Parameters:")
    logging.info(f"Max Epoch: {max_epoch:,}")
    logging.info(f"Dataset Path: {dataset_path}")
    logging.info(f"Output Path: {out_dir}")
    logging.info(f"Checkpoint Steps: {checkpoint_steps}")
    logging.info(f"Batch size: {batch_size:,}")
    logging.info(f"Diffusion LR: {diffusion_optim.param_groups[0]['lr']:.5f}")
    logging.info("#" * 100)
    logging.info(f"Model Parameters:")
    logging.info(f"In Channel: {in_channel:,}")
    logging.info(f"Out Channel: {out_channel:,}")
    logging.info(f"Num Layers: {num_layers:,}")
    logging.info(f"Num Resnet Block: {num_resnet_block:,}")
    logging.info(f"Attn Layers: {attn_layers}")
    logging.info(f"Attn Heads: {attn_heads:,}")
    logging.info(f"Attn dim per head: {attn_dim_per_head}")
    logging.info(f"Time Dim: {time_dim:,}")
    logging.info(f"Cond Dim: {cond_dim}")
    logging.info(f"Min Channel: {min_channel:,}")
    logging.info(f"Max Channel: {max_channel:,}")
    logging.info(f"Img Recon: {img_recon}")
    logging.info("#" * 100)
    logging.info(f"Diffusion Parameters:")
    if noise_scheduling == NoiseScheduler.LINEAR:
        logging.info(f"Beta_1: {beta_1:,.5f}")
        logging.info(f"Beta_T: {beta_T:,.5f}")
    logging.info(f"Min Noise Step: {min_noise_step:,}")
    logging.info(f"Max Noise Step: {max_noise_step:,}")
    logging.info(f"Max Actual Noise Step: {max_actual_noise_step:,}")
    logging.info("#" * 100)

    for epoch in range(starting_epoch, max_epoch):
        # Diffusion Loss.
        total_diffusion_loss = 0

        # Number of iterations.
        training_count = 0

        for index, (tr_data, lbl_data) in enumerate(dataloader):
            training_count += 1

            tr_data = tr_data.to(device)
            lbl_data = lbl_data.to(device)

            N, C, H, W = tr_data.shape

            # eps Noise.
            noise = torch.randn_like(tr_data)

            #################################################
            #             Diffusion Training.               #
            #################################################
            diffusion_optim.zero_grad()

            # Random Noise Step(t).
            rand_noise_step = torch.randint(
                low=min_noise_step,
                high=max_actual_noise_step,
                size=(N, ),
                device=device)

            # Train model.
            diffusion_net.train()

            # TODO: Allow for toggling in cases of Hardware that don't support this.
            # Enable autocasting for mixed precision.
            with torch.cuda.amp.autocast():
                # Noise degraded image (x_t).
                # x_t(X_0, eps) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.
                x_t = noise_degradation(
                    img=tr_data,
                    steps=rand_noise_step,
                    eps=noise)
                x_t_combined = torch.cat((x_t, lbl_data), dim=1)
        
                # Predicts noise from x_t.
                # eps_param(sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps, t). 
                noise_approx = diffusion_net(
                    x_t_combined,
                    rand_noise_step,
                    None)
                
                # Simplified Training Objective.
                # L_simple(param) = E[||eps - eps_param(sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps, t).||^2]
                diffusion_loss = F.mse_loss(
                    noise_approx,
                    noise)
                
                if torch.isnan(diffusion_loss):
                    raise Exception("NaN encountered during training")

            # Scale the loss and do backprop.
            scaler.scale(diffusion_loss).backward()
            
            # Update the scaled parameters.
            scaler.step(diffusion_optim)

            # Update the scaler for next iteration
            scaler.update()
            
            total_diffusion_loss += diffusion_loss.item()

            if global_steps % lr_steps == 0 and global_steps > 0:
                # Update Diffusion LR.
                for diffusion in diffusion_optim.param_groups:
                    diffusion['lr'] = diffusion['lr'] * 0.5

            # Checkpoint and Plot Images.
            if global_steps % checkpoint_steps == 0 and global_steps >= 0:
                config_state = {
                    "starting_epoch": starting_epoch,
                    "global_steps": global_steps}
                
                if noise_scheduling == NoiseScheduler.LINEAR:
                    config_state["beta_1"] = beta_1
                    config_state["beta_T"] = beta_T
                
                # Save Config Net.
                save_model(
                    model_net=config_state,
                    file_name="config",
                    dest_path=out_dir,
                    checkpoint=True,
                    steps=global_steps)

                # Save Diffusion Net.
                diffusion_state = {
                    "model": diffusion_net.state_dict(),
                    "optimizer": diffusion_optim.state_dict(),}
                save_model(
                    model_net=diffusion_state,
                    file_name="diffusion",
                    dest_path=out_dir,
                    checkpoint=True,
                    steps=global_steps)
                
                # Sample Images and plot.
                # diffusion_net.eval()  # Evaluate model.

                # X_T ~ N(0, I).
                if max_actual_noise_step < max_noise_step:
                    noise_plot = torch.randn((plot_img_count, C, H, W), device=device)
                    plot_imgs = plot_imgs.to(device)
                    x_t_plot = noise_degradation(
                        img=plot_imgs,
                        steps=torch.tensor([max_actual_noise_step], device=device),
                        eps=noise_plot)
                else:
                    x_t_plot = torch.randn((plot_img_count, C, H, W), device=device)

                plot_labels = plot_labels.to(device)

                if diffusion_alg == DiffusionAlg.DDPM:
                    x_t_plot = ddpm_sampling(
                        diffusion_net=diffusion_net,
                        noise_degradation=noise_degradation,
                        x_t=x_t_plot,
                        min_noise=min_noise_step,
                        max_noise=max_actual_noise_step,
                        cond_img=plot_labels,
                        labels_tensor=None,
                        device=device,
                        log=print)

                    plot_sampled_images(
                        sampled_imgs=x_t_plot,  # x_0
                        file_name=f"diffusion_plot_{global_steps}",
                        dest_path=out_dir)

                elif diffusion_alg == DiffusionAlg.DDIM:
                    x0_approx = ddim_sampling(
                        diffusion_net=diffusion_net,
                        noise_degradation=noise_degradation,
                        x_t=x_t_plot,
                        min_noise=min_noise_step,
                        max_noise=max_actual_noise_step,
                        cond_img=plot_labels,
                        labels_tensor=None,
                        ddim_step_size=skip_step,
                        device=device,
                        log=print)

                    plot_sampled_images(
                        sampled_imgs=x0_approx,  # t = 1
                        file_name=f"diffusion_plot_{global_steps}",
                        dest_path=out_dir)

            temp_avg_diffusion = total_diffusion_loss / training_count
            message = "Cum. Steps: {:,} | Steps: {:,} / {:,} | Diffusion: {:.5f} | LR: {:.9f}".format(
                global_steps + 1,
                index + 1,
                len(dataloader),
                temp_avg_diffusion, 
                diffusion_optim.param_groups[0]['lr']
            )
            logging.info(message)

            global_steps += 1

        # Save Config Data.
        config_state = {
            "starting_epoch": starting_epoch,
            "global_steps": global_steps
        }
        if noise_scheduling == NoiseScheduler.LINEAR:
            config_state["beta_1"] = beta_1
            config_state["beta_T"] = beta_T
        save_model(
            model_net=config_state,
            file_name="config",
            dest_path=out_dir,
            checkpoint=True,
            steps=global_steps)
        
        # Save Diffusion Net.
        diffusion_state = {
            "model": diffusion_net.state_dict(),
            "optimizer": diffusion_optim.state_dict(),}
        save_model(
            model_net=diffusion_state,
            file_name="diffusion",
            dest_path=out_dir,
            checkpoint=True,
            steps=global_steps)
        
        avg_diffusion = total_diffusion_loss / training_count
        message = "Epoch: {:,} | Diffusion: {:.5f} | LR: {:.9f}".format(
            epoch,
            avg_diffusion,
            diffusion_optim.param_groups[0]['lr']
        )
        logging.info(message)

if __name__ == "__main__":
    main()
