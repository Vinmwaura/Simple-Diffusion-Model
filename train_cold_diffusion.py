import os
import sys
import csv
import glob
import random
import logging

import torch
torch.manual_seed(69)

import torchvision
import torch.nn.functional as F

# U Net Model.
from models.U_Net import U_Net

# Custom Image Dataset Loader.
from img_dataset import ImageDataset

# Noise Degradation.
from noise_degradation import NoiseDegradation

from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    project_name = "cold_diffusion_noise"

    # Training Params.
    starting_epoch = 0
    global_steps = 0
    checkpoint_steps = 1000  # Global steps in between checkpoints
    lr_steps = 100_000  # Global steps in between halving learning rate.
    max_epoch = 1000
    dataset_path = ""
    out_dir = ""
    diffusion_checkpoint = None
    config_checkpoint = None

    # Model Params.
    diffusion_lr = 2e-5
    batch_size = 14
    
    # Diffusion Params.
    beta = 5e-3
    min_noise_step = 1
    max_noise_step = 1000

    os.makedirs(out_dir, exist_ok=True)

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

    # Model.
    diffusion_net = U_Net().to(device)

    # Initialize gradient scaler.
    scaler = torch.cuda.amp.GradScaler()

    # List of image dataset.
    img_regex = os.path.join(dataset_path, "*.jpg")
    img_list = glob.glob(img_regex)

    assert len(img_list) > 0
    
    # Dataset and DataLoader.
    dataset = ImageDataset(img_paths=img_list)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True)

    # Model Optimizer.
    diffusion_optim = torch.optim.Adam(
        diffusion_net.parameters(),
        lr=diffusion_lr,
        betas=(0.5, 0.999))

    # Load Config Checkpoints.
    if config_checkpoint is not None:
        config_status, config_dict = load_checkpoint(config_checkpoint)
        assert config_status

        beta = config_dict["beta"]
        min_noise_step = config_dict["min_noise_step"]
        max_noise_step = config_dict["max_noise_step"]
        starting_epoch = config_dict["starting_epoch"]
        global_steps = config_dict["global_steps"]

    # Load Diffusion Model Checkpoints.
    if diffusion_checkpoint is not None:
        diffusion_status, diffusion_dict= load_checkpoint(diffusion_checkpoint)
        assert diffusion_status

        diffusion_net.load_state_dict(diffusion_dict["model"])
        diffusion_optim.load_state_dict(diffusion_dict["optimizer"])

    logging.info("#" * 100)
    logging.info(f"Train Parameters:")
    logging.info(f"Max Epoch: {max_epoch:,}")
    logging.info(f"Dataset Path: {dataset_path}")
    logging.info(f"Output Path: {out_dir}")
    logging.info(f"Checkpoint Steps: {checkpoint_steps}")
    logging.info("#" * 100)
    logging.info(f"Model Parameters:")
    logging.info(f"Learning Rate: {diffusion_optim.param_groups[0]['lr']:.5f}")
    logging.info(f"Batch size: {batch_size:,}")
    logging.info("#" * 100)
    logging.info(f"Diffusion Parameters:")
    logging.info(f"Diffusion Beta: {beta:,.3f}")
    logging.info(f"Min Noise Step: {min_noise_step:,}")
    logging.info(f"Max Noise Step: {max_noise_step:,}")
    logging.info("#" * 100)

    # x_t = D(x_0 | t).
    noise_degradation = NoiseDegradation(beta=beta)

    for epoch in range(starting_epoch, max_epoch):
        # Diffusion Loss.
        total_diffusion_loss = 0

        # Number of iterations.
        training_count = 0

        for index, tr_data in enumerate(dataloader):
            training_count += 1
            tr_data = tr_data.to(device)

            #################################################
            #             Diffusion Training.               #
            #################################################
            diffusion_optim.zero_grad()

            # Random Noise Step(t).
            rand_noise_step = torch.randint(
                low=min_noise_step,
                high=max_noise_step,
                size=(len(tr_data), ),
                device=device)
            
            # Enable autocasting for mixed precision.
            with torch.cuda.amp.autocast():
                # eps Noise.
                noise = torch.randn_like(tr_data)

                # D(x_0, t) = x_t
                x_degraded = noise_degradation(
                    img=tr_data,
                    step=rand_noise_step,
                    eps=noise)
                
                # R(x_t, t) = x_0
                x_restoration = diffusion_net(x_degraded, rand_noise_step)
                
                # min || R(D(x,t), t) - x ||
                restoration_loss = F.l1_loss(
                    x_restoration,
                    tr_data)
            
            # Scale the loss and do backprop.
            scaler.scale(restoration_loss).backward()
            
            # Update the scaled parameters.
            scaler.step(diffusion_optim)

            # Update the scaler for next iteration
            scaler.update()
            
            total_diffusion_loss += restoration_loss.item()

            if global_steps % lr_steps == 0 and global_steps > 0:
                # Update Diffusion LR.
                for diffusion in diffusion_optim.param_groups:
                    diffusion['lr'] = diffusion['lr'] * 0.5
            
            # Checkpoint and Plot Images.
            if global_steps % checkpoint_steps == 0 and global_steps >= 0:
                config_state = {
                    "beta": beta,
                    "min_noise_step": min_noise_step,
                    "max_noise_step": max_noise_step,
                    "starting_epoch": starting_epoch,
                    "global_steps": global_steps}

                # Save Config Net.
                save_model(
                    model_net=config_state,
                    file_name="config",
                    dest_path=out_dir,
                    checkpoint=True,
                    steps=global_steps)

                diffusion_state = {
                    "model": diffusion_net.state_dict(),
                    "optimizer": diffusion_optim.state_dict(),}

                # Save Diffusion Net.
                save_model(
                    model_net=diffusion_state,
                    file_name="diffusion",
                    dest_path=out_dir,
                    checkpoint=True,
                    steps=global_steps)
                
                # Plot Sample Images.
                """
                Generation using deterministic noise degradation. Noise pattern is selected and
                frozen at the start of the generation process, and then treated as a constant.
                """
                with torch.no_grad():
                    noise = torch.randn((25, 3, 128, 128), device=device)
                    x_degraded_hat = 1 * noise
                    
                    for noise_step in range(max_noise_step, min_noise_step - 1, -1):
                        time_step = torch.tensor([noise_step], device=device)
                        
                        # R(x_s, s) = x_hat_0
                        x_restoration_hat = diffusion_net(
                            x_degraded_hat,
                            time_step)

                        if noise_step - 1 > 0:
                            # D(x_hat_0, s) = x_s
                            x_degraded_current_noise_lvl = noise_degradation(
                                img=x_restoration_hat,
                                step=noise_step,
                                eps=noise)
                            
                            # D(x_hat_0, s-1) = x_s-1
                            x_degraded_next_noise_lvl = noise_degradation(
                                img=x_restoration_hat,
                                step=noise_step - 1,
                                eps=noise)

                        x_degraded_hat = x_degraded_hat - x_degraded_current_noise_lvl + x_degraded_next_noise_lvl
                    
                    plot_sampled_images(
                        sampled_imgs=x_degraded_hat,
                        file_name=f"diffusion_plot_{global_steps}",
                        dest_path=out_dir)
            
            temp_avg_diffusion = total_diffusion_loss / training_count

            message = "Cum. Steps: {:,} | Steps: {:,} / {:,} | Diffusion: {:.5f} | lr: {:.9f}".format(
                global_steps + 1,
                index + 1,
                len(dataloader),
                temp_avg_diffusion, 
                diffusion_optim.param_groups[0]['lr']
            )
            logging.info(message)

            global_steps += 1

        # Checkpoint and Plot Images.
        config_state = {
            "beta": beta,
            "min_noise_step": min_noise_step,
            "max_noise_step": max_noise_step,
            "starting_epoch": starting_epoch,
            "global_steps": global_steps
        }

        # Save Config Net.
        save_model(
            model_net=config_state,
            file_name="config",
            dest_path=out_dir,
            checkpoint=True,
            steps=global_steps)

        diffusion_state = {
            "model": diffusion_net.state_dict(),
            "optimizer": diffusion_optim.state_dict(),
        }

        # Save Diffusion Net.
        save_model(
            model_net=diffusion_state,
            file_name="diffusion",
            dest_path=out_dir,
            checkpoint=True,
            steps=global_steps)
        
        avg_diffusion = total_diffusion_loss / training_count

        message = "Epoch: {:,} | Diffusion: {:.5f} | lr: {:.9f}".format(
            epoch,
            avg_diffusion, 
            diffusion_optim.param_groups[0]['lr']
        )
        logging.info(message)

if __name__ == "__main__":
    main()
