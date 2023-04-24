import os
import csv
import logging

import torch
torch.manual_seed(69)

import torchvision
import torch.nn.functional as F

# U Net Model.
from models.U_Net import U_Net

# Custom Image Dataset Loader.
from classifier_img_dataset import ClassifierImgDataset

# Degradation Operators.
from degraders import *

# Enums.
from diffusion_enums import NoiseScheduler

from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    project_name = "Conditional-Diffusion"

    # Model Params.
    in_channel = 3
    out_channel = in_channel
    num_layers = 5
    attn_layers = [2, 3, 4]
    time_dim = 64
    cond_dim = 21
    min_channel = 128
    max_channel = 512
    img_recon = False

    # Training Params.
    starting_epoch = 0
    global_steps = 0
    checkpoint_steps = 1_000  # Global steps in between checkpoints
    lr_steps = 50_000  # Global steps in between halving learning rate.
    max_epoch = 1_000
    plot_img_count = 25

    # JSON database (NOSQL)
    dataset_path = None
    out_dir = None

    assert dataset_path is not None
    assert out_dir is not None
    os.makedirs(out_dir, exist_ok=True)

    # Load Pre-trained optimization configs, ignored if no checkpoint is passed.
    load_diffusion_optim = False

    # Checkpoint Filepaths.
    diffusion_checkpoint = None
    config_checkpoint = None

    # Model Params.
    diffusion_lr = 2e-5
    batch_size = 8
    
    # Diffusion Params.
    # Linear, Cosine Schedulers
    noise_scheduling = NoiseScheduler.LINEAR
    
    if noise_scheduling == NoiseScheduler.LINEAR:
        beta_1 = 5e-3
        beta_T = 9e-3
    min_noise_step = 1  # t_1
    max_noise_step = 1_000  # T

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
    diffusion_net = U_Net(
        in_channel=in_channel,
        out_channel=out_channel,
        num_layers=num_layers,
        attn_layers=attn_layers,
        time_dim=time_dim,
        cond_dim=cond_dim,
        min_channel=min_channel,
        max_channel=max_channel,
        image_recon=img_recon)

    # Initialize gradient scaler.
    scaler = torch.cuda.amp.GradScaler()
    
    # Dataset and DataLoader.
    dataset = ClassifierImgDataset(dataset_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True)

    # Gets Conditional information for plotting sampling from data.
    plot_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=plot_img_count,
        num_workers=4,
        shuffle=True)
    _, plot_labels = next(iter(plot_dataloader))

    labels_path = os.path.join(out_dir, "labels.txt")

    header = dataset.get_labels()
    plot_values = plot_labels.tolist()
    csv_data = [header] + plot_values
    with open(labels_path, "a") as f:
        wr = csv.writer(f)
        wr.writerows(csv_data)

    # Load Diffusion Model Checkpoints.
    if diffusion_checkpoint is not None:
        diffusion_status, diffusion_dict= load_checkpoint(diffusion_checkpoint)
        assert diffusion_status

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
        config_status, config_dict = load_checkpoint(config_checkpoint)
        assert config_status

        if noise_scheduling == NoiseScheduler.LINEAR:
            beta_1 = config_dict["beta_1"]
            beta_T = config_dict["beta_T"]
        min_noise_step = config_dict["min_noise_step"]
        max_noise_step = config_dict["max_noise_step"]
        starting_epoch = config_dict["starting_epoch"]
        global_steps = config_dict["global_steps"]

    # x_t(X_0, eps) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.
    if noise_scheduling == NoiseScheduler.LINEAR:        
        noise_degradation = NoiseDegradation(
            beta_1,
            beta_T,
            max_noise_step,
            device)
    elif noise_scheduling == NoiseScheduler.COSINE:
        noise_degradation = CosineNoiseDegradation(max_noise_step)
    
    # Transformation Augmentations.
    hflip_transformations = torchvision.transforms.RandomHorizontalFlip(p=0.5)

    logging.info("#" * 100)
    logging.info(f"Train Parameters:")
    logging.info(f"Max Epoch: {max_epoch:,}")
    logging.info(f"Classifier Dataset Path: {dataset_path}")
    logging.info(f"Output Path: {out_dir}")
    logging.info(f"Checkpoint Steps: {checkpoint_steps:,}")
    logging.info("#" * 100)
    logging.info(f"Model Parameters:")
    logging.info(f"Diffusion LR: {diffusion_optim.param_groups[0]['lr']:.5f}")
    logging.info(f"Batch size: {batch_size:,}")
    logging.info("#" * 100)
    logging.info(f"Diffusion Parameters:")
    if noise_scheduling == NoiseScheduler.LINEAR:
        logging.info(f"beta_1: {beta_1:,.5f}")
        logging.info(f"beta_T: {beta_T:,.5f}")
    logging.info(f"Min Noise Step: {min_noise_step:,}")
    logging.info(f"Max Noise Step: {max_noise_step:,}")
    logging.info("#" * 100)

    for epoch in range(starting_epoch, max_epoch):
        # Diffusion Loss.
        total_diffusion_loss = 0

        # Number of iterations.
        training_count = 0

        for index, (tr_data, labels) in enumerate(dataloader):
            training_count += 1
            
            labels = labels.to(device)
            tr_data = tr_data.to(device)

            hflip_data = hflip_transformations(tr_data)

            N, C, H, W = hflip_data.shape

            #################################################
            #               Diffusion Training.             #
            #################################################
            diffusion_optim.zero_grad()

            # Random Noise Step(t).
            rand_noise_step = torch.randint(
                low=min_noise_step,
                high=max_noise_step,
                size=(N, ),
                device=device)
            
            # eps Noise.
            noise = torch.randn_like(hflip_data)
    
            # Train model.
            diffusion_net.train()

            # Enable autocasting for mixed precision.
            with torch.cuda.amp.autocast():
                # Noise degraded image (x_t).
                # x_t(X_0, eps) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.
                x_t = noise_degradation(
                    img=hflip_data,
                    steps=rand_noise_step,
                    eps=noise)

                noise_approx = diffusion_net(
                    x_t,
                    rand_noise_step,
                    labels)
                
                diffusion_loss = F.mse_loss(
                    noise_approx,
                    noise)
                
                assert not torch.isnan(diffusion_loss)

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
                    "min_noise_step": min_noise_step,
                    "max_noise_step": max_noise_step,
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
                diffusion_net.eval()  # Evaluate model.

                # X_T ~ N(0, I)
                noise = torch.randn((plot_img_count, C, H, W), device=device)
                x_t_plot = 1 * noise
                plot_labels = plot_labels.to(device)

                with torch.no_grad():
                    steps = list(range(max_noise_step, min_noise_step - 1, -10))

                    # 0 - Deterministic
                    # 1 - DDPM
                    eta = 0.0

                    for count in range(len(steps)):
                        # t: Time Step
                        t = torch.tensor([steps[count]], device=device).repeat(plot_img_count, )

                        # Reconstruction: (x0_hat).
                        noise_approx_plot = diffusion_net(
                            x_t_plot,
                            t,
                            plot_labels)

                        # Variables needed in computing x_t.
                        _, _, alpha_bar_t = noise_degradation.get_timestep_params(t)
                        alpha_bar_t = alpha_bar_t[:, None, None, None]
                        # Approximates x0 using x_t and eps_theta(x_t, t).
                        # x_t - sqrt(1 - alpha_bar_t) * eps_theta(x_t, t) / sqrt(alpha_bar_t).
                        scale = 1 / alpha_bar_t**0.5
                        x0_approx = scale * (x_t_plot - ((1 - alpha_bar_t)**0.5 * noise_approx_plot))

                        if count < len(steps) - 1:
                            tm1 = torch.tensor([steps[count + 1]], device=device)

                            # Variables needed in computing x_tm1.
                            _, _, alpha_bar_tm1 = noise_degradation.get_timestep_params(tm1)

                            # sigma = eta * (sqrt(1 - alpha_bar_tm1 / 1 - alpha_bar_t) * sqrt(1 - alpha_bar_t / alpha_bar_tm1)).
                            sigma = eta * (((1 - alpha_bar_tm1) / (1 - alpha_bar_t))**0.5 * (1 - (alpha_bar_t / alpha_bar_tm1))**0.5)
                            
                            # Noise to be added (Reparameterization trick).
                            eps = torch.randn_like(x0_approx)

                            # As implemented in "Denoising Diffusion Implicit Models" paper.
                            # x0_predicted = (1/sqrt(alpha_bar_t) * x_t - sqrt(1 - alpha_bar_t)) * eps_theta
                            # xt_direction = sqrt(1 - alpha_bar_tm1 - sigma^2 * eps_theta)
                            # random_noise = sqrt(sigma_squared) * eps
                            # x_tm1 = sqrt(alpha_bar_t-1) * x0_predicted + xt_direction + random_noise
                            x_t_plot = (alpha_bar_tm1**0.5 * x0_approx) + ((1 - alpha_bar_tm1 - sigma**2)**0.5 * noise_approx_plot) + (sigma * eps)

                        printProgressBar(
                            max_noise_step - steps[count],
                            max_noise_step,
                            prefix = 'Iterations:',
                            suffix = 'Complete',
                            length = 50)

                    plot_sampled_images(
                        sampled_imgs=x0_approx,  # x_0
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
            "min_noise_step": min_noise_step,
            "max_noise_step": max_noise_step,
            "starting_epoch": starting_epoch,
            "global_steps": global_steps
        }
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

        avg_diffusion = total_diffusion_loss / training_count
        message = "Epoch: {:,} | Diffusion: {:.5f} | lr: {:.9f}".format(
            epoch,
            avg_diffusion,
            diffusion_optim.param_groups[0]['lr'])
        logging.info(message)

if __name__ == "__main__":
    main()
