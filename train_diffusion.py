import os
import csv
import glob
import logging

import torch
import torchvision
import torch.nn.functional as F

# U Net Model.
from models.U_Net import U_Net

from diffusion_enums import *

# Degradation Operators.
from degraders import *

from utils.utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    project_name = "Diffusion"

    # Training Params.
    starting_epoch = 0
    global_steps = 0
    checkpoint_steps = 1_000  # Global steps in between checkpoints
    lr_steps = 100_000  # Global steps in between halving learning rate.
    max_epoch = 1_000
    plot_img_count = 25
    use_conditional = True  # Embed conditional information into the model i.e One-hot encoding.
    flip_imgs = False  # Toggles augmenting the images by randomly flipping images horizontally.
    
    dataset_path = None
    out_dir = None
    assert dataset_path is not None
    assert out_dir is not None
    os.makedirs(out_dir, exist_ok=True)

    # Checkpoints.
    diffusion_checkpoint = None
    config_checkpoint = None

    # Model Params.
    diffusion_lr = (2e-5) * (0.5**0)
    batch_size = 20

    # Linear, Cosine Schedulers
    noise_scheduling = NoiseScheduler.LINEAR
    if noise_scheduling == NoiseScheduler.LINEAR:
        beta_1 = 5e-3
        beta_T = 9e-3
    diffusion_alg = DiffusionAlg.DDIM

    min_noise_step = 1  # t_1
    max_noise_step = 1_000  # T
    max_actual_noise_step = 1_000  # Max timestep used in training step (For ensemble models training).
    skip_step = 10  # Step to be skipped when sampling.
    assert max_actual_noise_step > min_noise_step
    assert max_noise_step > min_noise_step
    assert skip_step < max_actual_noise_step and skip_step > 0
    assert min_noise_step > 0

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
    # Custom Image Dataset Loader.
    if use_conditional:
        from custom_dataset.conditional_img_dataset import ConditionalImgDataset
        dataset = ConditionalImgDataset(dataset_path=dataset_path)
    else:
        from custom_dataset.img_dataset import ImageDataset

        # List of image dataset.
        img_regex = os.path.join(dataset_path, "*.jpg")
        img_list = glob.glob(img_regex)

        assert len(img_list) > 0

        dataset = ImageDataset(img_paths=img_list)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True)
    plot_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=plot_img_count,
        num_workers=1,
        shuffle=False)
    
    if use_conditional:
        plot_imgs, plot_labels = next(iter(plot_dataloader))
        labels_path = os.path.join(out_dir, "labels.txt")

        header = dataset.get_labels()
        plot_values = plot_labels.cpu().tolist()
        csv_data = [header] + plot_values

        with open(labels_path, "a") as f:
            wr = csv.writer(f)
            wr.writerows(csv_data)
    else:
        plot_imgs = next(iter(plot_dataloader))
        plot_labels = None

    # Model Params.
    in_channel = 3
    out_channel = 3
    num_layers = 4
    num_resnet_block = 1
    attn_layers = [2, 3]
    attn_heads = 1
    attn_dim_per_head = None
    time_dim = 512
    cond_dim = None
    min_channel = 128
    max_channel = 512
    img_recon = False

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
    load_diffusion_optim = False

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

        for index, data in enumerate(dataloader):
            training_count += 1

            if use_conditional:
                tr_data, labels = data

                tr_data = tr_data.to(device)
                labels = labels.to(device)
            else:
                tr_data = data.to(device)
                labels = None  # No label placeholder.
            
            N, C, H, W = tr_data.shape

            # eps Noise.
            noise = torch.randn_like(tr_data)

            if flip_imgs:
                tr_data = torchvision.transforms.Lambda(
                    lambda x: torch.stack([hflip_transformations(x_) for x_ in x]))(tr_data)

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
        
                # Predicts noise from x_t.
                # eps_param(sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps, t). 
                noise_approx = diffusion_net(
                    x_t,
                    rand_noise_step,
                    labels)
                
                # Simplified Training Objective.
                # L_simple(param) = E[||eps - eps_param(sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps, t).||^2]
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

                if plot_labels is not None:
                    plot_labels = plot_labels.to(device)

                if diffusion_alg == DiffusionAlg.DDPM:
                    with torch.no_grad():
                        for noise_step in range(max_actual_noise_step, min_noise_step - 1, -1):
                            # t: Time Step
                            t = torch.tensor([noise_step], device=device)

                            # Variables needed in computing x_(t-1).
                            beta_t, alpha_t, alpha_bar_t = noise_degradation.get_timestep_params(t)
                            
                            # eps_param(x_t, t).
                            noise_approx = diffusion_net(
                                x_t_plot,
                                t,
                                plot_labels)

                            # z ~ N(0, I) if t > 1, else z = 0.
                            if noise_step > 1:
                                z = torch.randn((plot_img_count, C, H, W), device=device)
                            else:
                                z = 0
                            
                            # sigma_t ^ 2 = beta_t = beta_hat = (1 - alpha_bar_(t-1)) / (1 - alpha_bar_t) * beta_t
                            sigma_t = beta_t ** 0.5

                            # x_(t-1) = (1 / sqrt(alpha_t)) * (x_t - (1 - alpha_t / sqrt(1 - alpha_bar_t)) * eps_param(x_t, t)) + sigma_t * z
                            scale_1 = 1 / (alpha_t ** 0.5)
                            scale_2 = (1 - alpha_t) / ((1 - alpha_bar_t)**0.5)
                            
                            # x_(t-1).
                            x_t_plot = scale_1 * (x_t_plot - (scale_2 * noise_approx)) + (sigma_t * z)

                            printProgressBar(
                                max_actual_noise_step - noise_step,
                                max_actual_noise_step - (min_noise_step - 1),
                                prefix = "Iterations:",
                                suffix = "Complete",
                                length = 50)

                        plot_sampled_images(
                            sampled_imgs=x_t_plot,  # x_0
                            file_name=f"diffusion_plot_{global_steps}",
                            dest_path=out_dir)

                elif diffusion_alg == DiffusionAlg.DDIM:
                    with torch.no_grad():
                        steps = list(range(max_actual_noise_step, min_noise_step - 1, -skip_step))
                        
                        # Includes timestep 1 into the steps if not included.
                        if 1 not in steps:
                            steps = steps + [1]

                        # 0 - Deterministic
                        # 1 - DDPM
                        eta = 0.0

                        for count in range(len(steps)):
                            # t: Time Step
                            t = torch.tensor([steps[count]], device=device)

                            # eps_theta(x_t, t).
                            noise_approx = diffusion_net(
                                x_t_plot,
                                t,
                                plot_labels)

                            # Variables needed in computing x_t.
                            _, _, alpha_bar_t = noise_degradation.get_timestep_params(t)
                            
                            # Approximates x0 using x_t and eps_theta(x_t, t).
                            # x_t - sqrt(1 - alpha_bar_t) * eps_theta(x_t, t) / sqrt(alpha_bar_t).
                            scale = 1 / alpha_bar_t**0.5
                            x0_approx = scale * (x_t_plot - ((1 - alpha_bar_t)**0.5 * noise_approx))

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
                                x_t_plot = (alpha_bar_tm1**0.5 * x0_approx) + ((1 - alpha_bar_tm1 - sigma**2)**0.5 * noise_approx) + (sigma * eps)
                            
                                printProgressBar(
                                    max_actual_noise_step - steps[count],
                                    max_actual_noise_step - (min_noise_step - 1),
                                    prefix = "Iterations:",
                                    suffix = "Complete",
                                    length = 50)

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

        # Checkpoint and Plot Images.
        # Save Config Net.
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
