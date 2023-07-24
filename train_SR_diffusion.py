import os
import csv
import glob
import logging
import random

import torch

import torchvision
import torch.nn.functional as F

# U Net Model.
from models.U_Net import U_Net

# Degradation Operators.
from degraders import *

# Enums.
from diffusion_enums import NoiseScheduler

from utils.utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    # Loosely similar to SRDiff: Single Image Super-Resolution with Diffusion Probabilistic Models.
    project_name = "SR-Cold-Diffusion"

    # SR Params.
    lr_dim = 128
    sr_dim = 256

    # Training Params.
    starting_epoch = 0
    global_steps = 0
    checkpoint_steps = 1_000  # Global steps in between checkpoints
    lr_steps = 100_000  # Global steps in between halving learning rate.
    max_epoch = 1_000
    plot_img_count = 8
    use_conditional = True  # Embed conditional information into the model i.e One-hot encoding.
    flip_imgs = False  # Toggles augmenting the images by randomly flipping images horizontally.

    dataset_path = None
    out_dir = None
    assert dataset_path is not None
    assert out_dir is not None
    os.makedirs(out_dir, exist_ok=True)

    # Load Pre-trained optimization configs, ignored if no checkpoint is passed.
    load_diffusion_optim = False

    diffusion_checkpoint = None
    config_checkpoint = None

    # Model Params.
    diffusion_lr = (2e-5) * (0.5**0)
    batch_size = 4
    
    # Diffusion Params.
    # Linear, Cosine Schedulers
    noise_scheduling = NoiseScheduler.COSINE
    
    if noise_scheduling == NoiseScheduler.LINEAR:
        beta_1 = 5e-3
        beta_T = 9e-3
    min_noise_step = 1  # t_1
    max_noise_step = 1_000  # Max timestep used in noise scheduler.
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

    # Model Params.
    in_channel = 6
    out_channel = 3
    num_layers = 4
    num_resnet_block = 1
    attn_layers = [2, 3]
    attn_heads = 1
    attn_dim_per_head = None
    time_dim = 512
    cond_dim = 4
    min_channel = 128
    max_channel = 512
    img_recon = True

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

        random.shuffle(img_list)

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

        plot_labels = plot_labels.to(device)
    else:
        plot_imgs = next(iter(plot_dataloader))
        plot_labels = None

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

    logging.info(f"SR Parameters:")
    logging.info(f"Low Resolution Dim: {lr_dim:,}")
    logging.info(f"Super Resolution Dim: {sr_dim:,}")
    logging.info("#" * 100)
    logging.info(f"Train Parameters:")
    logging.info(f"Max Epoch: {max_epoch:,}")
    logging.info(f"Classifier Dataset Path: {dataset_path}")
    logging.info(f"Output Path: {out_dir}")
    logging.info(f"Checkpoint Steps: {checkpoint_steps:,}")
    logging.info(f"Batch size: {batch_size:,}")
    logging.info(f"Diffusion LR: {diffusion_optim.param_groups[0]['lr']:.5f}")
    logging.info(f"Use Conditional: {use_conditional}")
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
        logging.info(f"beta_1: {beta_1:,.5f}")
        logging.info(f"beta_T: {beta_T:,.5f}")
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

            if flip_imgs:
                tr_data = torchvision.transforms.Lambda(
                    lambda x: torch.stack([hflip_transformations(x_) for x_ in x]))(tr_data)

            N, C, H, W = tr_data.shape

            # Low Resolution Images.
            lr_data = F.interpolate(
                tr_data,
                size=(lr_dim, lr_dim),
                mode="area")
            lr_data = F.interpolate(
                lr_data,
                size=(sr_dim, sr_dim),
                mode="area")

            # eps Noise.
            noise = torch.randn_like(tr_data)

            #################################################
            #               Diffusion Training.             #
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

            # Enable autocasting for mixed precision.
            with torch.cuda.amp.autocast():
                diff_data = tr_data - lr_data

                # Noise degraded image (x_t).
                # x_t(X_0, eps) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.
                x_t_sr = noise_degradation(
                    img=tr_data,
                    steps=rand_noise_step,
                    eps=noise)

                # Noise degraded image (x_t).
                # x_t(X_0, eps) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.
                x_t_lr = noise_degradation(
                    img=lr_data,
                    steps=torch.tensor([250], device=device),
                    eps=noise)
                
                x_t_combined = torch.cat((x_t_sr, x_t_lr), dim=1)
                x0_approx_recon = diffusion_net(
                    x_t_combined,
                    rand_noise_step,
                    labels)

                diffusion_loss = F.mse_loss(
                    x0_approx_recon,
                    diff_data)
                
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

                # X_T ~ N(0, I)
                noise = torch.randn((plot_img_count, C, H, W), device=device)

                if max_actual_noise_step < max_noise_step:
                    plot_imgs = plot_imgs.to(device)
                    x_t_plot = noise_degradation(
                        img=plot_imgs,
                        steps=torch.tensor([max_actual_noise_step], device=device),
                        eps=noise)
                else:
                    # Low Resolution Images.
                    plot_imgs = plot_imgs.to(device)
                    lr_data_plot = F.interpolate(
                        plot_imgs,
                        size=(lr_dim, lr_dim),
                        mode="area")
                    lr_data_plot = F.interpolate(
                        lr_data_plot,
                        size=(sr_dim, sr_dim),
                        mode="area")
                    x_t_plot_lr = noise_degradation(
                        img=lr_data_plot,
                        steps=torch.tensor([250], device=device),
                        eps=noise)
                    x_t_plot_sr = 1 * noise

                with torch.no_grad():
                    steps = list(range(max_actual_noise_step, min_noise_step - 1, -skip_step)) + [1]

                    # Includes timestep 1 into the steps if not included.
                    if 1 not in steps:
                        steps = steps + [1]

                    for count in range(len(steps)):
                        # t: Time Step
                        t = torch.tensor([steps[count]], device=device)

                        x_t_plot = torch.cat((x_t_plot_sr, x_t_plot_lr), dim=1)

                        # Reconstruction: (x0_hat).
                        # x_t_combined = torch.cat((x_t_plot, eps_diffusion_approx), dim=1)
                        x0_recon_approx_plot = diffusion_net(
                            x_t_plot,
                            t,
                            plot_labels)

                        if count < len(steps) - 1:
                            # t-1: Time Step
                            tm1 = torch.tensor([steps[count + 1]], device=device)

                            # D(x0_hat, t).
                            # Noise degraded image (x_t).
                            # x_t(X_0, eps) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.
                            x_t_hat_plot = noise_degradation(
                                img=x0_recon_approx_plot,
                                steps=t,
                                eps=noise)

                            # D(x0_hat, t-1).
                            # Noise degraded image (x_t).
                            # x_t(X_0, eps) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.
                            x_tm1_hat_plot = noise_degradation(
                                img=x0_recon_approx_plot,
                                steps=tm1,
                                eps=noise)
                            
                            # q(x_t-1 | x_t, x_0).
                            # Improved sampling from Cold Diffusion paper.
                            x_t_plot_sr = x_t_plot_sr - x_t_hat_plot + x_tm1_hat_plot

                            printProgressBar(
                                max_actual_noise_step - steps[count],
                                max_actual_noise_step - (min_noise_step - 1),
                                prefix = 'Iterations:',
                                suffix = 'Complete',
                                length = 50)

                    plot_sampled_images(
                        sampled_imgs=x0_recon_approx_plot + lr_data_plot,  # x_0
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

        avg_diffusion = total_diffusion_loss / training_count
        message = "Epoch: {:,} | Diffusion: {:.5f} | LR: {:.9f}".format(
            epoch,
            avg_diffusion,
            diffusion_optim.param_groups[0]['lr'])
        logging.info(message)

if __name__ == "__main__":
    main()
