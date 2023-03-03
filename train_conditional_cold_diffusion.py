import os
import glob
import logging

import torch
torch.manual_seed(69)

import torchvision
import torch.nn.functional as F

# U Net Model.
from models.U_Net import U_Net

# Custom Image Dataset Loader.
from img_dataset import ImageDataset

# Degradation Operators.
from degraders import *

# Enums.
from diffusion_enums import NoiseScheduler

from utils import *
from kmeans_utils import *

from diffusion_enums import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    project_name = "Conditional-Cold-Diffusion"

    # Training Params.
    starting_epoch = 0
    global_steps = 0
    checkpoint_steps = 1_000  # Global steps in between checkpoints
    lr_steps = 50_000  # Global steps in between halving learning rate.
    max_epoch = 1_000
    plot_img_count = 25
    dataset_path = None
    out_dir = None

    assert dataset_path is not None
    assert out_dir is not None
    os.makedirs(out_dir, exist_ok=True)

    # Load Pre-trained optimization configs, ignored if no checkpoint is passed.
    load_diffusion_optim = False

    # Checkpoints.
    conditional_checkpoint = None
    diffusion_checkpoint = None
    config_checkpoint = None

    # Model Params.
    diffusion_lr = 2e-5
    batch_size = 8
    
    # Diffusion Params.
    # Linear, Cosine Schedulers
    noise_scheduling = NoiseScheduler.COSINE
    
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
        in_channel=6,
        out_channel=3,
        num_layers=5,
        attn_layers=[2, 3, 4],
        time_channel=64,
        min_channel=128,
        max_channel=512,
        image_recon=True)

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

    # Load Conditional Checkpoints.
    centroids_status, centroids_dict= load_checkpoint(conditional_checkpoint)
    assert centroids_status
    K = centroids_dict["K"]
    centroids = centroids_dict["centroids"]
    centroids = centroids.to(device)

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
    logging.info(f"Dataset Path: {dataset_path}")
    logging.info(f"Output Path: {out_dir}")
    logging.info(f"Checkpoint Steps: {checkpoint_steps}")
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
        
        for index, tr_data in enumerate(dataloader):
            training_count += 1
            
            tr_data = tr_data.to(device)

            hflip_data = hflip_transformations(tr_data)

            N, C, H, W = hflip_data.shape

            # TODO: Allow  for alternative conditional input like segmentation masks,
            # depth maps etc to giude diffusion.
            data_ = hflip_data.unsqueeze(0).to(device)  # (1, N, C, H, W)
            distance = compute_rmse(centroids, data_)[:, :, None, None, None]
            min_distance_index = torch.argmin(distance, dim=0).squeeze()
            mask = F.one_hot(min_distance_index, num_classes=K).T
            mask = mask[:, :, None, None, None]
            centroids_bm = torch.sum((mask * centroids), dim=0)

            # eps Noise.
            # Uses Uniform distribution: [-1, 1] instead of Gaussian distribution, seems to work just as well.
            noise = 2 * torch.rand_like(hflip_data) - 1

            #################################################
            #           Cold Diffusion Training.            #
            #################################################
            diffusion_optim.zero_grad()

            # Train model.
            diffusion_net.train()

            # Random Noise Step(t).
            rand_noise_step = torch.randint(
                low=min_noise_step,
                high=max_noise_step,
                size=(N, ),
                device=device)

            # Enable autocasting for mixed precision.
            with torch.cuda.amp.autocast():
                # Noise degraded image (x_t).
                # x_t(X_0, eps) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.
                x_t = noise_degradation(
                    img=hflip_data,
                    steps=rand_noise_step,
                    eps=noise)

                x_t_condition = torch.cat((x_t, centroids_bm), dim=1)

                x0_approx_recon = diffusion_net(
                    x_t_condition,
                    rand_noise_step)

                diffusion_loss = F.mse_loss(
                    x0_approx_recon,
                    hflip_data)

                assert not torch.isnan(diffusion_loss)

            # Scale the loss and do backprop.
            scaler.scale(diffusion_loss).backward()
            
            # Update the scaled parameters.
            scaler.step(diffusion_optim)

            # Update the scaler for next iteration
            scaler.update()
            
            total_diffusion_loss += diffusion_loss.item()

            if global_steps % lr_steps == 0 and global_steps > 0:
                # Halves Diffusion LR after N steps.
                for diffusion_lr in diffusion_optim.param_groups:
                    diffusion_lr['lr'] = diffusion_lr['lr'] * 0.5

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

                centroid_indices = torch.randint(low=0, high=K, size=(plot_img_count,))
                conditional_plot = centroids[centroid_indices].squeeze(1)

                # X_T ~ N(0, I).
                noise = 2 * torch.rand((plot_img_count, C, H, W), device=device) - 1
                x_t_plot = 1 * noise

                with torch.no_grad():
                    steps = list(range(max_noise_step, min_noise_step - 1, -10)) + [1]

                    for count in range(len(steps)):
                        # t: Time Step
                        t = torch.tensor([steps[count]], device=device)

                        # Reconstruction: (x0_hat).
                        x_t_conditional = torch.cat((x_t_plot, conditional_plot), dim=1)
                        x0_recon_approx_plot = diffusion_net(
                            x_t_conditional,
                            t)

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
                            x_t_plot = x_t_plot - x_t_hat_plot + x_tm1_hat_plot

                        printProgressBar(
                            max_noise_step - steps[count],
                            max_noise_step,
                            prefix = 'Iterations:',
                            suffix = 'Complete',
                            length = 50)
                        
                    plot_sampled_images(
                        sampled_imgs=x0_recon_approx_plot,  # x_0
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
