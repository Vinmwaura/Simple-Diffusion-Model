import os
import json
import pathlib
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
import torchvision

from degraders import *
from models.U_Net import U_Net
from diffusion_enums import *
from utils import load_checkpoint, plot_sampled_images, printProgressBar

def main():
    parser = argparse.ArgumentParser(
        description="Generate Images using diffusion models.")
    
    # Trainning Params.
    parser.add_argument("--seed", help = "Seed value for generating image(Default: None).", type=int, default=None)
    parser.add_argument("-m", "--model_path", help = "File path to load models json file.", required=True, type=pathlib.Path)
    parser.add_argument("--device", help = "Which device model will use(default='cpu').", choices=['cpu', 'cuda'], type=str, default="cpu")
    parser.add_argument("--dim", help = "Image Dimension. (default: 128).", default=128, type=int)
    parser.add_argument("-n", "--num_images", help = "Number of images to generate(default=1).", default=1, type=int)
    parser.add_argument("-d", "--dest_path", help = "File path to save images generated (Default: out folder).", type=pathlib.Path)
    parser.add_argument("--diff_alg", help = "Diffusion Algorithm to use (default: ddpm).", default="ddpm", choices=[diff_alg.name.lower() for diff_alg in DiffusionAlg])
    parser.add_argument("--noise_alg", help = "Noise Degradation scheduler to use (default: linear).", default="linear", choices=[noise_scheduler.name.lower() for noise_scheduler in NoiseScheduler])
    parser.add_argument("--ddim_step_size", help = "Number of steps to skip when using ddim.", default=10, type=int)
    parser.add_argument("-T", "--max_T", help = "Max T value for noise scheduling(In cases of Ensemble methods).", default=1_000, type=int)
    parser.add_argument("-l", "--labels", nargs="*", help="Labels Index.", type=float, default=None)

    args = vars(parser.parse_args())

    # Check if json containing model details exists.
    assert os.path.isfile(args["model_path"])

    # Loads model details from json file.
    with open(args["model_path"]) as f:
        models_details = json.load(f)

    # Seed value for generating images.
    if args["seed"] is not None:
        torch.manual_seed(args["seed"])

    # Checks if number of images to generate is valid.
    assert args["num_images"] > 0

    # Path to store generated images, defaults to same directory.
    if args["dest_path"] is None:
        out_dir = "./"
    else:
        assert args["dest_path"].exists()
        out_dir = str(args["dest_path"])

    # X_T ~ N(0, I)
    img_shape = (args["num_images"], 3, args["dim"], args["dim"])
    x_t = torch.randn(img_shape, device=args["device"])

    for model_dict in models_details["models"]:
        # TODO: Allow for multiple conditionals input.
        if model_dict["cond_dim"] is not None:
            assert args["labels"] is not None
            assert len(args["labels"]) == model_dict["cond_dim"]

            labels_tensor = torch.tensor(args["labels"]).float().to(args["device"])
        else:
            labels_tensor = None

        # x_t(X_0, eps) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.
        if args["noise_alg"] == NoiseScheduler.LINEAR.name.lower():
            noise_degradation = NoiseDegradation(
                model_dict["beta_1"],
                model_dict["beta_T"],
                args["max_T"],
                args["device"])
        elif args["noise_alg"] == NoiseScheduler.COSINE.name.lower():
            assert args["ddim_step_size"] > 0
            assert args["ddim_step_size"] <=  args["max_T"]
            noise_degradation = CosineNoiseDegradation(args["max_T"])

        # Diffusion Model.
        diffusion_net = U_Net(
            in_channel=model_dict["in_channel"],
            out_channel=model_dict["out_channel"],
            num_layers=model_dict["num_layers"],
            num_resnet_blocks=model_dict["num_resnet_block"],
            attn_layers=model_dict["attn_layers"],
            num_heads=model_dict["attn_heads"],
            dim_per_head=model_dict["attn_dim_per_head"],
            time_dim=model_dict["time_dim"],
            cond_dim=model_dict["cond_dim"],
            min_channel=model_dict["min_channel"],
            max_channel=model_dict["max_channel"],
            image_recon=model_dict["image_recon"],
        ).to(args["device"])

        # Load Diffusion Checkpoints.
        assert os.path.isfile(model_dict["model_path"])
        load_status, diffusion_checkpoint = load_checkpoint(model_dict["model_path"])
        assert load_status

        diffusion_net.load_state_dict(diffusion_checkpoint["model"])

        # Evaluate model.
        diffusion_net.eval()

        with torch.no_grad():
            if args["diff_alg"] == DiffusionAlg.DDPM.name.lower():
                for noise_step in range(model_dict["max_noise"], model_dict["min_noise"] - 1, -1):
                    # t: Time Step
                    t = torch.tensor([noise_step], device=args["device"])

                    # Variables needed in computing x_(t-1).
                    beta_t, alpha_t, alpha_bar_t = noise_degradation.get_timestep_params(t)
                    
                    # eps_param(x_t, t).
                    noise_approx = diffusion_net(
                        x_t,
                        t,
                        labels_tensor)

                    # z ~ N(0, I) if t > 1, else z = 0.
                    if noise_step > 1:
                        z = torch.randn(img_shape, device=args["device"])
                    else:
                        z = 0
                    
                    # sigma_t ^ 2 = beta_t = beta_hat = (1 - alpha_bar_(t-1)) / (1 - alpha_bar_t) * beta_t
                    sigma_t = beta_t ** 0.5

                    # x_(t-1) = (1 / sqrt(alpha_t)) * (x_t - (1 - alpha_t / sqrt(1 - alpha_bar_t)) * eps_param(x_t, t)) + sigma_t * z
                    scale_1 = 1 / (alpha_t ** 0.5)
                    scale_2 = (1 - alpha_t) / ((1 - alpha_bar_t)**0.5)
                    
                    # x_(t-1).
                    x_less_degraded = scale_1 * (x_t - (scale_2 * noise_approx)) + (sigma_t * z)
                    
                    x_t = x_less_degraded

                    printProgressBar(
                        model_dict["max_noise"] - noise_step,
                        model_dict["max_noise"] - model_dict["min_noise"],
                        prefix = 'Iterations:',
                        suffix = 'Complete',
                        length = 50)
                
            elif args["diff_alg"] == DiffusionAlg.DDIM.name.lower():
                steps = list(range(model_dict["max_noise"], model_dict["min_noise"] - 1, -args["ddim_step_size"]))

                if not model_dict["min_noise"] in steps:
                    steps = steps + [model_dict["min_noise"]]
                        
                # 0 - Deterministic
                # 1 - DDPM
                eta = 0.0

                for count in range(len(steps)):
                    # t: Time Step
                    t = torch.tensor([steps[count]], device=args["device"])

                    # eps_theta(x_t, t).
                    noise_approx = diffusion_net(
                        x_t,
                        t,
                        labels_tensor)

                    # Variables needed in computing x_t.
                    _, _, alpha_bar_t = noise_degradation.get_timestep_params(t)
                    
                    # Approximates x0 using x_t and eps_theta(x_t, t).
                    # x_t - sqrt(1 - alpha_bar_t) * eps_theta(x_t, t) / sqrt(alpha_bar_t).
                    scale = 1 / alpha_bar_t**0.5
                    x0_approx = scale * (x_t - ((1 - alpha_bar_t)**0.5 * noise_approx))

                    if count < len(steps) - 1:
                        tm1 = torch.tensor([steps[count + 1]], device=args["device"])

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
                        x_t = (alpha_bar_tm1**0.5 * x0_approx) + ((1 - alpha_bar_tm1 - sigma**2)**0.5 * noise_approx) + (sigma * eps)

                    printProgressBar(
                        model_dict["max_noise"] - steps[count],
                        model_dict["max_noise"] - model_dict["min_noise"],
                        prefix = 'Iterations:',
                        suffix = 'Complete',
                        length = 50)

    # now = datetime.now()
    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    plot_sampled_images(
        sampled_imgs=x_t,  # x_0
        file_name=now,
        dest_path=out_dir)

    print(f"\nSaving generated image: %s" % os.path.join(out_dir, now))

if __name__ == "__main__":
    main()
