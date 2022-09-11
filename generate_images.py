import os
import uuid
import argparse
import pathlib

import torch
import torchvision

# Degradation Operators.
from degraders import *

# U Net Model.
from models.U_Net import U_Net
from diffusion_alg import *
from utils import load_checkpoint, plot_sampled_images, printProgressBar


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Images using diffusion models.")
    parser.add_argument("--device", help = "Which device model will use(default='cpu').", choices=['cpu', 'cuda'], type=str, default="cpu")
    parser.add_argument("--dim", help = "Image dimension model was trained on. (default: 128).", default=128, type=int)
    parser.add_argument("-m", "--model", help = "File path of diffusion model to be used.", required=True, type=pathlib.Path)
    parser.add_argument("-d", "--dest_path", help = "File path to save images generated (Default: out folder).", type=pathlib.Path)
    parser.add_argument("--alg", help = "Diffusion Algorithm to use (default: ddpm).", default="ddpm", choices=[diff_alg.name.lower() for diff_alg in DiffusionAlg])
    parser.add_argument("-b1", "--beta_1", help = "Beta 1 value.", default=5e-3, type=float)
    parser.add_argument("-bT", "--beta_T", help = "Beta T value.", default=9e-3, type=float)
    parser.add_argument("-T", "--max_T", help = "Max T value to start from.", default=1000, type=int)
    parser.add_argument("-n", help = "Number of images to generate(default=1).", default=1, type=int)

    args = vars(parser.parse_args())
    
    try:
        # Images need to be greater than 64
        assert args["dim"] >= 64
        assert args["n"] > 0
        img_shape = (args["n"], 3, args["dim"], args["dim"])
        

        if args["dest_path"] is None:
            out_dir = "./"
        else:
            assert args["dest_path"].exists()
            out_dir = str(args["dest_path"])

        noise_degradation = NoiseDegradation(
            args["beta_1"],
            args["beta_T"],
            args["max_T"],
            args["device"])
        
        # Diffusion Model.
        diffusion_net = U_Net(img_dim=args["dim"]).to(args["device"])

        # Load Diffusion Checkpoints.
        assert args["model"].exists()
        assert args["model"].is_file()
        load_status, diffusion_checkpoint = load_checkpoint(str(args["model"]))
        assert load_status

        diffusion_net.load_state_dict(diffusion_checkpoint["model"])

        # Evaluate model.
        diffusion_net.eval()

        # X_T ~ N(0, I)
        x_t = torch.randn(img_shape, device=args["device"])

        with torch.no_grad():
            if args["alg"] == DiffusionAlg.DDPM.name.lower():
                
                for noise_step in range(args["max_T"], 0, -1):
                    # t: Time Step
                    t = torch.tensor([noise_step], device=args["device"])

                    # Variables needed in computing x_(t-1).
                    beta_t, alpha_t, alpha_bar_t = noise_degradation.get_timestep_params(t)
                    
                    # eps_param(x_t, t).
                    noise_approx = diffusion_net(
                        x_t,
                        t)

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
                        args["max_T"] - noise_step,
                        args["max_T"],
                        prefix = 'Iterations:',
                        suffix = 'Complete',
                        length = 50)

                plot_sampled_images(
                    sampled_imgs=x_less_degraded,  # x_0
                    file_name=str(uuid.uuid4()),
                    dest_path=out_dir)

    except Exception as e:
        print(f"An error occured while generating images: {e}")

