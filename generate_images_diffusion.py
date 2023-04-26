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
from diffusion_enums import *
from utils import load_checkpoint, plot_sampled_images, printProgressBar


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Images using diffusion models.")

    # Trainning Params.
    parser.add_argument("--device", help = "Which device model will use(default='cpu').", choices=['cpu', 'cuda'], type=str, default="cpu")
    parser.add_argument("--dim", help = "Image Dimension. (default: 128).", default=128, type=int)
    parser.add_argument("-n", help = "Number of images to generate(default=1).", default=1, type=int)
    
    # Model Params.
    parser.add_argument("-m", "--model", help = "File path of diffusion model to be used.", required=True, type=pathlib.Path)
    parser.add_argument("--in_channel", help = "U Net Model In channel config. (default: 3).", default=3, type=int)
    parser.add_argument("--out_channel", help = "U Net Model Out channel config. (default: 3).", default=3, type=int)
    parser.add_argument("--num_layers", help = "U Net Model Layers used. (default: 5).", default=5, type=int)
    parser.add_argument("--time_dim", help = "U Net time channels used. (default: 64).", default=64, type=int)
    parser.add_argument("--min_channel", help = "U Net min channels used. (default: 128).", default=128, type=int)
    parser.add_argument("--max_channel", help = "U Net max channels used. (default: 512).", default=512, type=int)
    parser.add_argument("--image_recon", help = "U Net model uses tanh in last layer. (default: False).", default=False, type=bool)
    parser.add_argument("--attn_layers", help = "U Net layers using attention. (default: 2, 3, 4).", nargs='*', default=[2, 3, 4], type=int)
    parser.add_argument("-d", "--dest_path", help = "File path to save images generated (Default: out folder).", type=pathlib.Path)
    parser.add_argument("--diff_alg", help = "Diffusion Algorithm to use (default: ddpm).", default="ddpm", choices=[diff_alg.name.lower() for diff_alg in DiffusionAlg])
    parser.add_argument("--noise_alg", help = "Noise Degradation scheduler to use (default: linear).", default="linear", choices=[noise_scheduler.name.lower() for noise_scheduler in NoiseScheduler])
    parser.add_argument("-b1", "--beta_1", help = "Beta 1 value.", default=5e-3, type=float)
    parser.add_argument("-bT", "--beta_T", help = "Beta T value.", default=9e-3, type=float)
    parser.add_argument("-T", "--max_T", help = "Max T value to start from.", default=1000, type=int)
    parser.add_argument("--ddim_step_size", help = "Number of steps to skip when using ddim.", default=10, type=int)

    args = vars(parser.parse_args())
    try:
        assert args["n"] > 0
        img_shape = (args["n"], 3, args["dim"], args["dim"])

        if args["dest_path"] is None:
            out_dir = "./"
        else:
            assert args["dest_path"].exists()
            out_dir = str(args["dest_path"])

        # x_t(X_0, eps) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.
        if args["noise_alg"] == NoiseScheduler.LINEAR.name.lower():
            noise_degradation = NoiseDegradation(
                args["beta_1"],
                args["beta_T"],
                args["max_T"],
                args["device"])
        elif args["noise_alg"] == NoiseScheduler.COSINE.name.lower():
            assert args["ddim_step_size"] > 0
            assert args["ddim_step_size"] <=  args["max_T"]
            noise_degradation = CosineNoiseDegradation(args["max_T"])

        # Diffusion Model.
        diffusion_net = U_Net(
            in_channel=args["in_channel"],
            out_channel=args["out_channel"],
            num_layers=args["num_layers"],
            attn_layers=args["attn_layers"],
            time_dim=args["time_dim"],
            min_channel=args["min_channel"],
            max_channel=args["max_channel"],
            image_recon=args["image_recon"],
        ).to(args["device"])

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
            if args["diff_alg"] == DiffusionAlg.DDPM.name.lower():
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
                
                picture_name = str(uuid.uuid4())
                plot_sampled_images(
                    sampled_imgs=x_less_degraded,  # x_0
                    file_name=picture_name,
                    dest_path=out_dir)
            
            elif args["diff_alg"] == DiffusionAlg.DDIM.name.lower():
                steps = list(range(args["max_T"], 0, -args["ddim_step_size"])) + [1]
                        
                # 0 - Deterministic
                # 1 - DDPM
                eta = 0.0

                for count in range(len(steps)):
                    # t: Time Step
                    t = torch.tensor([steps[count]], device=args["device"])

                    # eps_theta(x_t, t).
                    noise_approx = diffusion_net(
                        x_t,
                        t)

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
                        args["max_T"] - steps[count],
                        args["max_T"],
                        prefix = 'Iterations:',
                        suffix = 'Complete',
                        length = 50)

                x_0 = x_t
                
                picture_name = str(uuid.uuid4())
                plot_sampled_images(
                    sampled_imgs=x_0,  # x_0
                    file_name=picture_name,
                    dest_path=out_dir)

            print(f"\nSaving generated image: %s" % os.path.join(out_dir, picture_name))

    except Exception as e:
        print(f"An error occured while generating images: {e}")

