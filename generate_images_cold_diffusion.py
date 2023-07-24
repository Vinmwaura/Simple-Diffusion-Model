import os
import json
import pathlib
import argparse
from datetime import datetime

import torch
import torchvision

# U Net Model.
from models.U_Net import U_Net

# Degradation Operators.
from degraders import *
from diffusion_enums import *

from utils.utils import (
    load_checkpoint,
    plot_sampled_images,
    printProgressBar)

def main():
    parser = argparse.ArgumentParser(
        description="Generate Images using Cold Diffusion models.")
    
    # Sampling Params.
    parser.add_argument(
        "--seed",
        help="Seed value for generating image(default: None).",
        type=int,
        default=None)
    parser.add_argument(
        "-m",
        "--model_path",
        help="File path to load models json file.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "-T",
        "--max_T",
        help="Max T value for noise scheduling(In cases of Ensemble methods).",
        default=1_000,
        type=int)
    parser.add_argument(
        "--device",
        help="Which hardware device model will run on (default='cpu').",
        choices=['cpu', 'cuda'],
        type=str,
        default="cpu")
    parser.add_argument(
        "-n",
        "--num_images",
        help="Number of images to generate(default=1).",
        default=1,
        type=int)
    parser.add_argument(
        "-d",
        "--dest_path",
        help="File path to save images generated (Default: out folder).",
        type=pathlib.Path)
    parser.add_argument(
        "--noise_alg",
        help="Noise Degradation scheduler to use (default: linear).",
        default="linear",
        choices=[noise_scheduler.name.lower() for noise_scheduler in NoiseScheduler])
    parser.add_argument(
        "--cold_step_size",
        help="Number of steps to skip when using cold diffusion.",
        default=10,
        type=int)
    parser.add_argument(
        "-l",
        "--labels",
        nargs="*",
        help="Labels Index.",
        type=float,
        default=None)

    args = vars(parser.parse_args())

    # Check if json containing model details exists.
    assert os.path.isfile(args["model_path"])

    # Loads model details from json file.
    with open(args["model_path"]) as f:
        models_details = json.load(f)
    assert len(models_details["models"]) >= 1

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

    noise = None
    for model_dict in models_details["models"]:
        if noise is None:
            # X_T ~ N(0, I).
            img_C = model_dict["img_C"]
            img_H = model_dict["img_H"]
            img_W = model_dict["img_W"]
            img_num = args["num_images"]
            img_shape = (img_num, img_C, img_H, img_W)

            noise = torch.randn(img_shape, device=args["device"])
            x_t = 1 * noise

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
            assert args["cold_step_size"] > 0
            assert args["cold_step_size"] <=  args["max_T"]
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
            steps = list(range(model_dict["max_noise"], model_dict["min_noise"] - 1, -args["cold_step_size"]))

            # Includes timestep 1 into the steps if not included.
            if 1 not in steps:
                steps = steps + [1]

            for count in range(len(steps)):
                # t: Time Step.
                t = torch.tensor([steps[count]], device=args["device"])

                # Reconstruction: (x0_hat).
                # x_t_combined = torch.cat((x_t_plot, eps_diffusion_approx), dim=1)
                x0_approx = diffusion_net(
                    x_t,
                    t,
                    labels_tensor)

                if count < len(steps) - 1:
                    # t-1: Time Step
                    tm1 = torch.tensor([steps[count + 1]], device=args["device"])

                    # D(x0_hat, t).
                    # Noise degraded image (x_t).
                    # x_t(X_0, eps) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.
                    x_t_hat_plot = noise_degradation(
                        img=x0_approx,
                        steps=t,
                        eps=noise)

                    # D(x0_hat, t-1).
                    # Noise degraded image (x_t).
                    # x_t(X_0, eps) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.
                    x_tm1_hat_plot = noise_degradation(
                        img=x0_approx,
                        steps=tm1,
                        eps=noise)
                    
                    # q(x_t-1 | x_t, x_0).
                    # Improved sampling from Cold Diffusion paper.
                    x_t = x_t - x_t_hat_plot + x_tm1_hat_plot

                    printProgressBar(
                        model_dict["max_noise"] - steps[count],
                        model_dict["max_noise"] - (model_dict["min_noise"] - 1),
                        prefix = 'Iterations:',
                        suffix = 'Complete',
                        length = 50)

    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    filename = now + f"(H: {img_H}, W: {img_W})"
    plot_sampled_images(
        sampled_imgs=x0_approx,  # x_0
        file_name=filename,
        dest_path=out_dir)

    print(f"Saving generated image: %s" % os.path.join(out_dir, now))

if __name__ == "__main__":
    main()
