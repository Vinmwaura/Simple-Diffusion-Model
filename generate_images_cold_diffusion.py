import os
import uuid
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
    plot_sampled_images)
from diffusion_sampling_algorithms import cold_diffusion_sampling


def generate_images_cold_diffusion(raw_args=None, log=print, save_locally=True):
    parser = argparse.ArgumentParser(
        description="Generate Images using Cold Diffusion models.")
    
    # Sampling Params.
    parser.add_argument(
        "--device",
        help="Which hardware device model will run on (default='cpu').",
        choices=['cpu', 'cuda'],
        type=str,
        default="cpu")
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
        "-n",
        "--num_images",
        help="Number of images to generate(default=1).",
        default=1,
        type=int)
    parser.add_argument(
        "-d",
        "--dest_path",
        help="File path to save images generated (Default: ./plots).",
        type=pathlib.Path)
    parser.add_argument(
        "--cold_step_size",
        help="Number of steps to skip when using cold diffusion.",
        default=10,
        type=int)
    parser.add_argument(
        "-l",
        "--labels",
        nargs="*",
        help="Conditional Labels.",
        type=float,
        default=None)

    args = vars(parser.parse_args(raw_args))

    # Check if json containing model details exists.
    if not os.path.isfile(args["model_path"]):
        raise FileNotFoundError("Invalid path for json file containing models, kindly correct and try again!")

    # Loads model details from json file.
    with open(args["model_path"]) as f:
        models_details = json.load(f)

    if len(models_details["models"]) == 0:
        raise ValueError("Invalid/no model details in json, kindly correct and try again!")

    # Seed value for generating images.
    if args["seed"] is not None:
        torch.manual_seed(args["seed"])

    # Checks if number of images to generate is valid.
    if args["num_images"] <= 0:
        raise ValueError("Invalid image numbers, should be greater than 0!")

    # Path to store generated images, defaults to same directory.
    if args["dest_path"] is None:
        out_dir = "./"
    else:
        if not args["dest_path"].exists():
            raise ValueError("Invalid destination path!")
        out_dir = str(args["dest_path"])
    
    # Check if valid Sampling size.
    if args["cold_step_size"] < 0 or args["cold_step_size"] >  args["max_T"]:
        raise ValueError("Invalid step size for Cold Diffusion!")

    noise = None
    for model_dict in models_details["models"]:
        # x_t(X_0, eps) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.
        if model_dict["noise_alg"] == NoiseScheduler.LINEAR.name.lower():
            noise_degradation = NoiseDegradation(
                model_dict["beta_1"],
                model_dict["beta_T"],
                args["max_T"],
                args["device"])
        elif model_dict["noise_alg"] == NoiseScheduler.COSINE.name.lower():
            noise_degradation = CosineNoiseDegradation(args["max_T"])

        if noise is None:
            # X_T ~ N(0, I).
            img_C = model_dict["img_C"]
            img_H = model_dict["img_H"]
            img_W = model_dict["img_W"]
            img_num = args["num_images"]
            img_shape = (img_num, img_C, img_H, img_W)

            noise = torch.randn(img_shape, device=args["device"])
            x_t = 1 * noise
        else:
            # For Ensemble models.
            x_t = noise_degradation(
                img=x0_approx,
                steps=torch.tensor([model_dict["max_noise"]], device=args["device"]),
                eps=noise)

        if model_dict["cond_dim"] is not None:
            if args["labels"] is None or len(args["labels"]) != model_dict["cond_dim"]:
                raise ValueError("Invalid/No conditional labels passed!")
            labels_tensor = torch.tensor(args["labels"]).float().to(args["device"])
        else:
            labels_tensor = None
        
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
        if not os.path.isfile(model_dict["model_path"]):
            raise FileNotFoundError("Invalid path for model in json file, kindly correct and try again!")
        load_status, diffusion_checkpoint = load_checkpoint(model_dict["model_path"])
        if not load_status:
            raise Exception("Failed to load model!")

        diffusion_net.load_state_dict(diffusion_checkpoint["model"])

        x0_approx = cold_diffusion_sampling(
            diffusion_net=diffusion_net,
            noise_degradation=noise_degradation,
            x_t=x_t,
            noise=noise,
            min_noise=model_dict["min_noise"],
            max_noise=model_dict["max_noise"],
            cond_img=None,
            labels_tensor=labels_tensor,
            skip_step_size=args["cold_step_size"],
            device=args["device"],
            log=log)
        
    if save_locally:
        datetime_now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        unique_id = uuid.uuid4().hex
        unique_name = datetime_now + f"({img_H},{img_W})" + "_" + unique_id
        plot_sampled_images(
            sampled_imgs=x0_approx,  # x_0
            file_name=unique_name,
            dest_path=out_dir)

        return None
    else:
        # Returns generated image.
        return x0_approx
    
if __name__ == "__main__":
    generate_images_cold_diffusion()
