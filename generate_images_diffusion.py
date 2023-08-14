import os
import uuid
import json
import imghdr
import pathlib
import argparse
from datetime import datetime

import cv2
import numpy as np

import torch
import torch.nn.functional as F

import torchvision

from models.U_Net import U_Net

from degraders import *
from diffusion_enums import *

from utils.utils import (
    load_checkpoint,
    plot_sampled_images)
from diffusion_sampling_algorithms import (
    ddpm_sampling,
    ddim_sampling)

# Used in checking if conditional image is valid.
SUPPORTED_IMG_FORMATS = [
    "jpeg",
    "jpg",
    "png"]

def generate_images_diffusion(
        raw_args=None,
        log=print,
        cond_img=None,
        save_locally=True):

    parser = argparse.ArgumentParser(
        description="Generate Images using Diffusion models.")
    
    # Sampling Params.
    parser.add_argument(
        "--device",
        help="Which hardware device model will run on (default='cpu').",
        choices=['cpu', 'cuda'],
        type=str,
        default="cpu")
    parser.add_argument(
        "-c",
        "--config",
        help="File path to config file.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "-s",
        "--seed",
        help="Seed value for generating image(default: None).",
        type=int,
        default=None)
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
        "--diff_alg",
        help="Diffusion Sampling Algorithm to use (default: ddpm).",
        default="ddpm",
        choices=[diff_alg.name.lower() for diff_alg in DiffusionAlg])
    parser.add_argument(
        "--ddim_step_size",
        help="Number of steps to skip when using ddim.",
        default=10,
        type=int)
    parser.add_argument(
        "-T",
        "--max_T",
        help="Max T value for noise scheduling (In cases of Ensemble methods).",
        default=1_000,
        type=int)
    parser.add_argument(
        "--cond_img_path",
        help="File path to conditional image e.g Doodle image.",
        type=pathlib.Path,
        default=None)
    parser.add_argument(
        "-l",
        "--labels",
        nargs="*",
        help="Conditional Labels.",
        type=float,
        default=None)

    args = vars(parser.parse_args(raw_args))

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
            raise ValueError(
                "Invalid destination path, kindly correct and ensure it exists!")
        out_dir = str(args["dest_path"])
    
    # Check if valid DDIM sampling size where DDIM sampling is used.
    if args["diff_alg"] == DiffusionAlg.DDIM.name.lower():
        if args["ddim_step_size"] < 0 or args["ddim_step_size"] >  args["max_T"]:
            raise ValueError("Invalid step size for DDIM!")

    cond_img_path = args["cond_img_path"]
    if cond_img_path is not None:
        # Check if conditional img file exists.
        if not os.path.isfile(cond_img_path):
            raise FileNotFoundError(
                "Invalid path for conditional image, kindly correct and try again!")
        if not imghdr.what(cond_img_path) in SUPPORTED_IMG_FORMATS:
            raise ValueError("Image format is not supported!")
        
        cond_img = cv2.imread(str(cond_img_path))
    
    if cond_img is not None:
        if not isinstance(cond_img, np.ndarray):
            raise ValueError("Unsupported conditional image.")

        # Scale images to be between 1 and -1.
        cond_img = (cond_img.astype(float) - 127.5) / 127.5

        # Convert image as numpy to Tensor.
        cond_img = torch.from_numpy(cond_img).float()
        
        # Permute image to be of format: [C,H,W]
        cond_img = cond_img.permute(2, 0, 1)
        cond_img = cond_img.unsqueeze(0).repeat(args["num_images"], 1, 1, 1)

    # Loads model details from json file.
    with open(args["config"], "r") as f:
        models_details = json.load(f)

    if not "models" in models_details or len(models_details["models"]) == 0:
        raise ValueError(
            "Invalid/no model details in json, kindly correct and try again!")
    
    config_folder_path, _ = os.path.split(args["config"])

    # Low-Resolution Diffusion (Cascaded / Ensemble Models).
    noise = None
    for model_dict in models_details["models"]:
        """
        Inititalize noise and x_T at the begining but use generated x_t as input
        for other models in cases of ensemble models.
        """
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
            if args["labels"] is None or len(args["labels"]) != model_dict["cond_dim"]:
                raise ValueError("Invalid / No conditional labels passed!")
            labels_tensor = torch.tensor(args["labels"]).float().to(args["device"])
        else:
            labels_tensor = None

        # x_t(X_0, eps) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.
        if model_dict["noise_scheduler"] == NoiseScheduler.LINEAR.name:
            noise_degradation = NoiseDegradation(
                model_dict["beta_1"],
                model_dict["beta_T"],
                args["max_T"],
                args["device"])
        elif model_dict["noise_scheduler"] == NoiseScheduler.COSINE.name:
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

        # Model path, assumed to be in same directory as config.
        model_path = os.path.join(
            config_folder_path,
            model_dict["model_name"]
        )

        # Load Diffusion Checkpoints.
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                "Invalid path for model in json file, kindly correct and try again!")
        load_status, diffusion_checkpoint = load_checkpoint(model_path)
        if not load_status:
            raise Exception("Failed to load model!")

        diffusion_net.load_state_dict(diffusion_checkpoint["model"])

        sampling_alg = args["diff_alg"].lower()
        if sampling_alg == DiffusionAlg.DDPM.name.lower():
            x_t = ddpm_sampling(
                diffusion_net=diffusion_net,
                noise_degradation=noise_degradation,
                x_t=x_t,
                min_noise=model_dict["min_noise"],
                max_noise=model_dict["max_noise"],
                cond_img=cond_img,
                labels_tensor=labels_tensor,
                device=args["device"],
                log=log)
        elif sampling_alg == DiffusionAlg.DDIM.name.lower():
            x_t = ddim_sampling(
                diffusion_net=diffusion_net,
                noise_degradation=noise_degradation,
                x_t=x_t,
                min_noise=model_dict["min_noise"],
                max_noise=model_dict["max_noise"],
                cond_img=cond_img,
                labels_tensor=labels_tensor,
                ddim_step_size=args["ddim_step_size"],
                device=args["device"],
                log=log)
        else:
            raise ValueError("Invalid Diffusion Algorithm type.")

    if save_locally:
        # Save Image from Base Diffusion.
        datetime_now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        unique_id = uuid.uuid4().hex
        unique_name = datetime_now + "_" + f"({img_H},{img_W})" + "_" + unique_id

        plot_sampled_images(
            sampled_imgs=x_t,  # x_0
            file_name=unique_name,
            dest_path=out_dir,
            log=log)
        return None
    else:
        # Returns generated image.
        return x_t

if __name__ == "__main__":
    generate_images_diffusion()
