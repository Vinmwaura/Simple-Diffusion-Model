import os
import glob
import json

import click

def create_diffusion_config():
    config_name = click.prompt(
        "Name of model, will be reflected in json file name?",
        type=str)
    destination_path = click.prompt(
        "Destination path for config file?",
        type=click.Path(exists=True))

    json_file = os.path.join(
        destination_path,
        config_name + ".json")

    json_params = {}
    if click.confirm("Will the model include conditional input for training?"):
        # Dataset Path: file path to json file.
        json_params["dataset_path"] = click.prompt(
            "File path to training dataset?",
            type=click.Path(exists=True))
        
        # Model uses conditional information such as class labels in dataset.
        json_params["use_conditional"] = True

        # Dimension of conditional information such as class labels or embeddings.
        json_params["cond_dim"] = click.prompt(
            "Dimension of conditional input vector?",
            type=click.IntRange(min=1),
            default=1)
    else:
        # Dataset Path: Regex to images folder.
        json_params["dataset_path"] = click.prompt(
            "Regex to training dataset?",
            type=str)

        # Checks if dataset path is either json file path or regex to dataset images.
        regex_files = glob.glob(json_params["dataset_path"])
        if len(regex_files) == 0:
            raise TypeError("Invalid Dataset Path passed!")
        
        json_params["use_conditional"] = False
        json_params["cond_dim"] = None

    # Output directory for checkpoint.
    json_params["out_dir"] = click.prompt(
        "Destination path for output?",
        type=click.Path())

    # Training Parameters.
    json_params["checkpoint_steps"] = click.prompt(
        "Steps to be performed before checkpoint?",
        type=click.IntRange(min=1),
        default=1_000)
    json_params["lr_steps"] = click.prompt(
        "Steps before halving learning rate?",
        type=click.IntRange(min=1),
        default=100_000)
    json_params["max_epoch"] = click.prompt(
        "Total epoch for training?",
        type=click.IntRange(min=1),
        default=1_000)
    json_params["plot_img_count"] = click.prompt(
        "Number of images in sampled ploting grid?",
        type=click.IntRange(min=1),
        default=10)
    json_params["flip_imgs"] = click.prompt(
        "Randomly flip images horizontally during training (Image Augmentation)?",
        type=bool,
        default=True)

    # Load checkpoints.
    if click.confirm('Do you want to load a previous model checkpoint?'):
        json_params["model_checkpoint"] = click.prompt(
            "Model checkpoint?",
            type=click.Path(exists=True))
        json_params["load_diffusion_optim"] = click.prompt(
            "Load model's checkpoint optim values?",
            type=bool,
            default=False)
    else:
        json_params["model_checkpoint"] = None
        json_params["load_diffusion_optim"] = False

    if click.confirm('Do you want to load a previous configuration checkpoint?'):
        json_params["config_checkpoint"] = click.prompt(
            "Config chekpoint?",
            type=click.Path(exists=True))
    else:
        json_params["config_checkpoint"] = None

    # Model Params.
    json_params["diffusion_lr"] = click.prompt(
        "Learning Rate for model training?",
        type=click.FloatRange(min=0, min_open=True),
        default=2e-5)
    json_params["batch_size"] = click.prompt(
        "Batch size for training?",
        type=click.IntRange(min=1),
        default=20)

    # Diffusion Params.
    json_params["noise_scheduler"] = click.prompt(
        "Noise scheduler to use?",
        type=click.Choice(["LINEAR", "COSINE"], case_sensitive=False),
        default="LINEAR")

    if json_params["noise_scheduler"] == "LINEAR":
        # Forward process variances used in linear noise scheduling.
        json_params["beta1"] = click.prompt(
            "Beta1 for Linear Noise scheduling?",
            type=click.FloatRange(min=0, min_open=True),
            default=5e-3)
        
        # Forward process variances used in linear noise scheduling.
        json_params["betaT"] = click.prompt(
            "BetaT for Linear Noise scheduling?",
            type=click.FloatRange(min=0, min_open=True),
            default=9e-3)
    else:
        # Default Params just in case changes are made manually.
        json_params["beta1"] = 5e-3
        json_params["betaT"] = 9e-3
    
    json_params["diffusion_alg"] = click.prompt(
        "Diffusion algorithm to use?",
        type=click.Choice(["DDPM", "DDIM", "COLD"], case_sensitive=False),
        default="DDPM")
    
    if json_params["diffusion_alg"] == "DDIM" or json_params["diffusion_alg"] == "COLD":
        json_params["skip_step"] = click.prompt(
            "Number of steps to be skipped in DDIM/COLD sampling?",
            type=click.IntRange(min=1),
            default=100)
    else:
        # Placeholder value just in case json file is manually changed.
        json_params["skip_step"] = 100
    
    json_params["min_noise_step"] = click.prompt(
        "Min noise step for diffusion model?",
        type=click.IntRange(min=1),
        default=1)
    json_params["max_noise_step"] = click.prompt(
        "Max noise step for diffusion model?",
        type=click.IntRange(min=1),
        default=1_000)
    json_params["max_actual_noise_step"] = click.prompt(
        "Max actual noise step, needed for noise scheduler?",
        type=click.IntRange(min=1),
        default=1_000)

    # Model Params.
    json_params["in_channel"] = click.prompt(
        "Model In Channel?",
        type=click.IntRange(min=1),
        default=3)
    json_params["out_channel"] = click.prompt(
        "Model Out Channel?",
        type=click.IntRange(min=1),
        default=3)
    json_params["num_layers"] = click.prompt(
        "Number of layers in model?",
        type=click.IntRange(min=1),
        default=4)
    json_params["num_resnet_block"] = click.prompt(
        "Number of Residual layers in each model's layer?",
        type=click.IntRange(min=1),
        default=1)
    
    # Attention Mechanism for each layer.
    json_params["attn_layers"] = []
    for layer_num in range(json_params["num_layers"]):
        if click.confirm(f"Do you want to add attention mechanism in Layer {layer_num} / {json_params['num_layers'] - 1}?"):
            json_params["attn_layers"].append(layer_num)
    json_params["attn_heads"] = click.prompt(
        "Number of attention heads in attention layers?",
        type=click.IntRange(min=1),
        default=1)
    attn_dim_per_head_val = click.prompt(
        "Dimensions of attention head (-1 for None)?",
        type=click.IntRange(min=-1),
        default=-1)
    json_params["attn_dim_per_head"] = None if attn_dim_per_head_val == -1 else attn_dim_per_head_val
    json_params["time_dim"] = click.prompt(
        "Dimension of time conditional input?",
        type=click.IntRange(min=4),
        default=512)
    json_params["min_channel"] = click.prompt(
        "Minimum channel in model?",
        type=click.IntRange(min=4),
        default=128)
    json_params["max_channel"] = click.prompt(
        "Maximum channel in model?",
        type=click.IntRange(min=4),
        default=512)
    json_params["img_recon"] = click.prompt(
        "Reconstruct image in final layer (Use Tanh: for cold diffusion)?",
        type=bool,
        default=False)

    try:
        if click.confirm(f"File will be saved in: {json_file}, Are you sure?", default=True):
            with open(json_file, "w") as f:
                json.dump(json_params, f)
            click.echo(f"File saved at: {json_file}")
    except Exception as e:
        click.echo(f"An error occured saving json file: {e}.")

if __name__ == "__main__":
    create_diffusion_config()
