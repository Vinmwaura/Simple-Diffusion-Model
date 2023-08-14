import os
import json
import shutil

import click

def export_models():
    # Config File Details.
    config_name = click.prompt(
        "Config Name (Will be reflected in model names)?",
        type=str)

    # Destination Path.
    export_dest_path = click.prompt(
        "Destination path for model and config file?",
        type=click.Path(exists=True))
    
    # Creates folder to store model and config.
    new_dest_path = os.path.join(export_dest_path, config_name)
    try:
        os.makedirs(new_dest_path)
    except Exception as e:
        raise e
    
    # Image Dimension.
    img_C = click.prompt(
        "Model was trained on images with channel(C)?",
        type=click.IntRange(min=1),
        default=3)
    img_H = click.prompt(
        "Model was trained on images with Height (H)?",
        type=click.IntRange(min=2),
        default=128)
    img_W = click.prompt(
        "Model was trained on images with Width (W)?",
        type=click.IntRange(min=2),
        default=128)
    
    # Model Type.
    model_type = click.prompt(
        "Model type?",
        type=click.Choice(["BASE", "BASE-COLD", "SR"], case_sensitive=False),
        default="BASE")
    models_num = click.prompt(
        "How many models do you want to combine (For ensemble diffusion)?",
        type=click.IntRange(min=1),
        default=1)

    json_vals = {
        "models": []
    }
    
    for model_index in range(models_num):
        click.echo(f"Model: {model_index + 1} / {models_num}")

        config_path = click.prompt(
            "File path to config file?",
            type=click.Path(exists=True))
        model_path = click.prompt(
            "File path to model checkpoint?",
            type=click.Path(exists=True))
        
        with open(config_path, "r") as json_file:
            json_data = json_file.read()
        
        config_dict = json.loads(json_data)

        min_step = config_dict['min_noise_step']
        max_step = config_dict['max_noise_step']
        model_name = f"{config_name}_{min_step}-{max_step}.pt"

        temp_dict = {
            "model_name": model_name,
            "img_C": img_C,
            "img_H": img_H,
            "img_W": img_W,
            "in_channel": config_dict["in_channel"],
            "out_channel": config_dict["out_channel"],
            "num_layers": config_dict["num_layers"],
            "num_resnet_block": config_dict["num_resnet_block"],
            "attn_layers": config_dict["attn_layers"],
            "attn_heads": config_dict["attn_heads"],
            "attn_dim_per_head": config_dict["attn_dim_per_head"],
            "time_dim": config_dict["time_dim"],
            "cond_dim": config_dict["cond_dim"],
            "min_channel": config_dict["min_channel"],
            "max_channel": config_dict["max_channel"],
            "image_recon": config_dict["img_recon"],
            "max_noise": max_step,
            "min_noise": min_step,
            "noise_scheduler": config_dict["noise_scheduler"],
        }

        # Specific param values based on model.
        if model_type == "BASE":
            temp_dict["beta_1"] = config_dict["beta1"]
            temp_dict["beta_T"] = config_dict["betaT"]
        elif model_type == "SR":
            temp_dict["cond_t"] = config_dict["cond_t"]

        json_vals["models"].append(temp_dict)

        dest_path = os.path.join(new_dest_path, model_name)
        try:
            shutil.copy(model_path, dest_path)
            click.echo(f"Successfully copied model file to {dest_path}.")
        except Exception as e:
            raise e

    json_file = os.path.join(new_dest_path, "config.json")
    try:
        with open(json_file, "w") as f:
            json.dump(json_vals, f)
        click.echo(f"Successfully saved {json_file}")
    except Exception as e:
        raise e

if __name__ == "__main__":
    export_models()
