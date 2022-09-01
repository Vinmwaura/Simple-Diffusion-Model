import os

import torch
import torchvision


def plot_sampled_images(sampled_imgs, file_name, dest_path=None):
    grid_img = torchvision.utils.make_grid(sampled_imgs, nrow=5)

    if dest_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dir_path = os.path.join(current_dir, "plots")
    else:
        dir_path = os.path.join(dest_path, "plots")
    
    os.makedirs(dir_path, exist_ok=True)
    try:
        torchvision.utils.save_image(
            grid_img,
            os.path.join(dir_path, str(file_name) + ".jpg"))
    except Exception as e:
        print(f"An error occured while plotting reconstructed image: {e}")


def save_model(model_net, file_name, dest_path, checkpoint=False, steps=0):
    try:
        if checkpoint:
            f_path = os.path.join(dest_path, "checkpoint")
        else:
            f_path = os.path.join(dest_path, "models")
        
        os.makedirs(f_path, exist_ok=True)

        model_name = f"{file_name}_{str(steps)}.pt"
        torch.save(
            model_net,
            os.path.join(f_path, model_name))
        return True
    except Exception as e:
        print(f"Exception occured while saving model: {e}.")
        return False

def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path)
            return True, checkpoint
        except Exception as e:
            return False, None
    else:
        print(f"Checkpoint does not exist.")
        return False, None
