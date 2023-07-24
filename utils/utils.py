import os

import torch
import torchvision


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def plot_sampled_images(sampled_imgs, file_name, dest_path=None):
    # Convert from BGR to RGB,
    permute = [2, 1, 0]
    sampled_imgs = sampled_imgs[:, permute]


    grid_img = torchvision.utils.make_grid(
        sampled_imgs,
        nrow=5,
        normalize=True,
        value_range=(-1, 1))

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
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            return True, checkpoint
        except Exception as e:
            return False, None
    else:
        print(f"Checkpoint does not exist.")
        return False, None
