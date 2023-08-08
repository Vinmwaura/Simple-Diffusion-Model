import math
import glob
import logging

import torch
torch.manual_seed(69)

import torchvision
import torch.nn.functional as F

from utils.utils import *
from utils.kmeans_utils import *

from img_dataset import ImageDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
Creates Centroids from the image dataset to be used as labels for diffusion models.
"""
def main():
    project_name = "K-Means-Images"

    # K-Means Params.
    K = 75
    min_delta = 1e-8  # Determines stopping condition.
    batch_size = 64
    max_training_steps = 1_000
    
    # Training Params.
    dataset_path = None
    if dataset_path is None:
        raise ValueError("Invalid/No dataset_path entered.")

    out_dir = None
    if out_dir is None:
        raise ValueError("Invalid/No output path entered.")

    os.makedirs(out_dir, exist_ok=True)

    # Centroids.
    centroids = None

    # Checkpoints.
    centroids_checkpoint = None
    
    # Load Centroids.
    if centroids_checkpoint is not None:
        centroids_status, centroids_dict= load_checkpoint(centroids_checkpoint)
        if not centroids_status:
            raise Exception("An error occured while loading centroids checkpoint!")

        centroids = centroids_dict["centroids"]
        centroids = centroids.to(device)

    log_path = os.path.join(out_dir, f"{project_name}.log")
    logging.basicConfig(
        # filename=log_path,
        format="%(asctime)s %(message)s",
        encoding='utf-8',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ],
        level=logging.DEBUG)

    # List of image dataset.
    img_regex = os.path.join(dataset_path, "*.jpg")
    img_list = glob.glob(img_regex)

    if len(img_list) <= 0:
        raise Exception("An error occured loading the dataset!")

    # Dataset and DataLoader.
    dataset = ImageDataset(img_paths=img_list)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True)
    
    logging.info("#" * 100)
    logging.info(f"Train Parameters:")
    logging.info(f"Dataset Path: {dataset_path}")
    logging.info(f"Output Path: {out_dir}")
    logging.info(f"Max training steps: {max_training_steps:,}")
    logging.info("#" * 100)
    logging.info(f"K-Means Parameters:")
    logging.info(f"K: {K:,}")
    logging.info(f"Batch size: {batch_size:,}")
    logging.info("#" * 100)
    
    # Generate centroids by iterating over all the images and picking first K.
    if centroids is None:
        for tr_data in dataloader:
            if centroids is None:
                centroids = tr_data
            else:
                centroids = torch.cat((centroids, tr_data), dim=0)
            
            N, _, _, _ = centroids.shape
            if N >= K:
                break
    
    centroids = centroids[:K, :, :, :].unsqueeze(1)
    centroids = centroids.to(device)

    stopping_condition = False
    training_steps = 0
    prev_mqe = None

    while not stopping_condition:
        total_data = 0
        total_distances = 0
        total_data_counter = 0

        total_count = len(dataloader)
        for index, tr_data in enumerate(dataloader):
            tr_data = tr_data.unsqueeze(0).to(device)  # (1, N, C, H, W)

            distance = compute_rmse(centroids, tr_data)[:, :, None, None, None]
            min_distance_index = torch.argmin(distance, dim=0).squeeze()
            
            mask = F.one_hot(min_distance_index, num_classes=K).T
            mask = mask[:, :, None, None, None]

            total_data = total_data + torch.sum((mask * tr_data), dim=1, keepdim=True)
            total_data_counter = total_data_counter + torch.sum(mask, dim=1, keepdim=True)
            total_distances = total_distances + torch.sum((mask * distance), dim=1, keepdim=True)
            
            printProgressBar(
                index,
                total_count,
                prefix = "K-Means Iterations:",
                suffix = "Complete",
                length = 50)

        if (total_data_counter <= 0).all():
            raise ValueError("Value can't be 0 or negative!")

        centroids = total_data / total_data_counter
        mqe_ = total_distances / total_data_counter
        mqe_mean = torch.mean(mqe_).item()
        
        if prev_mqe is None:
            prev_mqe = mqe_mean
            diff_ = None
        else:
            diff_ = prev_mqe - mqe_mean
            if diff_ < min_delta:
                stopping_condition = True
            prev_mqe = mqe_mean
        logging.info(f"\nTraining Step: {training_steps} / {max_training_steps} | MQE_mean: {mqe_mean} | Centroid min count: {total_data_counter.min()} | Centroid max count: {total_data_counter.max()}")
        training_steps += 1
        if training_steps >= max_training_steps:
            stopping_condition = True
        
        # Plot Centroids.
        plot_sampled_images(
            sampled_imgs=centroids.squeeze(1),
            file_name=f"centroids_plot_{training_steps}",
            dest_path=out_dir)
        
        # Save Centroids.
        centroids_state = {
            "centroids": centroids.cpu(),
            "K": K}
        save_model(
            model_net=centroids_state,
            file_name="centroids",
            dest_path=out_dir,
            checkpoint=True,
            steps=training_steps)
        
    logging.info(f"\nFinished training.")

if __name__ == "__main__":
    main()
