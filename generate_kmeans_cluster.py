import os
import csv
import glob
import logging

import torch
import torchvision
import torch.nn.functional as F

from utils.utils import *
from utils.kmeans_utils import *
from custom_dataset.img_dataset import ImageDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    """
    Creates dataset using index of nearest k-means centroid.
    """
    # Training Params.
    dataset_path = None
    if dataset_path is None or not os.path.isdir(dataset_path):
        raise ValueError("Invalid/No dataset_path entered.")

    out_csv_file = None
    if out_csv_file is None or not os.path.isfile(out_csv_file):
        raise ValueError("Invalid/No out path entered.")

    # Batch size of images computed at once.
    batch_size = 128

    # Centroid checkpoints.
    centroids_checkpoint = None

    # Load Centroids.
    centroids_status, centroids_dict= load_checkpoint(centroids_checkpoint)
    if not centroids_status:
        raise Exception("An error occured while loading centroid checkpoint.")
    centroids = centroids_dict["centroids"]
    centroids = centroids.to(device)
    
    # List of image dataset.
    img_regex = os.path.join(dataset_path, "*.jpg")
    img_list = glob.glob(img_regex)

    if len(img_list) == 0:
        raise Exception("No dataset found!")

    # Dataset and DataLoader.
    dataset = ImageDataset(
        img_paths=img_list,
        return_filepaths=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True)
    
    total_count = len(dataloader)

    with open(out_csv_file, "w+") as out_f:
        writer = csv.writer(out_f)
        
        for idx, (tr_data, f_path) in enumerate(dataloader):
            tr_data = tr_data.unsqueeze(0).to(device)  # (1, N, C, H, W)

            distance = compute_rmse(centroids, tr_data)  # [:, :, None, None, None]
            min_distance_index = torch.argmin(distance, dim=0).squeeze()
            min_distance_index = min_distance_index.tolist()

            for f_path_, index_data in zip(f_path, min_distance_index):
                writer.writerow([f_path_, index_data])

            printProgressBar(
                idx,
                total_count,
                prefix = "K-Means Iterations:",
                suffix = "Complete",
                length = 50)

if __name__ == "__main__":
    main()
