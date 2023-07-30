import random

import cv2
from tinydb import TinyDB

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


"""
Custom Image Loader using Opencv2 + TinyDB(NoSQL) to load images and their conditional labels.
"""
class ConditionalImgDataset(Dataset):
    def __init__(
            self,
            dataset_path=None):
        dataset_db = TinyDB(dataset_path)
        data_tbl = dataset_db.table("Data")
        if len(data_tbl) <= 0:
            raise Exception("No data found in Data table.")
        labels_tbl = dataset_db.table("Labels")
        if len(labels_tbl) <= 0:
            raise Exception("No data found in Labels table.")

        self.all_labels = labels_tbl.all()[0]["labels"]
        self.all_data = data_tbl.all()
        random.shuffle(self.all_data)  # Initial shuffle in case dataset is sorted.

        self.dataset = []
        for data in self.all_data:
            out_labels = []
            for label in self.all_labels:
                out_labels.append(float(data[label]))
            self.dataset.append((data["filename"], out_labels))

    def get_labels(self):
        return self.all_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if len(self.dataset) <= 0:
            raise Exception("No data found in dataset.")

        img_path, img_label = self.dataset[index]
        
        # Load images using opencv2.
        img = cv2.imread(img_path)
        
        # Convert labels to Tensor.
        labels_tensor = torch.Tensor(img_label)

        # Scale images to be between 1 and -1.
        img = (img.astype(float) - 127.5) / 127.5

        # Convert image as numpy to Tensor.
        img_tensor = torch.from_numpy(img).float()

        # Permute image to be of format: [C,H,W]
        img_tensor = img_tensor.permute(2, 0, 1)

        return img_tensor, labels_tensor
