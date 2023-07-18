import random

import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from tinydb import TinyDB


"""
Custom Image Loader using Opencv2 + TinyDB(NoSQL) to load images and their labels.
"""
class DoodleImgDataset(Dataset):
    def __init__(
            self,
            dataset_path=None):
        dataset_db = TinyDB(dataset_path)
        data_tbl = dataset_db.table("Data")
        assert len(data_tbl) > 0

        labels_tbl = dataset_db.table("Labels")
        assert len(labels_tbl) > 0

        self.all_labels = labels_tbl.all()[0]["labels"]
        self.all_data = data_tbl.all()
        random.shuffle(self.all_data)  # Initial shuffle in case dataset is sorted.

        label = self.all_labels[0]

        self.dataset = []
        for data in self.all_data:
            self.dataset.append((data["filename"], data[label]))

    def get_labels(self):
        return self.all_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        assert len(self.dataset) > 0

        img_path, label_path = self.dataset[index]
        
        # Load images using opencv2.
        img = cv2.imread(img_path)
        label = cv2.imread(label_path)

        # Scale images to be between 1 and -1.
        img = (img.astype(float) - 127.5) / 127.5
        label = (label.astype(float) - 127.5) / 127.5

        # Convert image as numpy to Tensor.
        img_tensor = torch.from_numpy(img).float()
        label_tensor = torch.from_numpy(label).float()

        # Permute image to be of format: [C,H,W].
        img_tensor = img_tensor.permute(2, 0, 1)
        label_tensor = label_tensor.permute(2, 0, 1)

        return img_tensor, label_tensor
