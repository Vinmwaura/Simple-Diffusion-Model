import os
import json


import cv2

import torch
from torch.utils.data import Dataset


from tinydb import TinyDB


"""
Custom Image Loader using Opencv2 + Multi Output Classification.
"""
class ClassifierImgDataset(Dataset):
    def __init__(self, dataset_path=None):
        # Path to images + Labels.
        self.dataset_path = dataset_path

        dataset_db = TinyDB(self.dataset_path)

        data_tbl = dataset_db.table("Data")
        assert len(data_tbl) > 0

        labels_tbl = dataset_db.table("Labels")
        assert len(labels_tbl) > 0

        all_labels = labels_tbl.all()[0]["labels"]
        
        self.dataset = []
        for data in data_tbl.all():
            out_labels = []
            for label in all_labels:
                out_labels.append(float(data[label]))
            self.dataset.append((data["filename"], out_labels))

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        assert len(self.dataset) > 0

        img_path, img_label = self.dataset[index]

        # Convert labels to Tensor.
        labels_tensor = torch.Tensor(img_label)

        # Load images using opencv2.
        img = cv2.imread(img_path)
              
        # Scale images to be between 1 and -1.
        img = (img.astype(float) - 127.5) / 127.5

        # Convert image as numpy to Tensor.
        img_tensor = torch.from_numpy(img).float()
        
        # Permute image to be of format: [C,H,W]
        img_tensor = img_tensor.permute(2, 0, 1)
        return img_tensor, labels_tensor

