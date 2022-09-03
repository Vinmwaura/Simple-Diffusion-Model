import cv2
import torch

from torch.utils.data import Dataset


"""
Custom Image Loader using Opencv2.
"""
class ImageDataset(Dataset):
    def __init__(self, img_paths=[]):
        # List of file paths for images.
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        assert len(self.img_paths) > 0

        img_path = self.img_paths[index]

        # Load images using opencv2.
        img = cv2.imread(img_path)
        
        # Scale images to be between 1 and -1.
        img = (img.astype(float) - 127.5) / 127.5

        # Convert image as numpy to Tensor.
        img_tensor = torch.from_numpy(img).float()
        
        # Permute image to be of format: [C,H,W]
        img_tensor = img_tensor.permute(2, 0, 1)
        return img_tensor
