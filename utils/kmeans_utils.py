import torch

# Root Mean Squared Error (RMSE).
def compute_rmse(centroids, data):
    _, _, C, H, W = data.shape
    scale = 1 / (C * H * W)
    rmse = (scale * torch.sum((centroids - data)**2, dim=(2,3,4)))**0.5
    return rmse