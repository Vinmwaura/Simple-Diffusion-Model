import torch
import torch.nn as nn


class NoiseDegradation(nn.Module):
    def __init__(self, beta=0.9):
        super().__init__()
        self.beta = beta

    def forward(self, img, step=0, eps=None):
        if eps is None:
            eps = torch.randn_like(img)
        
        if isinstance(step, torch.Tensor):
            # Add dimension: (N,1, 1, 1).
            step = step[:,None, None, None]
        
        alpha = 1 - self.beta
        img_degradation = (alpha**step)**0.5 * img + (1 - alpha**step)**0.5 * eps  # X_t
        return img_degradation
