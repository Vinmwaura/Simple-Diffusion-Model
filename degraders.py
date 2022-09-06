import torch
import torch.nn as nn

import torchvision


class NoiseDegradation(nn.Module):
    def __init__(self, beta=0.005):
        super().__init__()

        self.beta = beta
        """
        min_variance_val = 0.0001
        max_variance_val = 0.02
        max_noise_step = 1000
        self.beta = torch.linspace(min_variance_val, max_variance_val, int(max_timestep + 1))
        self.alpha = 1 - self.beta
        self.alpha_cumulative_prod = torch.cumprod(self.alpha, dim=0).to(device)
        """

    def forward(self, img, steps, eps=None):
        if eps is None:
            eps = torch.randn_like(img)
        """
        alpha_bar = torch.gather(self.alpha_cumulative_prod, dim=0, index=steps)
        alpha_bar = alpha_bar[:, None, None, None]
        img_degradation = alpha_bar**0.5 * img + (1 - alpha_bar)**0.5 * eps
        """
        alpha = 1 - self.beta
        alpha_bar = alpha**steps
        alpha_bar = alpha_bar[:, None, None, None]
        img_degradation = (alpha_bar)**0.5 * img + (1 - alpha_bar)**0.5 * eps  # X_t
        return img_degradation
