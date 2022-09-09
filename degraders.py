import torch
import torch.nn as nn

import torchvision


class NoiseDegradation(nn.Module):
    def __init__(
            self,
            beta_1,
            beta_T,
            max_noise_step,
            device="cpu"):
        super().__init__()
        
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.max_noise_step = max_noise_step
        
        self.beta = torch.linspace(
            start=self.beta_1,
            end=self.beta_T,
            steps=int(self.max_noise_step + 1),
            device=device)
        self.alpha = 1 - self.beta
        self.alpha_cumulative_prod = torch.cumprod(self.alpha, dim=0)

    def get_timestep_params(self, step):
        beta = torch.gather(self.beta, dim=0, index=step)
        alpha = torch.gather(self.alpha, dim=0, index=step)
        alpha_bar = torch.gather(self.alpha_cumulative_prod, dim=0, index=step)
        return beta, alpha, alpha_bar

    def forward(self, img, steps, eps=None):
        if eps is None:
            eps = torch.randn_like(img)

        alpha_bar = torch.gather(self.alpha_cumulative_prod, dim=0, index=steps)
        alpha_bar = alpha_bar[:, None, None, None]
        
        img_degradation = alpha_bar**0.5 * img + (1 - alpha_bar)**0.5 * eps
        return img_degradation
