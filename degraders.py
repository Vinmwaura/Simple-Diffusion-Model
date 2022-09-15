import math

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


# Cosine Scheduler
class CosineNoiseDegradation(nn.Module):
    def __init__(self, max_noise_step=1000):
        super().__init__()
        
        self.max_noise_step = max_noise_step
        self.offset = 0.008

    def compute_alpha_bar(self, steps):
        # f(t)
        eq_1 = ((steps / self.max_noise_step) + self.offset) / (1 + self.offset)
        eq_2 = math.pi / 2
        alpha_bar_t = torch.cos(eq_1 * eq_2)**2

        # f(0)
        zero_ = torch.zeros_like(steps)
        eq_1 = ((zero_ / self.max_noise_step) + self.offset) / (1 + self.offset)
        alpha_bar_0 = torch.cos(eq_1 * eq_2)**2

        # f(t) / f(0)
        alpha_bar = alpha_bar_t / alpha_bar_0
        return alpha_bar

    def get_timestep_params(self, steps):
        alpha_bar = self.compute_alpha_bar(steps)
        alpha_bar_prev = self.compute_alpha_bar(steps - 1)

        # Clip Beta.
        beta = 1 - (alpha_bar / alpha_bar_prev)
        beta = torch.clip(beta, min=0.001, max=0.999)

        alpha = 1 - beta
        return beta, alpha, alpha_bar

    def forward(self, img, steps, eps=None):
        if eps is None:
            eps = torch.randn_like(img)
        
        alpha_bar = self.compute_alpha_bar(steps)
        alpha_bar = alpha_bar[:, None, None, None]
        
        img_degradation = alpha_bar**0.5 * img + (1 - alpha_bar)**0.5 * eps
        return img_degradation
