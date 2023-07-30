import torch

from utils.utils import printProgressBar

def ddpm_sampling(
        diffusion_net,
        noise_degradation,
        x_t,
        min_noise=1,
        max_noise=1_000,
        cond_img=None,
        labels_tensor=None,
        device="cpu",
        log=print):
    if cond_img is not None:
        cond_img = cond_img.to(device)

    diffusion_net.eval()

    with torch.no_grad():
        for noise_step in range(max_noise, min_noise - 1, -1):
            # t: Time Step
            t = torch.tensor([noise_step], device=device)

            # Variables needed in computing x_(t-1).
            beta_t, alpha_t, alpha_bar_t = noise_degradation.get_timestep_params(t)

            if cond_img is not None:
                x_t_combined = torch.cat((x_t, cond_img), dim=1)
            else:
                x_t_combined = 1 * x_t

            # eps_param(x_t, t).
            noise_approx = diffusion_net(
                x_t_combined,
                t,
                labels_tensor)

            img_shape = x_t.shape

            # z ~ N(0, I) if t > 1, else z = 0.
            if noise_step > 1:
                z = torch.randn(img_shape, device=device)
            else:
                z = 0
            
            # sigma_t ^ 2 = beta_t = beta_hat = (1 - alpha_bar_(t-1)) / (1 - alpha_bar_t) * beta_t
            sigma_t = beta_t ** 0.5

            # x_(t-1) = (1 / sqrt(alpha_t)) * (x_t - (1 - alpha_t / sqrt(1 - alpha_bar_t)) * eps_param(x_t, t)) + sigma_t * z
            scale_1 = 1 / (alpha_t ** 0.5)
            scale_2 = (1 - alpha_t) / ((1 - alpha_bar_t)**0.5)
            
            # x_(t-1).
            x_t = scale_1 * (x_t - (scale_2 * noise_approx)) + (sigma_t * z)

            printProgressBar(
                iteration=max_noise - noise_step,
                total=max_noise - min_noise,
                prefix='Iterations:',
                suffix='Complete',
                length=50,
                log=log)
    return x_t

def ddim_sampling(
        diffusion_net,
        noise_degradation,
        x_t,
        min_noise=1,
        max_noise=1_000,
        cond_img=None,
        labels_tensor=None,
        ddim_step_size=10,
        device="cpu",
        log=print):
    diffusion_net.eval()

    steps = list(range(max_noise, min_noise - 1, -ddim_step_size))

    if not min_noise in steps:
        steps = steps + [min_noise]
            
    # 0 - Deterministic
    # 1 - DDPM
    eta = 0.0
    with torch.no_grad():
        if cond_img is not None:
            cond_img = cond_img.to(device)

        for count in range(len(steps)):
            # t: Time Step
            t = torch.tensor([steps[count]], device=device)

            if cond_img is not None:
                x_t_combined = torch.cat((x_t, cond_img), dim=1)
            else:
                x_t_combined = 1 * x_t

            # eps_theta(x_t, t).
            noise_approx = diffusion_net(
                x_t_combined,
                t,
                labels_tensor)

            # Variables needed in computing x_t.
            _, _, alpha_bar_t = noise_degradation.get_timestep_params(t)
            
            # Approximates x0 using x_t and eps_theta(x_t, t).
            # x_t - sqrt(1 - alpha_bar_t) * eps_theta(x_t, t) / sqrt(alpha_bar_t).
            scale = 1 / alpha_bar_t**0.5
            x0_approx = scale * (x_t - ((1 - alpha_bar_t)**0.5 * noise_approx))

            if count < len(steps) - 1:
                tm1 = torch.tensor([steps[count + 1]], device=device)

                # Variables needed in computing x_tm1.
                _, _, alpha_bar_tm1 = noise_degradation.get_timestep_params(tm1)

                # sigma = eta * (sqrt(1 - alpha_bar_tm1 / 1 - alpha_bar_t) * sqrt(1 - alpha_bar_t / alpha_bar_tm1)).
                sigma = eta * (
                    (
                        (1 - alpha_bar_tm1) / (1 - alpha_bar_t)
                    )**0.5 * (
                        1 - (alpha_bar_t / alpha_bar_tm1))**0.5
                )
                
                # Noise to be added (Reparameterization trick).
                eps = torch.randn_like(x0_approx)

                # As implemented in "Denoising Diffusion Implicit Models" paper.
                # x0_predicted = (1/sqrt(alpha_bar_t) * x_t - sqrt(1 - alpha_bar_t)) * eps_theta
                # xt_direction = sqrt(1 - alpha_bar_tm1 - sigma^2 * eps_theta)
                # random_noise = sqrt(sigma_squared) * eps
                # x_tm1 = sqrt(alpha_bar_t-1) * x0_predicted + xt_direction + random_noise
                x_t = (alpha_bar_tm1**0.5 * x0_approx) + ((1 - alpha_bar_tm1 - sigma**2)**0.5 * noise_approx) + (sigma * eps)

                printProgressBar(
                    iteration=max_noise - steps[count],
                    total=max_noise - min_noise,
                    prefix='Iterations:',
                    suffix='Complete',
                    length=50,
                    log=log)
    return x0_approx

def cold_diffusion_sampling(
        diffusion_net,
        noise_degradation,
        x_t,
        noise,
        min_noise=1,
        max_noise=1_000,
        cond_img=None,
        labels_tensor=None,
        skip_step_size=10,
        device="cpu",
        log=print):
    diffusion_net.eval()

    steps = list(range(max_noise, min_noise - 1, -skip_step_size))

    # Includes minimum timestep into the steps if not included.
    if not min_noise in steps:
        steps = steps + [min_noise]

    with torch.no_grad():
        for count in range(len(steps)):
            # t: Time Step
            t = torch.tensor([steps[count]], device=device)

            # Reconstruction: (x0_hat).
            if cond_img is not None:
                x_t_combined = torch.cat((x_t, cond_img), dim=1)
            else:
                x_t_combined = 1 * x_t

            x0_recon_approx = diffusion_net(
                x_t_combined,
                t,
                labels_tensor)

            if count < len(steps) - 1:
                # t-1: Time Step
                tm1 = torch.tensor([steps[count + 1]], device=device)

                # D(x0_hat, t).
                # Noise degraded image (x_t).
                # x_t(X_0, eps) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.
                x_t_hat = noise_degradation(
                    img=x0_recon_approx,
                    steps=t,
                    eps=noise)

                # D(x0_hat, t-1).
                # Noise degraded image (x_t).
                # x_t(X_0, eps) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.
                x_tm1_hat = noise_degradation(
                    img=x0_recon_approx,
                    steps=tm1,
                    eps=noise)
                
                # q(x_t-1 | x_t, x_0).
                # Improved sampling from Cold Diffusion paper.
                x_t = x_t - x_t_hat + x_tm1_hat

                printProgressBar(
                    iteration=max_noise - steps[count],
                    total=max_noise - min_noise,
                    prefix='Iterations:',
                    suffix='Complete',
                    length=50,
                    log=log)
    return x0_recon_approx
