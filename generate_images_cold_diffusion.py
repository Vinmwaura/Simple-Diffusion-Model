import os
import uuid
import argparse
import pathlib

import torch
torch.manual_seed(69)

import torchvision

# Degradation Operators.
from degraders import *

# U Net Model.
from models.U_Net import U_Net
from diffusion_enums import *
from utils import load_checkpoint, plot_sampled_images, printProgressBar


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Images using diffusion models.")
    parser.add_argument("--device", help = "Which device model will use(default='cpu').", choices=['cpu', 'cuda'], type=str, default="cpu")
    parser.add_argument("--dim", help = "Image Dimension. (default: 128).", default=128, type=int)
    parser.add_argument("--in_channel", help = "U Net Model In channel config. (default: 3).", default=3, type=int)
    parser.add_argument("--out_channel", help = "U Net Model Out channel config. (default: 3).", default=3, type=int)
    parser.add_argument("--num_layers", help = "U Net Model Layers used. (default: 5).", default=5, type=int)
    parser.add_argument("--time_dim", help = "U Net time channels used. (default: 64).", default=64, type=int)
    parser.add_argument("--min_channel", help = "U Net min channels used. (default: 64).", default=128, type=int)
    parser.add_argument("--max_channel", help = "U Net max channels used. (default: 512).", default=512, type=int)
    parser.add_argument("--image_recon", help = "U Net model uses tanh in last layer. (default: False).", default=False, type=bool)
    parser.add_argument("--attn_layers", help = "U Net layers using attention. (default: 3, 4).", nargs='*', default=[2, 3, 4], type=int)

    parser.add_argument("-dm", "--diff_model", help = "File path of diffusion model to be used.", required=True, type=pathlib.Path)
    parser.add_argument("-d", "--dest_path", help = "File path to save images generated (Default: out folder).", type=pathlib.Path)
    
    parser.add_argument("--noise_alg", help = "Noise Degradation scheduler to use (default: linear).", default="linear", choices=[noise_scheduler.name.lower() for noise_scheduler in NoiseScheduler])
    parser.add_argument("-b1", "--beta_1", help = "Beta 1 value.", default=5e-3, type=float)
    parser.add_argument("-bT", "--beta_T", help = "Beta T value.", default=5e-3, type=float)
    parser.add_argument("-T", "--max_T", help = "Max T value to start from.", default=1000, type=int)
    parser.add_argument("-n", help = "Number of images to generate(default=1).", default=1, type=int)

    args = vars(parser.parse_args())
    
    try:
        assert args["n"] > 0
        img_shape = (args["n"], 3, args["dim"], args["dim"])
        

        if args["dest_path"] is None:
            out_dir = "./"
        else:
            assert args["dest_path"].exists()
            out_dir = str(args["dest_path"])

        # x_t(X_0, eps) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.
        if args["noise_alg"] == NoiseScheduler.LINEAR.name.lower():
            noise_degradation = NoiseDegradation(
                args["beta_1"],
                args["beta_T"],
                args["max_T"],
                args["device"])
        elif args["noise_alg"] == NoiseScheduler.COSINE.name.lower():
            noise_degradation = CosineNoiseDegradation(args["max_T"])
        
        # Diffusion Model.
        diffusion_net = U_Net(
            in_channel=args["in_channel"],
            out_channel=args["out_channel"],
            num_layers=args["num_layers"],
            attn_layers=args["attn_layers"],
            time_dim=args["time_dim"],
            min_channel=args["min_channel"],
            max_channel=args["max_channel"],
            image_recon=True
        ).to(args["device"])

        # Load Diffusion Checkpoints.
        assert args["diff_model"].exists()
        assert args["diff_model"].is_file()
        load_diff_status, diffusion_checkpoint = load_checkpoint(str(args["diff_model"]))
        assert load_diff_status

        diffusion_net.load_state_dict(diffusion_checkpoint["model"])
        
        # Evaluate model.
        diffusion_net.eval()
        
        # X_T ~ N(0, I)
        noise = torch.randn(img_shape, device=args["device"])
        x_t_plot = 1 * noise

        with torch.no_grad():
            # steps_start = list(range(1000, 900, -10))
            # steps_end = list(range(90, 0, -1))
            # steps = steps_start + [900, 800, 700, 600, 500, 400, 300, 200, 100] + steps_end
            steps = list(range(1000, 0, -10)) + [1]
            # steps = [1000, 950, 900, 850, 750, 500, 250, 125, 100, 75, 1]
            for count in range(len(steps)):
                # t: Time Step
                t = torch.tensor([steps[count]], device=args["device"])

                # x_t_combined = torch.cat((x_t_plot, eps_approx), dim=1)
                x0_recon_approx_plot = diffusion_net(
                    x_t_plot,
                    t)

                # q(x_t-1 | x_t, x_0).
                if count < len(steps) - 1:
                    # t-1: Time Step
                    tm1 = torch.tensor([steps[count + 1]], device=args["device"])

                    # Noise degraded image (x_t-1).
                    # x_t(X_0, eps) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.
                    x_t_prev_plot = noise_degradation(
                        img=x0_recon_approx_plot,
                        steps=tm1,
                        eps=noise)

                    # Noise degraded image (x_t-1).
                    # x_t(X_0, eps) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.
                    x_t_curr_plot = noise_degradation(
                        img=x0_recon_approx_plot,
                        steps=t,
                        eps=noise)

                    # Noise degraded image (x_t-1).
                    # x_t(X_0, eps) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps.
                    x_t_plot = x_t_plot - x_t_curr_plot + x_t_prev_plot

                printProgressBar(
                    args["max_T"] - steps[count],
                    args["max_T"],
                    prefix = 'Iterations:',
                    suffix = 'Complete',
                    length = 50)

            picture_name = str(uuid.uuid4())
            plot_sampled_images(
                sampled_imgs=x0_recon_approx_plot,  # x_0
                file_name=picture_name,
                dest_path=out_dir)
        
        print(f"\nSaving generated image: %s" % os.path.join(out_dir, picture_name))

    except Exception as e:
        print(e)
        print(f"An error occured while generating images: {e}")
