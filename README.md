# Simple-Diffusion-Model

This repository contains a simple implementation of Diffusion generative models, implemented using Pytorch, which is a machine learning algorithm predominately used in the generation of images from noise.

## What is Diffusion Model (Simplified)
These are machine learning models that are trained to be able to generate data, usually images, by systematically and slowly destroying the structure in the data distribution through an iterative **forward diffusion process** i.e Gaussian Noise is slowly added to the data in incremental steps until the degraded data is approximately equivalent to the Gaussian Noise being added. In each step the model is trained to learn a **reverse diffusion process** that restores the structure in the data.

This works due to the fact that the model learns to gradually convert one distribution i.e a normal distribution: $$X \sim \mathcal{N}(0,I)$$, which is simple and easy to model using artificial neural networks into the target distribution which is usually complex and hard to model e.g picture of cats over a series of steps where each step is only dependant on the prior one excluding the initial step.

Generating data from a trained model begins by sampling from noise e.g Gaussian distribution and over a series of steps (Reverse Diffusion) the target distribution e.g images of a cat is slowly restored.

![Depiction of Forward Diffusion Process and Reverse Diffusion Process](https://github.com/Vinmwaura/Simple-Diffusion-Model/assets/12788331/f172f38b-34b6-4383-ace9-3c642bf879f6)
*Figure 1: Image showcasing Forward Diffusion Process of an image of a cat slowly being degraded into noise and Reverse Diffusion Process of noise being converted back to an image of a cat. From Denoising Diffusion-based Generative Modeling: Foundations and Applications, by Karsten Kreis, Ruiqi Gao, Arash Vahdat. Retrieved from https://cvpr2022-tutorial-diffusion-models.github.io*

## Implementation
The various techniques to add noise to the data distribution include, but are not limited to, the following:

### Linear Noise Scheduler
Here two parameters beta<sub>1</sub> and beta<sub>T</sub> are used to determine the rate at which noise is added to the data at each timestep. For the initial step T the value will be close to 1 and at the final step t<sub>1</sub>, the value is closer to 0.

### Cosine Noise Scheduler
This was designed as an improvement to the Linear Noise Scheduler, it was shown to improve the quality of images in some papers. It only requires the final step T as a parameter.

![Linear and Cosine scheduler comparison](https://github.com/Vinmwaura/Simple-Diffusion-Model/assets/12788331/580b2009-dad0-4652-9bba-588f0733a77c)
*Figure 2: Graph comparison of Linear and Cosine Scheduler. From Improved Denoising Diffusion Probabilistic Models, by Nichol & Dhariwal, 2021. Retrieved from https://arxiv.org/abs/2102.09672*

The various techniques to restore the data distribution from Gaussian Noise include the following:

### Denoising Diffusion Probabilistic Model (DDPM)
Here the data is systematically restored slowly from noise step by step by having the model attempt to predict the noise that was added to the data from the initial step, T. The predicted noise value, denoted as \epsilon and the current degraded data denoted as x<sub>t</sub> where t is the current step, is then used to predict the next step’s degraded data denoted as x<sub>t-1</sub>. This is repeated until the final step of t<sub>1</sub> is computed to get back the data denoted as x<sub>0</sub>.

![Depiction of DDPM](https://github.com/Vinmwaura/Simple-Diffusion-Model/assets/12788331/7c405b83-31b0-4e34-8c46-04f0f461085e)
*Figure 3: Representation of the multiple steps in a DDPM (Markov chain) of forward diffusion process and reverse diffusion process. From Denoising Diffusion Probabilistic Models, by Ho et al. 2020. Retrieved from https://arxiv.org/abs/2006.11239*

### Denoising Diffusion Implicit Model (DDIM)
This technique is similar to DDPM in that the model tries to predict the noise added to the image at each step, however this technique attempts to speed up the process by using x<sub>t</sub> and noise approximated by the model and computes the data as it would appear in the final step. Then, some steps will be skipped and the predicted data will have noise re-added to it for the next step. This process is repeated until the final step t<sub>1</sub> is reached. Skipping multiple steps makes this the faster option. However with an increase in skipped steps, data quality goes down.

![Depiction of DDIM](https://github.com/Vinmwaura/Simple-Diffusion-Model/assets/12788331/4a02d7e1-4132-4cbe-8686-eeb07b575de9)

*Figure 4: Representation of the skipped steps in a DDIM (Markov chain) of forward diffusion process and reverse diffusion process. From Denoising Diffusion Implicit Models, by Song et al. 2022. Retrieved from https://arxiv.org/abs/2010.02502*

### Cold Diffusion
In this approach the model directly attempts to reconstruct the data (x<sub>0</sub>) at each step from the degraded image x<sub>t</sub>, where some steps can be skipped similar to DDIM.

Combining all of the above techniques, the model(s) can be trained in various configurations to improve image quality, and image size. These include but are not limited to:
### Cascaded Diffusion Models for High Fidelity Image Generation
This comprises a pipeline of multiple diffusion models that generate images of increasing resolution, beginning with a standard diffusion model (Base model) at the lowest resolution, followed by one or more Super-Resolution diffusion models that successively upsamples the image.

![Example of cascaded diffusion models with a photo of a dog](https://github.com/Vinmwaura/Simple-Diffusion-Model/assets/12788331/d42e88c6-9603-4afd-b82c-5f6d0616f619)
*Figure 5: A cascaded pipeline of multiple diffusion models at increasing resoltions. From Cascaded Diffusion Models for High Fidelity Image Generation, by Ho et al. 2022. Retrieved from https://arxiv.org/abs/2106.15282*

### eDiffi: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers
This comprises an ensemble of diffusion models i.e multiple models, specialized for different synthesis stages. Multiple models are trained on one dataset but each model is only trained on a specific range of steps so it becomes specialized in that range only. This can improve data quality as the capacity of each diffusion model is increased due to only focussing on a smaller range of degraded images. Sampling from these models requires loading the models one after the other for their respective range.

![Depiction of ensemble of models generating an image](https://github.com/Vinmwaura/Simple-Diffusion-Model/assets/12788331/8651c67f-d791-4f16-976e-836d33152c6d)
*Figure 6: An ensemble of diffusion models that are specialized for denoising at different intervals of the generative process. From eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers, by Balaji et al. 2023. Retrieved from https://arxiv.org/pdf/2211.01324.pdf*

## Requirements
+ Python 3
+ [Optional] [Cuda-enabled GPU](https://developer.nvidia.com/cuda-gpus) or equivalent supported hardware.

## Set-Up
1. Install [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/install.html).
2. Create a virtual enviroment:
```
mkvirtualenv diffusion_env
```
3. To activate virtualenv:
```
workon diffusion_env
```
4. To install the libraries needed by the project (Pytorch + Others) run the following script (This installs the CPU-only Pytorch libraries, which is lighter and runs on most machine):
```
sudo chmod +x install_cpu_requirements.sh
sh install_cpu_requirements.sh
```
[Optional] (Ignore Step 4) If you want to run and/or train using this project with the benefit of speed (hardware acceleration), you will require to install the appropriate [Pytorch](https://pytorch.org/) library and it's dependencies specific to you machine then you can install the additional python libraries needed by the project:
```
pip install -r model_requirements.txt
```

**NOTE**: The code was tested and run on Ubuntu 22.04.3 LTS, which was running on a CUDA-enabled GPU with additional libraries not shown above.

## Training Models
1. Create a training config file by executing the following command and follow the prompts:
+ For Base Diffusion (DDPM / DDIM) and Cold Diffusion Models:
```
python create_diffusion_config.py
```

+ For Super-Resolution Diffusion Models:
```
python create_sr_diffusion_config.py
```

+ For Base Diffusion (DDPM / DDIM) using images e.g doodles as conditional input:
```
python create_doodle_diffusion_config.py
```

2. To train a model, use the config files created above and run the folowing commands:
+ For Base Diffusion (DDPM / DDIM) models:
```
python train_diffusion.py --config-path "<File path to training config file>" --device <Device model will use>
```

+ For Cold Diffusion models:
```
python train_noise_cold_diffusion.py --config-path "<File path to training config file>" --device <Device model will use>
```

+ For Base Diffusion (DDPM / DDIM) models using images e.g doodles as conditional input:
```
python train_doodle_diffusion.py --config-path "<File path to training config file>" --device <Device model will use>
```

+ For Super-Resolution Diffusion models (Uses Cold-Diffusion algorithm):
```
python train_SR_diffusion.py --config-path "<File path to training config file>" --device <Device model will use>
```

## Generating Images
The scripts requires there to be **folder** with **model checkpoint** file and a **json config** file to work.

To generate the above mentioned folder and respective files, run the following command in the terminal and follow the prompts (Requires a trained model checkpoint file and training config file):
```
python export_models.py
```

To generate an image from a model, run one of the following commands in a terminal:
+ From trained Base Diffusion (DDPM / DDIM) models:
```
python generate_images_diffusion.py --num_images <Number of images shown in grid> -l <Conditional input i.e class labels, ignore if none> --device <Device model will run on> --diff_alg <DDPM/DDIM algorithm to use> --ddim_step_size <Steps skipped if using DDIM> --seed <Optional seed value for same output> --config "<File Path to model config file>" --dest_path <Optional path to save generated image> --max_T <Model's parameter value for noise scheduler>
```

+ From trained Cold Diffusion models:
```
python generate_images_cold_diffusion.py --num_images <Number of images shown in grid> --labels <OConditional input i.e class labels, ignore if none> --device <Device model will run on> --cold_step_size <Steps to be skipped> --seed <Optional seed value for same output> --config "<File Path to model config file>" --dest_path <Optional path to save generated image> --max_T <Model's parameter value for noise scheduler>
```

+ From trained Doodle Diffusion models:
```
python generate_images_diffusion.py --num_images <Number of images shown in grid> --labels <Conditional input i.e class labels, ignore if none> --device <Device model will run on> --diff_alg <DDPM/DDIM algorithm to use> --ddim_step_size <Steps skipped if using DDIM> --seed <Optional seed value for same output> --config "<File Path to model config file>" --dest_path <Optional path to save generated image> --max_T <Model's parameter value for noise scheduler> --cond_img_path <File path to conditional image e.g Doodle image.>
```

+ From trained Super-Resolution models:
```
python generate_sr_images_diffusion.py --device <Device model will run on> --config "<File Path to model config file>" --seed <Optional seed value for same output> --dest_path <Optional path to save generated image> --cold_step_size <Steps to be skipped> --labels <Conditional input i.e class labels, ignore if none> --lr_img_path <File path to Low resoluion image that is to be upsampled> --max_T <Model's parameter value for noise scheduler>
```

## Trained Model Weights
You can find some trained models [here](https://huggingface.co/VinML/Custom-Simple-Diffusion-Model), they include the following files:
+ AnimePortraits
+ AnimePortraits_SR
+ CelebFaces
+ MyBodyPose
+ MyBodyPose_SR
+ MyFace
+ MyFace_SR

**NOTE**: In this project the base resolution models (AnimePortraits, CelebFaces, MyBodyPose, MyFace) was trained on images with dimensions of 128\*128\*3 and the super-resolution models (AnimePortraits_SR, MyBodyPose_SR, MyFace_SR) upsamples the respective base resolution images to 256\*256\*3.

## Examples of generated outputs
### AnimePortraits
![AnimePortraits](https://github.com/Vinmwaura/Simple-Diffusion-Model/assets/12788331/c81bc8fd-8e93-4cba-9297-f2a9ac0a0143)

### CelebFaces
![CelebFaces](https://github.com/Vinmwaura/Simple-Diffusion-Model/assets/12788331/482b7896-c8ab-4ef4-97e1-7e3d6b6cfcf1)

### MyBodyPose
**Input** (Conditional input displayed as a grid):

![MyBodyPose_input](https://github.com/Vinmwaura/Simple-Diffusion-Model/assets/12788331/7d03ea79-da5a-4ad4-951e-fd1017e08b7f)

**Output**:

![MyBodyPose_output](https://github.com/Vinmwaura/Simple-Diffusion-Model/assets/12788331/3fa5fc25-7b16-46f6-ab99-f9cca907f8fc)

### MyFace

![MyFace](https://github.com/Vinmwaura/Simple-Diffusion-Model/assets/12788331/6f0b85a6-9d23-4592-9ccc-8e3de8d5fbf1)

**NOTE**: Text conditional input was not used in the project like with most text-to-image diffusion models however some form of single label, multi-label or conditional image input was used to steer the resulting output of the models.

## Learning Resources Used to understand and implement Diffusion Models
+ Anime Portraits Dataset -> https://gwern.net/crop#danbooru2019-portraits
+ Celebrity Faces Dataset -> http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
+ Attention mechanism -> https://arxiv.org/abs/1706.03762
+ Deep Unsupervised Learning using Nonequilibrium Thermodynamics -> https://arxiv.org/abs/1503.03585
+ Denoising Diffusion Probabilistic Models -> https://arxiv.org/abs/2006.11239
+ Denoising Diffusion Implicit Models -> https://arxiv.org/abs/2010.02502
+ Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise -> https://arxiv.org/abs/2208.09392
+ Improved Denoising Diffusion Probabilistic Models -> https://arxiv.org/abs/2102.09672
+ Cascaded Diffusion Models for High Fidelity Image Generation -> https://arxiv.org/abs/2106.15282
+ eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers -> https://arxiv.org/abs/2211.01324
+ Denoising Diffusion-based Generative Modeling: Foundations and Applications -> https://cvpr2022-tutorial-diffusion-models.github.io/
+ SRDiff: Single Image Super-Resolution with Diffusion Probabilistic Models -> https://arxiv.org/abs/2104.14951
+ Understanding Diffusion Models: A Unified Perspective -> https://arxiv.org/abs/2208.11970
+ U-Net: Convolutional Networks for Biomedical Image Segmentation -> https://arxiv.org/abs/1505.04597
+ What are Diffusion Models? -> https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
+ U-Net model for Denoising Diffusion Probabilistic Models (DDPM) Implementation -> https://nn.labml.ai/diffusion/ddpm/unet.html
+ Denoising Diffusion Implicit Models (DDIM) Sampling Implementation -> https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddim.html
+ Deriving the variational lower bound -> http://paulrubenstein.co.uk/deriving-the-variational-lower-bound/
+ Introduction to Diffusion Models for Machine Learning -> https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/
+ How diffusion models work: the math from scratch -> https://theaisummer.com/diffusion-models/
