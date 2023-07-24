In this folder there are two JSON files showing the format to represent the trained models for use with the generate_* scripts:
* generate_images_cold_diffusion.py - Will generate images using Cold Diffusion (https://arxiv.org/abs/2208.09392)
* generate_images_diffusion.py - Will generate images using eDiffi implementaion, DDPM/DDIM model or Cascaded Diffusion Models or a mixture of all.

They include the following parameters:
- models: This is either a list of models, in cases of ensemble models (multiple models each focusing on a small
            timestep: https://arxiv.org/abs/2211.01324) or a list of one model.
- upsample_models: This is either a list of super resolution models, in cases of ensemble models
                (multiple models each focusing on a small timestep: https://arxiv.org/abs/2211.01324) or a list of 
                one model that will be used in upscaling output from prior model: https://arxiv.org/abs/2106.15282 .(Can be ignored if no upsampling model is used)

Both of these parameters include the following:
- "model_path": File path to saved model .pt checkpoint (Note: pytorch load and save uses pickle and so has security vulnerabilities,
                only load models you trust and can verify where they come from.)
- "img_C": Int image channel to be generated, usually int value of 3.
- "img_H": Int image Height dimension to be generated, should be the one model was trained on otherwise model gives nonsensical output (GIGO).
- "img_W": Int image Width dimension to be generated, should be the one model was trained on otherwise model gives nonsensical output (GIGO).
- "in_channel": Int parameter to be passed in model for input channel that goes in the model.
- "out_channel": Int parameter to be passed in model for output channel that is generated from the model.
- "num_layers": Int parameter to be passed in model for number of layers used in the model. 
- "num_resnet_block": Int parameter to be passed in model for number of residual layers in each layer used in the model.
- "attn_layers": Array of int parameters to be passed in model for each layer containing attention mechanism 
                (Represented as a list of indices starting from 0 i.e for a model with 4 layers: [0, 1, 2, 3]).
- "attn_heads": Int parameter to be passed in model for how many heads in attention mechanism.
- "attn_dim_per_head": Int or null parameter to be passed in model for dimension used in heads of attention layer, leave null to use default channel size per layer.
- "time_dim": Int parameter to be passed in model for determining dimension of timestep information: t.
- "cond_dim": Int parameter to be passed in model for determining dimension of conditional information such as class labels or embeddings.
- "min_channel": Int parameter to be passed in model for determining minimum channel in earlier layers, each layer's channel size is doubled from the prior.
- "max_channel": Int parameter to be passed in model for determining maximum channel size, each layer's channel size is doubled from the prior.
- "image_recon": Boolean Parameter to be passed in model to determine if the last layer will use a Tanh(-1, 1) activation layer usually used in Cold Diffusion model or not.
- "max_noise": Int Parameter to be passed in model to determine maximum timestep for adding Gaussian noise. 
- "min_noise": Int Parameter to be passed in model to determine minimum timestep for adding Gaussian noise, minimum value can be 1.
- "beta_1": Double Parameter to determine first forward process variances used in linear noise scheduling, for more details check: https://arxiv.org/abs/2006.11239
- "beta_T": Double Parameter to determine final forward process variances used in linear noise scheduling, for more details check: https://arxiv.org/abs/2006.11239

List of resources to better understand the model parameters (In no particular order):
> https://arxiv.org/pdf/2006.11239.pdf (DDPM)
> https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/
> https://nn.labml.ai/diffusion/ddpm/index.html
> https://arxiv.org/abs/2102.09672 (Improved Denoising Diffusion Probabilistic Models)
> https://theaisummer.com/diffusion-models/
> https://arxiv.org/abs/1706.03762 (Attention Is All You Need)
> https://arxiv.org/abs/2106.15282 (Cascaded Diffusion Models for High Fidelity Image Generation)
> https://arxiv.org/abs/2211.01324 (eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers)
> https://arxiv.org/abs/2010.02502 (Denoising Diffusion Implicit Models)

NOTE: Model, code and parameters are not exactly similar to how they have been implemented in the various papers, some liberty was taken based on my understanding and hardware limitation.