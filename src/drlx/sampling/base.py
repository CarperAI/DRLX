from typing import Union, Iterable, Tuple, Any, Optional
from torchtyping import TensorType

import torch
from tqdm import tqdm
import math
import einops as eo
import torch.nn.functional as F

from drlx.utils import rescale_noise_cfg
from drlx.configs import SamplerConfig

class Sampler:
    """
    Generic class for sampling generations using a denoiser. Assumes LDMUnet
    """
    def __init__(self, config : SamplerConfig = SamplerConfig()):
        self.config = config

    def cfg_rescale(self, pred : TensorType["2 * b", "c", "h", "w"]):
        """
        Applies classifier free guidance to prediction and rescales if cfg_rescaling is enabled

        :param pred:
            Assumed to be batched repeated prediction with first half consisting of
            unconditioned (empty token) predictions and second half being conditioned
            predictions
        """

        pred_uncond, pred_cond = pred.chunk(2)
        pred = pred_uncond + self.config.guidance_scale * (pred_cond - pred_uncond)

        if self.config.guidance_rescale is not None:
            pred = rescale_noise_cfg(pred, pred_cond, self.config.guidance_rescale)
        
        return pred

    @torch.no_grad()
    def sample(self, prompts : Iterable[str], denoiser, device = None, show_progress : bool = False, accelerator = None):
        """
        Samples latents given some prompts and a denoiser

        :param prompts: Text prompts for image generation (to condition denoiser)
        :param denoiser: Model to use for denoising
        :param device: Device on which to perform model inference
        :param show_progress: Whether to display a progress bar for the sampling steps
        :param accelerator: Accelerator object for accelerated training (optional)

        :return: Latents unless postprocess flag is set to true in config, in which case VAE decoded latents are returned (i.e. images)
        """
        if accelerator is None:
            denoiser_unwrapped = denoiser
        else:
            denoiser_unwrapped = accelerator.unwrap_model(denoiser)

        scheduler = denoiser_unwrapped.scheduler
        preprocess = denoiser_unwrapped.preprocess
        noise_shape = denoiser_unwrapped.get_input_shape()

        text_embeds = preprocess(
            prompts, mode = "embeds", device = device,
            num_images_per_prompt = 1,
            do_classifier_free_guidance = self.config.guidance_scale > 1.0
        ).detach()

        scheduler.set_timesteps(self.config.num_inference_steps, device = device)
        latents = torch.randn(len(prompts), *noise_shape, device = device)

        for i, t in enumerate(tqdm(scheduler.timesteps), disable = not show_progress):
            input = torch.cat([latents] * 2)
            input = scheduler.scale_model_input(input, t)

            pred = denoiser(
                pixel_values=input, 
                time_step = t,
                text_embeds = text_embeds
            )

            # guidance
            pred = self.cfg_rescale(pred)

            # step backward
            scheduler_out = scheduler.step(pred, t, latents, self.config.eta)
            latents = scheduler_out.prev_sample

        if self.config.postprocess:
            return denoiser_unwrapped.postprocess(latents)
        else:
            return latents