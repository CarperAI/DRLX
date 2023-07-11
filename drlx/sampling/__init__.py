from typing import Union, Iterable, Tuple, Any

import torch
from fastprogress import progress_bar
import math

from drlx.configs import SamplerConfig

# Credit to Tanishq Abraham (tmabraham) for notebook from which
# both sampler and ddpo sampler code was adapted

class Sampler:
    """
    Generic class for sampling generations using a denoiser. Assumes LDMUnet
    """
    def __init__(self, config : SamplerConfig = SamplerConfig()):
        self.config = config

    @torch.no_grad()
    def sample(self, prompts : Iterable[str], denoiser, guidance_scale = None, num_inference_steps = None, eta = None, device = None) -> Any:
        if not guidance_scale:
            guidance_scale = self.config.guidance_scale
        if not num_inference_steps:
            num_inference_steps = self.config.num_inference_steps
        if not eta:
            eta = self.config.eta
        if not device:
            device = self.config.device

        scheduler = denoiser.scheduler
        
        text_embeds = denoiser.encode_prompt(prompts,device, 1, do_classifier_free_guidance=guidance_scale > 1.0).detach()

        scheduler.set_timesteps(num_inference_steps, device = device)
        latents = torch.randn(len(prompts), *denoiser.get_input_shape(), device = device)

        for i, t in enumerate(progress_bar(scheduler.timesteps)):
            input = torch.cat([latents] * 2)
            input = scheduler.scale_model_input(input, t)

            pred = denoiser(
                pixel_values=input, 
                time_step = t,
                text_embeds = text_embeds
            )

            # guidance
            pred_uncond, pred_text = pred.chunk(2)
            pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)
    
            # step backward
            scheduler_output = scheduler.step(pred, t, latents, eta, variance_noise=0)
            t_1 = t - scheduler.config.num_train_timesteps // num_inference_steps

            variance = scheduler._get_variance(t, t_1)
            std_dev_t = eta * variance ** 0.5
            prev_sample_mean = scheduler_output.prev_sample
            prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

            latents = prev_sample

        if self.config.postprocess:
            return denoiser.postprocess(latents)
        else:
            return latents

class LDMSampler(Sampler):
    @torch.no_grad()
    def sample(self, prompt, denoiser, **kwargs):
        latents = super().sample(prompt, denoiser, **kwargs)
        if self.config.postprocess:
            return denoiser.postprocess(latents)
        else:
            return latents

class DDPOSampler(Sampler):

    def calc_log_probs(self, prev_sample, prev_sample_mean, std_dev_t):
        std_dev_t = torch.clip(std_dev_t, 1e-6)
        log_probs = -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * std_dev_t ** 2) - torch.log(std_dev_t) - math.log(math.sqrt(2 * math.pi))
        return log_probs.mean(dim=tuple(range(1, prev_sample_mean.ndim)))

    @torch.no_grad()
    def sample(self, prompts : Iterable[str], denoiser, guidance_scale = None, num_inference_steps = None, eta = None, device = None) -> Any:
        if not guidance_scale:
            guidance_scale = self.config.guidance_scale
        if not num_inference_steps:
            num_inference_steps = self.config.num_inference_steps
        if not eta:
            eta = self.config.eta
        if not device:
            device = self.config.device

        scheduler = denoiser.scheduler

        text_embeds = denoiser.encode_prompt(prompts,device, 1, do_classifier_free_guidance=guidance_scale > 1.0).detach()

        scheduler.set_timesteps(num_inference_steps, device = device)
        latents = torch.randn(len(prompts), *denoiser.get_input_shape(), device = device)

        all_step_preds, log_probs = [latents], []

        for i, t in enumerate(progress_bar(scheduler.timesteps)):
            input = torch.cat([latents] * 2)
            input = scheduler.scale_model_input(input, t)

            pred = denoiser(
                pixel_values=input, 
                time_step = t,
                text_embeds = text_embeds
            )

            # guidance
            pred_uncond, pred_text = pred.chunk(2)
            pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)

            # step backward
            scheduler_output = scheduler.step(pred, t, latents, eta, variance_noise=0)
            t_1 = t - scheduler.config.num_train_timesteps // num_inference_steps

            variance = scheduler._get_variance(t, t_1)
            std_dev_t = eta * variance ** 0.5
            prev_sample_mean = scheduler_output.prev_sample
            prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

            # Main difference here
            all_step_preds.append(prev_sample)
            log_probs.append(
                self.calc_log_probs(
                    prev_sample, prev_sample_mean, std_dev_t
                )
            )

            latents = prev_sample

        return latents, all_step_preds, log_probs





        



        