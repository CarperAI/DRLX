from typing import Union, Iterable, Tuple, Any

import torch
from fastprogress import progress_bar
from tqdm import tqdm
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

    def sample(self, use_grad = False, **kwargs):
    if not use_grad:
        with torch.no_grad():
            return _sample(**kwargs)
    else:
        return _sample(**kwargs)

    def _sample(self, prompts : Iterable[str], denoiser, guidance_scale = None, num_inference_steps = None, eta = None, device = None) -> Any:
        guidance_scale = self.config.guidance_scale
        num_inference_steps = self.config.num_inference_steps
        eta = self.config.eta

        scheduler = denoiser.scheduler
        
        text_embeds = denoiser.preprocess(prompts, mode = "embeds", device, 1, do_classifier_free_guidance=guidance_scale > 1.0).detach()

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

    def _sample(self, prompts : Iterable[str], denoiser, device = None, show_progress : bool = False,
     advantages = None, old_preds = None, old_log_probs = None, method_config : 'DDPOConfig' = None
     ) -> Any:
        """
        Using model and prompts, sample for DDPO training. In normal state, samples final samples from prompts with denoiser.
        Enters train mode (compute_loss is True) when all of: advantages, old_preds, old_log_probs, method_config are provided.
        In this mode, computes loss and calls backward. Uses the method_config for hyperparameters. This is tied to the sampler
        as sampling and loss computation are heavily related for DDPO.
        """

        # Computing loss after sampling if advantage, samples and log probabilities are provided
        compute_loss : bool = (advantages is not None) and (old_log_probs is not None) and (old_preds is not None) and (method_config is not None)

        if compute_loss:
            advantages = advantages.to(device)
            old_preds = old_preds.to(device)
            old_log_probs = old_log_probs.to(device)

        scheduler = denoiser.scheduler
        guidance_scale = self.config.guidance_scale
        eta = self.config.eta
        num_inference_steps = self.config.num_inference_steps

        text_embeds = denoiser.preprocess(prompts, mode = "embeds", device, 1, do_classifier_free_guidance=guidance_scale > 1.0).detach()

        scheduler.set_timesteps(num_inference_steps, device = device)
        latents = torch.randn(len(prompts), *denoiser.get_input_shape(), device = device)

        all_step_preds, all_log_probs = [latents], []
        total_loss = 0.

        for i, t in enumerate(tqdm(scheduler.timesteps, disable=not show_progress)):
            if compute_loss:
                clipped_advantages = torch.clip(advantages, -method_config.clip_advantages, method_config.clip_advantages).detach()
            
            input = torch.cat([old_preds[i].detach() if compute_loss else latents] * 2)
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
            scheduler_output = scheduler.step(pred, t, old_preds[i].detach() if compute_loss else latents, eta, variance_noise=0)
            t_1 = t - scheduler.config.num_train_timesteps // num_inference_steps

            variance = scheduler._get_variance(t, t_1)
            std_dev_t = eta * variance ** 0.5
            prev_sample_mean = scheduler_output.prev_sample
            prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

            # DDPO specific code
            log_probs = self.calc_log_probs(
                old_preds[i+1].detach() if compute_loss else prev_sample,
                prev_sample_mean,
                std_dev_t
            )

            if compute_loss:
                ratio = torch.exp(log_probs - old_log_probs[i].detach())
                surr1 = -clipped_advantages * ratio
                surr2 = -clipped_advantages *  torch.clip(ratio, 1. - method_config.clip_ratio, 1. + method_config.clip_ratio)
                loss = torch.max(surr1, surr2).mean()
                loss.backward()
                total_loss += loss.item()
            else:
                all_step_preds.append(prev_sample)
                all_log_probs.append(log_probs)
                latents = prev_sample

        if compute_loss:
            return total_loss
        else:
            return latents, all_step_preds, all_log_probs





        



        