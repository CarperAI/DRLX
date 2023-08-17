from typing import Union, Iterable, Tuple, Any, Optional
from torchtyping import TensorType

import torch
from tqdm import tqdm
import math
import einops as eo

from drlx.utils import rescale_noise_cfg

from drlx.configs import SamplerConfig, DDPOConfig

# Credit to Tanishq Abraham (tmabraham) for notebook from which
# both sampler and ddpo sampler code was adapted

# TODO: Normal sampler doesn't work with accelerate
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

class DDPOSampler(Sampler):
    def step_and_logprobs(self,
        scheduler,
        pred : TensorType["b", "c", "h", "w"],
        t : float,
        latents : TensorType["b", "c", "h", "w"],
        old_pred : Optional[TensorType["b", "c", "h", "w"]] = None
    ):
        """
        Steps backwards using scheduler. Considers the prediction as an action sampled
        from a normal distribution and returns average log probability for that prediction.
        Can also be used to find probability of current model giving some other prediction (old_pred)

        :param scheduler: Scheduler being used for diffusion process
        :param pred: Denoiser prediction with CFG and scaling accounted for
        :param t: Timestep in diffusion process
        :param latents: Latent vector given as input to denoiser
        :param old_pred: Alternate prediction. If given, computes log probability of current model predicting alternative output.
        """
        scheduler_out = scheduler.step(pred, t, latents, self.config.eta, variance_noise=0)
        
        # computing log_probs
        t_1 = t - scheduler.config.num_train_timesteps // self.config.num_inference_steps
        variance = scheduler._get_variance(t, t_1)
        std_dev_t = self.config.eta * variance ** 0.5
        prev_sample_mean = scheduler_out.prev_sample
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

        std_dev_t = torch.clip(std_dev_t, 1e-6) # force sigma > 1e-6

        # If old_pred provided, we are finding probability of new model outputting same action as before
        # Otherwise finding probability of current action
        action = old_pred if old_pred is not None else prev_sample # Log prob of new model giving old output
        log_probs = -((action.detach() - prev_sample_mean) ** 2) / (2 * std_dev_t ** 2) - torch.log(std_dev_t) - math.log(math.sqrt(2 * math.pi))
        log_probs = eo.reduce(log_probs, 'b c h w -> b', 'mean')

        return prev_sample, log_probs

    @torch.no_grad()
    def sample(
        self, prompts, denoiser, device,
        show_progress : bool = False,
        accelerator = None
    ) -> Iterable[torch.Tensor]:
        """
        DDPO sampling is analagous to playing a game in an RL environment. This function samples
        given denoiser and prompts but in addition to giving latents also gives log probabilities
        for predictions as well as ALL predictions (i.e. at each timestep)

        :param prompts: Text prompts to condition denoiser
        :param denoiser: Denoising model
        :param device: Device to do inference on
        :param show_progress: Display progress bar?
        :param accelerator: Accelerator object for accelerated training (optional)
        
        :return: triple of final denoised latents, all model predictions,  all log probabilities for each prediction
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

        all_step_preds, all_log_probs = [latents], []

        for t in tqdm(scheduler.timesteps, disable = not show_progress):
            latent_input = torch.cat([latents] * 2)  
            latent_input = scheduler.scale_model_input(latent_input, t)

            pred = denoiser(
                pixel_values = latent_input,
                time_step = t,
                text_embeds = text_embeds
            )

            # cfg
            pred = self.cfg_rescale(pred)

            # step
            prev_sample, log_probs = self.step_and_logprobs(scheduler, pred, t, latents)

            all_step_preds.append(prev_sample)
            all_log_probs.append(log_probs)
            latents = prev_sample
        
        return latents, torch.stack(all_step_preds), torch.stack(all_log_probs)
    
    def compute_loss(
        self, prompts, denoiser, device,
        show_progress : bool = False,
        advantages = None, old_preds = None, old_log_probs = None,
        method_config : DDPOConfig = None,
        accelerator = None
    ):


        """
        Computes the loss for the DDPO sampling process. This function is used to train the denoiser model.

        :param prompts: Text prompts to condition the denoiser
        :param denoiser: Denoising model
        :param device: Device to perform model inference on
        :param show_progress: Whether to display a progress bar for the sampling steps
        :param advantages: Normalized advantages obtained from reward computation
        :param old_preds: Previous predictions from past model
        :param old_log_probs: Log probabilities of predictions from past model
        :param method_config: Configuration for the DDPO method
        :param accelerator: Accelerator object for accelerated training (optional)

        :return: Total loss computed over the sampling process
        """


        if accelerator is None:
            denoiser_unwrapped = denoiser
        else:
            denoiser_unwrapped = accelerator.unwrap_model(denoiser)

        scheduler = denoiser_unwrapped.scheduler
        preprocess = denoiser_unwrapped.preprocess

        adv_clip = method_config.clip_advantages # clip value for advantages
        pi_clip = method_config.clip_ratio # clip value for policy ratio

        text_embeds = preprocess(
            prompts, mode = "embeds", device = device,
            num_images_per_prompt = 1,
            do_classifier_free_guidance = self.config.guidance_scale > 1.0
        ).detach()

        scheduler.set_timesteps(self.config.num_inference_steps, device = device)
        total_loss = 0.

        for i, t in enumerate(tqdm(scheduler.timesteps, disable = not show_progress)):
            latent_input = torch.cat([old_preds[i].detach()] * 2)
            latent_input = scheduler.scale_model_input(latent_input, t)

            pred = denoiser(
                pixel_values = latent_input,
                time_step = t,
                text_embeds = text_embeds
            )

            # cfg
            pred = self.cfg_rescale(pred)

            # step 
            prev_sample, log_probs = self.step_and_logprobs(
                scheduler, pred, t, old_preds[i],
                old_preds[i+1]
            )

            # Need to be computed and detached again because of autograd weirdness
            clipped_advs = torch.clip(advantages,-adv_clip,adv_clip).detach()

            # ppo actor loss
            
            ratio = torch.exp(log_probs - old_log_probs[i].detach())
            surr1 = -clipped_advs * ratio
            surr2 = -clipped_advs * torch.clip(ratio, 1. - pi_clip, 1. + pi_clip)
            loss = torch.max(surr1, surr2).mean()
            if accelerator is not None:
                accelerator.backward(loss)
            else:
                loss.backward()
            
            total_loss += loss.item()
        
        return total_loss