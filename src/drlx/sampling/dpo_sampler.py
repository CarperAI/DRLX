import torch
import torch.nn.functional as F
import einops as eo

from drlx.sampling.base import Sampler
from drlx.configs import DPOConfig
from drlx.utils.sdxl import get_time_ids

class DPOSampler(Sampler):
    def compute_loss(
        self,
        prompts,
        chosen_img,
        rejected_img,
        denoiser,
        vae,
        device,
        method_config : DPOConfig,
        accelerator = None,
        ref_denoiser = None
    ):
        """
        Compute metrics and do backwards pass on loss. Assumes LoRA if reference is not given.
        """
        do_lora = ref_denoiser is None

        scheduler = accelerator.unwrap_model(denoiser).scheduler
        preprocess = accelerator.unwrap_model(denoiser).preprocess
        sdxl_flag = accelerator.unwrap_model(denoiser).sdxl_flag
        encode = accelerator.unwrap_model(vae).encode

        beta = method_config.beta
        ref_strategy = method_config.ref_mem_strategy

        # Text and image preprocessing
        with torch.no_grad():
            text_embeds = preprocess(
                prompts, mode = "embeds", device = device,
                num_images_per_prompt = 1,
                do_classifier_free_guidance = self.config.guidance_scale > 1.0
            )

            # The value returned above varies depending on model
            # With most models its two values, positive and negative prompts
            # With DPO we don't care about CFG, so just only get the positive prompts
            added_cond_kwargs = {}
            if sdxl_flag:
                added_cond_kwargs['text_embeds'] = text_embeds[2].detach() # Pooled prompt embeds
                added_cond_kwargs['time_ids'] = get_time_ids(chosen_img)

            text_embeds = text_embeds[0].detach()

            chosen_latent = encode(chosen_img).latent_dist.sample()
            rejected_latent = encode(rejected_img).latent_dist.sample()

        # sample random ts
        timesteps = torch.randint(
            0, self.config.num_inference_steps, (len(chosen_img),), device = device, dtype = torch.long
        )

        # One step of noising to samples
        noise = torch.randn_like(chosen_latent) # [B, C, H, W]

        # Doubling across chosen and rejeceted
        def double_up(x):
            return torch.cat([x,x], dim = 0)
        
        def double_down(x):
            n = len(x)
            return x[:n//2], x[n//2:]

        # Double everything up so we can input both chosen and rejected at the same time
        timesteps = double_up(timesteps)
        noise = double_up(noise)
        text_embeds = double_up(text_embeds)

        if sdxl_flag:
            added_cond_kwargs['text_embeds'] = double_up(added_cond_kwargs['text_embeds'])
            added_cond_kwargs['time_ids'] = double_up(added_cond_kwargs['time_ids'])

        latent = torch.cat([chosen_latent, rejected_latent])

        noisy_inputs = scheduler.add_noise(
            latent,
            noise,
            timesteps
        )

        # Get targets
        if scheduler.config.prediction_type == "epsilon":
            target = noise
        elif scheduler.config.prediction_type == "v_prediction":
            target = scheduler.get_velocity(
                latent,
                noise,
                timesteps
            )
        
        # utility function to get loss simpler
        def split_mse(pred, target):
            mse = eo.reduce(F.mse_loss(pred, target, reduction = 'none'), 'b ... -> b', reduction = "mean")
            chosen, rejected = double_down(mse)
            return chosen - rejected, mse.mean()

        # Forward pass and loss for DPO denoiser
        pred = denoiser(
            pixel_values = noisy_inputs,
            time_step = timesteps,
            text_embeds = text_embeds,
            added_cond_kwargs = added_cond_kwargs
        )
        model_diff, base_loss = split_mse(pred, target)

        # Forward pass and loss for refrence
        with torch.no_grad():
            if do_lora:
                accelerator.unwrap_model(denoiser).disable_adapters()

                ref_pred = denoiser(
                    pixel_values = noisy_inputs,
                    time_step = timesteps,
                    text_embeds = text_embeds,
                    added_cond_kwargs = added_cond_kwargs
                )
                ref_diff, ref_loss = split_mse(ref_pred, target)

                accelerator.unwrap_model(denoiser).enable_adapters()
            else:
                ref_inputs = {
                    "sample" : noisy_inputs.half() if ref_strategy == "half" else noisy_inputs,
                    "timestep" : timesteps,
                    "encoder_hidden_states" : text_embeds.half() if ref_strategy == "half" else text_embeds,
                    "added_cond_kwargs" : added_cond_kwargs
                }
                ref_pred = ref_denoiser(**ref_inputs).sample
                ref_diff, ref_loss = split_mse(ref_pred, target)
        
        # DPO Objective
        surr_loss = -beta * (model_diff - ref_diff)
        loss = -1 * F.logsigmoid(surr_loss.mean())

        # Get approx accuracy as models probability of giving chosen over rejected
        acc = (surr_loss > 0).sum().float() / len(surr_loss)
        acc += 0.5 * (surr_loss == 0).sum().float() / len(surr_loss) # 50% for when both match

        if accelerator is None:
            loss.backward()
        else:
            accelerator.backward(loss)

        return {
            "loss" : loss.item(),
            "diffusion_loss" : base_loss.item(),
            "accuracy" : acc.item(),
            "ref_deviation" : (ref_loss - base_loss) ** 2
        }