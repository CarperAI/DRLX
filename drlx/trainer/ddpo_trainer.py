from torchtyping import TensorType
from typing import Iterable

from accelerate import Accelerator
from collections import deque
from drlx.configs import DRLXConfig, DDPOConfig
from drlx.trainer import BaseTrainer
from drlx.sampling import DDPOSampler

import torch
import einops as eo
import os
import logging
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import wandb
from PIL import Image

from diffusers import StableDiffusionPipeline

class PerPromptStatTracker:
    def __init__(self, buffer_size, min_count):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = {}

    def update(self, prompts, rewards):
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = deque(maxlen=self.buffer_size)
            self.stats[prompt].extend(prompt_rewards)

            if len(self.stats[prompt]) < self.min_count:
                mean = np.mean(rewards)
                std = np.std(rewards) + 1e-6
            else:
                mean = np.mean(self.stats[prompt])
                std = np.std(self.stats[prompt]) + 1e-6
            advantages[prompts == prompt] = (prompt_rewards - mean) / std

        return advantages

class DDPOExperienceReplay(Dataset):
    def __init__(self,
        reward_fn, ppst: PerPromptStatTracker,
        imgs : Iterable[Iterable], prompts : Iterable[Iterable[str]], 
        all_step_preds : Iterable[TensorType["t","b","c","h","w"]],
        log_probs : Iterable[TensorType["t", "b"]],
        **dataloader_kwargs
    ):
        """
        Create dataloader to use for training given input samples.
        Inputs should still be in their original batches. Also returns
        rewards for logging purposes.
        """

        # Compute rewards first
        rewards = [reward_fn(img_batch, prompt_batch) 
                    for img_batch, prompt_batch in zip(imgs, prompts)]
        advantages = [torch.from_numpy(ppst.update(np.array(prompt_batch), reward_batch.squeeze().cpu().detach().numpy())).float() 
                        for prompt_batch, reward_batch in zip(prompts, rewards)]

        # Combine all_step_preds, log_probs, and advantages
        all_step_preds = torch.cat(all_step_preds, dim = 1) # [t, n, c, h, w]
        log_probs = torch.cat(log_probs, dim = 1) # [t, n]
        advantages = torch.cat(advantages)
        rewards = torch.cat(rewards)

        self.all_step_preds = all_step_preds
        self.log_probs = log_probs
        self.advantages = advantages
        self.rewards = rewards
        self.prompts = [prompt for prompt_list in prompts for prompt in prompt_list]
        
    def __getitem__(self, i):
        return self.all_step_preds[:,i], self.log_probs[:,i], self.advantages[i], self.prompts[i]

    def __len__(self):
        return self.all_step_preds.size(1)

    def create_loader(self, **kwargs):
        def collate(batch):
            # unzip the batch
            all_steps, log_probs, advs, prompts = list(zip(*batch))
            all_steps = torch.stack(all_steps, dim = 1)
            log_probs = torch.stack(log_probs, dim = 1)
            advs = torch.stack(advs)
            prompts = list(prompts)

            return (all_steps, log_probs, advs, prompts)

        return DataLoader(self, collate_fn=collate, **kwargs)
    
class DDPOTrainer(BaseTrainer):
    def __init__(self, config : DRLXConfig):
        """ 
        DDPO Accelerate Trainer initialization
        """
        
        super().__init__(config)

        assert isinstance(self.config.method, DDPOConfig), "ERROR: Method config must be DDPO config"

        self.accelerator = Accelerator(log_with = config.logging.log_with) # TODO: Add accelerator args

        # Disable tokenizer warnings since they clutter the CLI
        logging.getLogger("pytorch_transformers.tokenization_utils").setLevel(logging.ERROR)

        self.model = self.setup_model()
        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()

        # Disentangle this for accelerated training to work
        self.sampler = self.model.sampler
        self.sampler.set_denoiser_fns(self.model) 

        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler
        )

        # Setup tracking

        tracker_kwargs = {}
        self.use_wandb = not (config.logging.wandb_project is None)
        if self.use_wandb:
            log = config.logging
            tracker_kwargs["wandb"] = {
                "name" : log.run_name,
                "entity" : log.wandb_entity,
                "mode" : "online"
            }

            self.accelerator.init_trackers(
                project_name = log.wandb_project,
                config = config.to_dict(),
                init_kwargs = tracker_kwargs
            )

        self.world_size = self.accelerator.state.num_processes

    def setup_model(self):
        model = self.get_arch(self.config)(self.config.model, sampler = DDPOSampler())
        if self.config.model.model_path is not None:
            model.from_pretrained_pipeline(StableDiffusionPipeline, self.config.model.model_path)
        return model

    def loss(self, x_t, log_probs_t, advantages, prompts):
        return self.sampler.sample(
            use_grad=True,
            prompts=prompts,
            denoiser=self.model,
            device=self.accelerator.device,
            advantages=advantages,
            old_preds=x_t,
            old_log_probs=log_probs_t,
            show_progress=self.accelerator.is_local_main_process,
            method_config=self.config.method,
            accelerator=self.accelerator
        )

    # TODO: Remove if we can verify above function works
    def _loss(self, x_t, log_probs_t, advantages, prompts):
        """
        Computes the loss for a batch and returns a tuple containing the loss and
        a dictionary of metrics to log.

        Args:
            x_t (torch.Tensor): The batch of intermediate images
            log_probs_t (torch.Tensor): The log probabilities of the intermediate images
            advantages (torch.Tensor): The advantages of the final images
            prompts (list[str]): The prompts used to generate the batch

        """

        x_t = x_t.to(self.accelerator.device)
        log_probs_t = log_probs_t.to(self.accelerator.device)
        advantages = advantages.to(self.accelerator.device)

        scheduler = self.model.scheduler
        unet = self.model.unet
        text_embeddings = self.model.preprocess(prompts, mode = "embeds", device = self.accelerator.device, num_images_per_prompt= 1, do_classifier_free_guidance=self.config.sampler.guidance_scale > 1.0).detach()
        scheduler.set_timesteps(self.config.sampler.num_inference_steps, device=self.accelerator.device)
        loss_value = 0.
        for i, t in enumerate(tqdm(self.model.scheduler.timesteps, disable=not self.accelerator.is_local_main_process)): # note that we need to redo the sampling stuff since before with no_grad (may be possible to refactor)
            clipped_advantages = torch.clip(advantages, -self.config.method.clip_advantages, self.config.method.clip_advantages).detach()
        
            input = torch.cat([x_t[i].detach()] * 2)
            input = scheduler.scale_model_input(input, t)

            # predict the noise residual
            pred = unet(input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            pred_uncond, pred_text = pred.chunk(2)
            pred = pred_uncond + self.config.sampler.guidance_scale * (pred_text - pred_uncond)

            # compute the "previous" noisy sample mean and variance, and get log probs
            scheduler_output = scheduler.step(pred, t, x_t[i].detach(), self.config.sampler.eta, variance_noise=0)
            t_1 = t - scheduler.config.num_train_timesteps // self.config.sampler.num_inference_steps
            variance = scheduler._get_variance(t, t_1)
            std_dev_t = self.config.sampler.eta * variance ** (0.5)
            prev_sample_mean = scheduler_output.prev_sample
            current_log_probs = self.sampler.calc_log_probs(x_t[i+1].detach(), prev_sample_mean, std_dev_t)

            # calculate loss

            ratio = torch.exp(current_log_probs - log_probs_t[i].detach()) # this is the ratio of the new policy to the old policy
            unclipped_loss = -clipped_advantages * ratio # this is the surrogate loss
            clipped_loss = -clipped_advantages * torch.clip(ratio, 1. - self.config.method.clip_ratio, 1. + self.config.method.clip_ratio) # this is the surrogate loss, but with artificially clipped ratios
            loss = torch.max(unclipped_loss, clipped_loss).mean() # we take the max of the clipped and unclipped surrogate losses, and take the mean over the batch
            self.accelerator.backward(loss)
            loss_value += loss.item()

        return loss_value
    
    def sample(self, prompts):
        preds, all_preds, log_probs = self.sampler.sample(
            prompts = prompts,
            denoiser = self.model,
            device = self.accelerator.device
        )

        return preds, all_preds, log_probs

    def sample_and_calculate_rewards(self, prompts, reward_fn):
        """
        Samples a batch of images and calculates the rewards for each image

        Args:
            prompts (list[str]): The prompts to sample with
        """

        preds, all_preds, log_probs = self.sample(prompts)
        imgs = self.model.postprocess(preds)

        rewards = reward_fn(imgs, prompts).to(self.accelerator.device)
        return imgs, rewards, all_preds, log_probs
    
    def train(self, prompt_pipeline, reward_fn):
        """
        Trains the model based on config parameters. Needs to be passed a prompt pipeline and reward function.

        :param prompt_pipeline: Pipeline to draw text prompts from. Should be composed of just strings.
        :type prompt_pipeline: PromptPipeline

        :param reward_fn: Any function that returns a tensor of scalar rewards given np array of images (uint8) and text prompts (strings).
        It is fine to have a reward function that only rewards images without looking at prompts, simply add prompts as a dummy input.
        :type reward_fn: Callable[[np.array, Iterable[str], torch.Tensor]
        """

        # === SETUP ===

        dataloader = self.accelerator.prepare(
            prompt_pipeline.create_train_loader(batch_size = self.config.train.batch_size, shuffle = False)
        )

        assert isinstance(self.sampler, DDPOSampler), "Error: Model Sampler for DDPO training must be DDPO sampler"

        per_prompt_stat_tracker = PerPromptStatTracker(self.config.method.buffer_size, self.config.method.min_count)

        if isinstance(reward_fn, torch.nn.Module):
            reward_fn = self.accelerator.prepare(reward_fn)

        sample_prompts = next(iter(dataloader))

        # Set the epoch count
        outer_epochs = self.config.train.num_epochs
        if self.config.train.total_samples is not None:
            outer_epochs = self.config.train.total_samples // num_samples_per_epoch

        # ==========

        mean_rewards = []
        accum = 0
        for epoch in range(outer_epochs):

            preds, all_step_preds, log_probs, all_prompts = [], [], [], []

            # Make a new dataloader to reshuffle data
            dataloader = self.accelerator.prepare(
                prompt_pipeline.create_train_loader(batch_size = self.config.train.sample_batch_size, shuffle = True)
            )

            # Sample (play the game)
            data_steps = self.config.train.num_samples_per_epoch // self.config.train.sample_batch_size // self.world_size
            for i, prompts in enumerate(tqdm(dataloader, total = data_steps, disable=not self.accelerator.is_local_main_process)):            
                batch_preds, batch_all_step_preds, batch_log_probs = self.sample(prompts)
                
                preds.append(batch_preds)
                all_step_preds.append(batch_all_step_preds)
                log_probs.append(batch_log_probs)
                all_prompts.append(prompts)
            
                if i + 1 >= data_steps:
                    break

            # Get rewards from experiences
            self.accelerator.wait_for_everyone()
            self.model = self.accelerator.unwrap_model(self.model)
            imgs = [self.model.postprocess(pred) for pred in preds]

            exp_replay = DDPOExperienceReplay(
                reward_fn, per_prompt_stat_tracker,
                imgs, all_prompts,
                all_step_preds, log_probs
            )
            all_rewards = exp_replay.rewards
            experience_loader = exp_replay.create_loader(batch_size = self.config.train.batch_size)

            mean_rewards.append(all_rewards.mean().item())

            # Consistent prompt sample for logging
            sample_imgs, sample_rewards, _, _ = self.sample_and_calculate_rewards(sample_prompts, reward_fn)
            sample_imgs = [wandb.Image(Image.fromarray(img), caption=prompt + f', {reward.item()}') for img, prompt, reward in zip(sample_imgs, sample_prompts, sample_rewards)]
            batch_imgs = [wandb.Image(Image.fromarray(img), caption=prompt) for img, prompt in zip(imgs[-1], all_prompts[-1])]

            # Logging
            if self.use_wandb:
                self.accelerator.log({
                    "mean_reward" : all_rewards.mean().item(),
                    "reward_hist" : wandb.Histogram(all_rewards.detach().cpu().numpy()),
                    "img_batch" : batch_imgs,
                    "img_sample" : sample_imgs
                })

            self.model, experience_loader = self.accelerator.prepare(self.model, experience_loader)
            for inner_epoch in tqdm(range(self.config.method.num_inner_epochs), disable=not self.accelerator.is_local_main_process):
                for (all_step_preds, log_probs, advantages, prompts) in tqdm(experience_loader, disable=not self.accelerator.is_local_main_process):
                    self.optimizer.zero_grad()
                    loss = self.loss(all_step_preds, log_probs, advantages, prompts)
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
                    self.optimizer.step()
                    self.scheduler.step()

                    accum += 1
                    if accum % self.config.train.checkpoint_interval == 0 \
                    and self.config.train.checkpoint_interval > 0:
                        base_path = f"checkpoints/{self.config.logging.run_name}"
                        if not os.path.exists(base_path):
                            os.makedirs(base_path)
                        self.save_checkpoint(f"{base_path}/{accum}")


    def save_checkpoint(self, fp, components = None):
        self.accelerator.save_state(output_dir=fp)

    def load_checkpoint(self, fp):
        self.accelerator.load_state(fp)