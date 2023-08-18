from torchtyping import TensorType
from typing import Iterable, Tuple, Callable

from accelerate import Accelerator
from drlx.configs import DRLXConfig, DDPOConfig
from drlx.trainer import BaseTrainer
from drlx.sampling import DDPOSampler
from drlx.utils import suppress_warnings, Timer, PerPromptStatTracker

import torch
import einops as eo
import os
import gc
import logging
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import wandb
from PIL import Image

from diffusers import StableDiffusionPipeline

class DDPOExperienceReplay(Dataset):
    """
    Utility class to compute advantages and create dataloader from sampling experiences.
    """

    def __init__(self,
        reward_fn: callable, 
        ppst: PerPromptStatTracker,
        imgs : Iterable[Iterable], 
        prompts : Iterable[Iterable[str]], 
        all_step_preds : Iterable[TensorType["t","b","c","h","w"]],
        log_probs : Iterable[TensorType["t", "b"]],
        **dataloader_kwargs
    ):
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

        # Prompts is list of batches of prompts (list of list of strings)
        # Iterate through each batch and each prompt within it to unwrap into single list of prompts
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
    """ 
    DDPO Accelerated Trainer initilization from config. During init, sets up model, optimizer, sampler and logging

    :param config: DRLX config
    :type config: DRLXConfig
    """

    def __init__(self, config : DRLXConfig):    
        super().__init__(config)

        assert isinstance(self.config.method, DDPOConfig), "ERROR: Method config must be DDPO config"

        # Figure out batch size and accumulation steps
        if self.config.train.target_batch is not None: # Just use normal batch_size
            self.accum_steps = (self.config.train.target_batch // self.config.train.batch_size)
        else:
            self.accum_steps = 1

        self.accelerator = Accelerator(
            log_with = config.logging.log_with,
            gradient_accumulation_steps = self.accum_steps
        )

        # Disable tokenizer warnings since they clutter the CLI
        kw_str = self.config.train.suppress_log_keywords
        if kw_str is not None:
            for prefix in kw_str.split(","):
                suppress_warnings(prefix.strip())

        self.pipe = None # Store reference to pipeline so that we can use save_pretrained later
        self.model = self.setup_model()
        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()

        self.sampler = self.model.sampler
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
        """
        Set up model from config.
        """
        model = self.get_arch(self.config)(self.config.model, sampler = DDPOSampler(self.config.sampler))
        if self.config.model.model_path is not None:
            model, pipe = model.from_pretrained_pipeline(StableDiffusionPipeline, self.config.model.model_path)

        self.pipe = pipe            
        return model

    def loss(
        self,
        x_t : TensorType["timesteps", "batch", "channels", "height", "width"],
        log_probs_t : TensorType["timesteps", "batch"], 
        advantages : TensorType["batch"],
        prompts : Iterable[str]
    ):
        """
        Get loss for training

        :param x_t: Samples across time steps and across batch
        :type x_t: torch.Tensor

        :param log_probs_t: Log probabilities for each sample prediction
        :type log_probs_t: torch.Tensor

        :advantages: Advantages associated with each image across batch
        :type advantages: torch.Tensor

        :prompts: Prompts used for generation across the batch
        :type prompts: Iterable[str]

        :return: loss
        :rtype: torch.Tensor
        """
        return self.sampler.compute_loss(
            prompts=prompts,
            denoiser=self.model,
            device=self.accelerator.device,
            advantages=advantages,
            old_preds=x_t,
            old_log_probs=log_probs_t,
            show_progress=self.accelerator.is_main_process,
            method_config=self.config.method,
            accelerator=self.accelerator
        )
    
    def sample(self, prompts : Iterable[str]) -> Tuple[torch.Tensor]:
        """
        Sample predictions, predictions at time steps and log probabilities from sampler

        :param prompts: Batched prompts to use for sampling
        :type prompts: Iterable[str]

        :return: 3 Tensors: final predictions for latent, all step predictions during denoising process, and log probabilities for each prediction
        :rtype: Tuple[torch.Tensor]
        """
        preds, all_preds, log_probs = self.sampler.sample(
            prompts = prompts,
            denoiser = self.model,
            device = self.accelerator.device,
            accelerator = self.accelerator
        )

        return preds, all_preds, log_probs

    def sample_and_calculate_rewards(self, prompts : Iterable[str], reward_fn : Callable) -> Tuple:
        """
        Samples a batch of images and calculates the rewards for each image

        :param prompts: Batch of prompts to sample with
        :type prompts: Iterable[str]

        :param reward_fn: Function to be called on final images and prompts to be used for reward computation
        :type reward_fn: Callable[[np.ndarray, Iterable[str]], Iterable[float]]

        :return: Final images, rewards, all step predictions, log probabilities for predictions
        :rtype: Tuple
        """

        preds, all_preds, log_probs = self.sample(prompts)
        imgs = self.accelerator.unwrap_model(self.model).postprocess(preds)

        rewards = reward_fn(imgs, prompts).to(self.accelerator.device)
        return imgs, rewards, all_preds, log_probs
    
    def print_in_main(self, txt):
        """
        Utility function to use with accelerate so that messages are only printed once in main process
        """
        if self.accelerator.is_main_process:
            print(txt)
    
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
            outer_epochs = int(self.config.train.total_samples // self.config.train.num_samples_per_epoch)

        # Timer to measure time per 1k images (as metric)
        timer = Timer()
        def time_per_1k(n_samples : int):
            total_time = timer.hit()
            return total_time * 1000 / n_samples

        # === MAIN TRAINING LOOP ===

        mean_rewards = []
        accum = 0
        last_epoch_time = timer.hit()
        for epoch in range(outer_epochs):
            preds, all_step_preds, log_probs, all_prompts = [], [], [], []
            self.print_in_main(f"Epoch {epoch}/{outer_epochs}. {epoch * self.config.train.num_samples_per_epoch} samples seen. Averaging {last_epoch_time:.2f}s/1k samples.")
            # Make a new dataloader to reshuffle data
            dataloader = self.accelerator.prepare(
                prompt_pipeline.create_train_loader(batch_size = self.config.train.sample_batch_size, shuffle = True)
                )
            # Sample (play the game)
            data_steps = self.config.train.num_samples_per_epoch // self.config.train.sample_batch_size // self.world_size
            self.print_in_main("Sampling...")
            for i, prompts in enumerate(tqdm(dataloader, total = data_steps, disable=not self.accelerator.is_main_process)):            
                batch_preds, batch_all_step_preds, batch_log_probs = self.sample(prompts)
                
                preds.append(batch_preds)
                all_step_preds.append(batch_all_step_preds)
                log_probs.append(batch_log_probs)
                all_prompts.append(prompts)
            
                if i + 1 >= data_steps:
                    break

            # Get rewards from experiences
            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            imgs = [unwrapped_model.postprocess(pred) for pred in preds]

            # Experience replay computes normalized rewards,
            # then is used to create a loader for training
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
                    "time_per_1k" : last_epoch_time,
                    "img_batch" : batch_imgs,
                    "img_sample" : sample_imgs
                })

            # Inner epochs and actual training
            self.print_in_main("Training...")
            experience_loader = self.accelerator.prepare(experience_loader)
            # Inner epochs normally one, disable progress bar when this is the case 
            for inner_epoch in tqdm(range(self.config.method.num_inner_epochs),
                disable=(not self.accelerator.is_main_process) or (self.config.method.num_inner_epochs == 1)
            ):
                for (all_step_preds, log_probs, advantages, prompts) in tqdm(experience_loader, disable=not self.accelerator.is_main_process):
                    with self.accelerator.accumulate(self.model): # Accumulate across minibatches
                        loss = self.loss(all_step_preds, log_probs, advantages, prompts)
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        if self.use_wandb:
                            self.accelerator.log({ #TODO: add approx_kl tracking
                                "loss" : loss, 
                                "lr" : self.scheduler.get_last_lr()[0],
                                "epoch": epoch
                            })

            # Save model every [interval] epochs
            accum += 1
            if accum % self.config.train.checkpoint_interval == 0 and self.config.train.checkpoint_interval > 0:
                self.print_in_main("Saving...")
                base_path = f"./checkpoints/{self.config.logging.run_name}"
                output_path = f"./output/{self.config.logging.run_name}"
                self.accelerator.wait_for_everyone()
                self.save_checkpoint(f"{base_path}/{accum}")
                self.save_pretrained(output_path)

            last_epoch_time = time_per_1k(self.config.train.num_samples_per_epoch)
            
            del loss, experience_loader
            gc.collect()
            torch.cuda.empty_cache()

    def save_checkpoint(self, fp : str, components = None):
        """
        Save checkpoint in main process

        :param fp: File path to save checkpoint to
        """
        if self.accelerator.is_main_process:
            os.makedirs(fp, exist_ok = True)
            self.accelerator.save_state(output_dir=fp)
        self.accelerator.wait_for_everyone() # need to use this twice or a corrupted state is saved

    def save_pretrained(self, fp : str):
        """
        Save model into pretrained pipeline so it can be loaded in pipeline later

        :param fp: File path to save to
        """
        if self.accelerator.is_main_process:
            os.makedirs(fp, exist_ok = True)
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            #unwrapped_model.unet.save_pretrained(fp)
            self.pipe.unet = unwrapped_model.unet
            self.pipe.save_pretrained(fp)
        self.accelerator.wait_for_everyone()

    def extract_pipeline(self):
        """
        Return original pipeline with finetuned denoiser plugged in

        :return: Diffusers pipeline
        """

        self.pipe.unet = self.accelerator.unwrap_model(self.model).unet
        return self.pipe

    def load_checkpoint(self, fp : str):
        """
        Load checkpoint

        :param fp: File path to checkpoint to load from
        """
        self.accelerator.load_state(fp)
        self.print_in_main("Succesfully loaded checkpoint")