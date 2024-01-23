from torchtyping import TensorType
from typing import Iterable, Tuple, Callable

from accelerate import Accelerator
from drlx.configs import DRLXConfig, DPOConfig
from drlx.trainer.base_accelerate import AcceleratedTrainer
from drlx.sampling import DPOSampler
from drlx.utils import suppress_warnings, Timer, scoped_seed, save_images

import torch
import einops as eo
import os
import gc
import logging
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import wandb
import accelerate.utils
from PIL import Image

from diffusers import StableDiffusionPipeline

class DPOTrainer(AcceleratedTrainer):
    """ 
    DDPO Accelerated Trainer initilization from config. During init, sets up model, optimizer, sampler and logging

    :param config: DRLX config
    :type config: DRLXConfig
    """

    def __init__(self, config : DRLXConfig):    
        super().__init__(config)

        assert isinstance(self.config.method, DPOConfig), "ERROR: Method config must be DPO config"

    def setup_model(self):
        """
        Set up model from config.
        """
        model = self.get_arch(self.config)(self.config.model, sampler = DPOSampler(self.config.sampler))
        if self.config.model.model_path is not None:
            model, pipe = model.from_pretrained_pipeline(StableDiffusionPipeline, self.config.model.model_path)

        self.pipe = pipe          
        return model

    def loss(
        self,
        prompts, chosen_img, rejected_img, ref_denoiser
    ):
        """
        Get loss for training

        :param chosen_batch_preds: Predictions for the ba
        """
        return self.sampler.compute_loss(
            prompts=prompts, chosen_img=chosen_img, rejected_img=rejected_img,
            denoiser=self.model, ref_denoiser=ref_denoiser, vae=self.model.vae
            device=self.accelerator.device,
            method_config=self.config.method,
            accelerator=self.accelerator
        )
    
    @torch.no_grad()
    def deterministic_sample(self, prompts):
        """
        Sample images deterministically. Utility for visualizing changes for fixed prompts through training.
        """
        gen = torch.Generator(device=self.pipe.device).manual_seed(self.config.train.seed)
        self.pipe.unet = self.model.unet
        return self.pipe(prompts, generator = gen).images

    def train(self, pipeline):
        """
        Trains the model based on config parameters. Needs to be passed a prompt pipeline and reward function.

        :param pipeline: Pipeline to draw tuples from with prompts
        :type prompt_pipeline: DPOPipeline
        """

        # === SETUP ===

        # Singular dataloader made to get a sample of prompts
        # This sample batch is dependent on config seed so it can be same across runs
        with scoped_seed(self.config.train.seed):
            dataloader = self.accelerator.prepare(
                pipeline.create_loader(batch_size = self.config.train.batch_size, shuffle = False)
            )
            sample_prompts = self.config.train.sample_prompts
            if sample_prompts is None:
                sample_prompts = []
            if len(sample_prompts) < self.config.train.batch_size:
                new_sample_prompts = next(iter(dataloader))["prompts"]
                sample_prompts += new_sample_prompts
                sample_prompts = sample_prompts[:self.config.train.batch_size]

        # Now make main dataloader

        assert isinstance(self.sampler, DPOSampler), "Error: Model Sampler for DPO training must be DPO sampler"

        if isinstance(reward_fn, torch.nn.Module):
            reward_fn = self.accelerator.prepare(reward_fn)

        # Set the epoch count
        epochs = self.config.train.num_epochs
        if self.config.train.total_samples is not None:
            epochs = int(self.config.train.total_samples // self.config.train.num_samples_per_epoch)

        # Timer to measure time per 1k images (as metric)
        timer = Timer()
        def time_per_1k(n_samples : int):
            total_time = timer.hit()
            return total_time * 1000 / n_samples
        last_batch_time = timer.hit()
        
        # Ref model
        ref_model = self.setup_model()

        # === MAIN TRAINING LOOP ===

        mean_rewards = []
        accum = 0
        last_epoch_time = timer.hit()
        for epoch in range(epochs):
            dataloader = pipeline.create_loader(batch_size = self.config.train.batch_size, shuffle = True)
            dataloader = self.accelerator.prepare(dataloader)

            # Clean up unused resources
            self.accelerator._dataloaders = [] # Clear dataloaders
            gc.collect()
            torch.cuda.empty_cache()

            self.accelerator.print(f"Epoch {epoch}/{epochs}.")

            for batch in dataloader:
                metrics = self.compute_loss(
                    prompts = batch['prompts'],
                    chosen_img = batch['chosen_pixel_values'],
                    rejected_img = batch['rejected_pixel_values'],
                    ref_model
                )

                self.accelerator.wait_for_everyone()
                # Generate the sample prompts
                self.pipe.unet = self.model.unet
                with torch.no_grad():
                    with scoped_seed(self.config.train.seed):
                        sample_imgs = self.deterministic_sample(sample_prompts)
                        sample_imgs = [wandb.Image(img, caption = prompt) for (img, prompt) in zip(sample_imgs, sample_prompts)]

                # Logging
                if self.use_wandb:
                    self.accelerator.log({
                        "base_loss" : metrics["diffusion_loss"],
                        "accuracy" : metrics["accuracy"],
                        "dpo_loss" : metrics["loss"],
                        "time_per_1k" : last_batch_time,
                        "img_sample" : sample_imgs
                    })
                # save images
                if self.accelerator.is_main_process and self.config.train.save_samples:
                    save_images(sample_imgs, f"./samples/{self.config.logging.run_name}/{epoch}")

            

                # Save model every [interval] epochs
                accum += 1
                if accum % self.config.train.checkpoint_interval == 0 and self.config.train.checkpoint_interval > 0:
                    self.accelerator.print("Saving...")
                    base_path = f"./checkpoints/{self.config.logging.run_name}"
                    output_path = f"./output/{self.config.logging.run_name}"
                    self.accelerator.wait_for_everyone()
                    self.save_checkpoint(f"{base_path}/{accum}")
                    self.save_pretrained(output_path)

                last_epoch_time = time_per_1k(self.config.train.num_samples_per_epoch)
            
                del metrics
            del dataloader

