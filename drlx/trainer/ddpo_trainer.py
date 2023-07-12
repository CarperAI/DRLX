from accelerate import Accelerator
from collections import deque
from drlx.data.configs import DRLXConfig, DDPOConfig, ModelConfig, OptimizerConfig, SchedulerConfig, PromptPipelineConfig, RewardModelConfig
from drlx.trainer import BaseTrainer
from drlx.pipeline.prompt_pipeline import PromptPipeline
from drlx.denoisers import LDMUNet, BaseConditionalDenoiser
from drlx.reward_modelling import RewardModel
from drlx.sampling import DDPOSampler
from drlx.utils import get_prompt_pipeline_class, get_optimizer_class, get_scheduler_class, get_diffusion_pipeline_class

import torch
import tqdm
import numpy as np

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
class DDPOTrainer(BaseTrainer):
    def __init__(self, config: DDPOConfig, reward_fn=None, **kwargs):
        """ 
        DDPO Accelerate Trainer initialization
        
        Args:
            config (DDPOConfig): Configuration for DDPO Trainer

        """
        
        super().__init__(**kwargs)
        self.accelerator = Accelerator() # TODO: Add accelerator args


        self.model = self.setup_model()
        self.reward_fn = reward_fn
        self.optimizer = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()
        self.prompt_pipeline = self.setup_pipeline()

        self.dataloader = self.prompt_pipeline.create_loader(self.config.train.batch_size)

        self.model, self.reward_model, self.optimizer, self.scheduler, self.dataloader = self.accelerator.prepare(
            self.model, self.reward_model, self.optimizer, self.scheduler, self.dataloader
        )


    
    def setup_model(self):
        model = self.get_arch(self.config)(self.config.model, sampler = DDPOSampler())
        if self.config.model_path is not None:
            model.from_pretrained_pipeline(self.config.pipeline, self.config.model_path)
        return model


    def setup_reward_model(self):
        model = RewardModel(self.config.reward_model)
        return model
    
    def setup_optimizer(self):
        """
        Returns an optimizer derived from an instance's config
        """
        optimizer_class = get_optimizer_class(self.config.optimizer.name)
        optimizer = optimizer_class(
            self.model.parameters(),
            **self.config.optimizer.kwargs,
        )
        return optimizer

    def setup_scheduler(self):
        """
        Returns a learning rate scheduler derived from an instance's config
        """
        scheduler_class = get_scheduler_class(self.config.scheduler.name)
        scheduler = scheduler_class(self.opt, **self.config.scheduler.kwargs)
        return scheduler

    def setup_pipeline(self):
        """
        Returns a prompt pipeline derived from an instance's config
        """
        pipeline = get_prompt_pipeline_class(self.config.pipeline.name)
        return pipeline


    def get_arch(self, config):
        model_name = LDMUNet # nothing else is supported for now (TODO: add support for other models)
        return model_name


    def loss(self, x_t, log_probs_t, advantages, prompts):
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
        prompts = prompts.to(self.accelerator.device)


        scheduler = self.model.scheduler
        unet = self.model.unet
        text_embeddings = self.model._encode_prompt(prompts,self.accelerator.device, 1, do_classifier_free_guidance=self.config.sampler.guidance_scale > 1.0).detach()
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
            current_log_probs = self.model.sampler.calculate_log_probs(x_t[i+1].detach(), prev_sample_mean, std_dev_t).mean(dim=tuple(range(1, prev_sample_mean.ndim)))

            # calculate loss

            ratio = torch.exp(current_log_probs - log_probs_t[i].detach()) # this is the ratio of the new policy to the old policy
            unclipped_loss = -clipped_advantages * ratio # this is the surrogate loss
            clipped_loss = -clipped_advantages * torch.clip(ratio, 1. - self.config.method.clip_ratio, 1. + self.config.method.clip_ratio) # this is the surrogate loss, but with artificially clipped ratios
            loss = torch.max(unclipped_loss, clipped_loss).mean() # we take the max of the clipped and unclipped surrogate losses, and take the mean over the batch
            loss.backward() 
            loss_value += loss.item()

        return loss_value


    def sample_and_calculate_rewards(self, prompts):
        """
        Samples a batch of images and calculates the rewards for each image

        Args:
            prompts (list[str]): The prompts to sample with
        """
        preds, all_preds, log_probs = self.model.sampler.sample(prompts, self.model, device = self.accelerator.device)
        imgs = self.model.postprocess(preds)
        rewards = self.reward_model(imgs)
        return imgs, rewards, all_preds, log_probs
    
    def train(self, reward_fn, prompt_pipeline):
        """
        Trains the model based on config parameters
        """

        assert isinstance(self.model.sampler, DDPOSampler), "Error: Model Sampler for DDPO training must be DDPO sampler"

        if self.config.per_prompt_stat_tracking:
            per_prompt_stat_tracker = PerPromptStatTracker(self.config.per_prompt_stat_tracking.buffer_size, self.config.per_prompt_stat_tracking.min_count)
        
        mean_rewards = []
        for epoch in range(self.config.train.num_epochs):

            all_step_preds, log_probs, advantages, all_prompts, all_rewards = [], [], [], [], []

            # sampling `num_samples_per_epoch` images and calculating rewards
            for i, prompts in enumerate(tqdm(self.dataloader, disable=not self.accelerator.is_local_main_process)):
                batch_imgs, rewards, batch_all_step_preds, batch_log_probs = self.sample_and_calculate_rewards(prompts)
                batch_advantages = torch.from_numpy(per_prompt_stat_tracker.update(np.array(prompts), rewards.squeeze().cpu().detach().numpy())).float().to(self.accelerator.device)
                all_step_preds.append(batch_all_step_preds)
                log_probs.append(batch_log_probs)
                advantages.append(batch_advantages)
                all_prompts += prompts
                all_rewards.append(rewards)
            
            all_step_preds = torch.cat(all_step_preds, dim=1)
            log_probs = torch.cat(log_probs, dim=1)
            advantages = torch.cat(advantages)
            all_rewards = torch.cat(all_rewards)

            mean_rewards.append(all_rewards.mean().item())

            for inner_epoch in tqdm(range(self.config.method.num_inner_epochs), disable=not self.accelerator.is_local_main_process):

                # chunk them into batches (how does the official pytorch implementation do it?)
                all_step_preds_chunked = torch.chunk(all_step_preds, self.config.train.num_samples_per_epoch // self.config.train.batch_size, dim=1)
                log_probs_chunked = torch.chunk(log_probs, self.config.train.num_samples_per_epoch // self.config.train.batch_size, dim=1)
                advantages_chunked = torch.chunk(advantages, self.config.train.num_samples_per_epoch // self.config.train.batch_size, dim=0)
                
                # chunk the prompts (list of strings) into batches
                all_prompts_chunked = [all_prompts[i:i + self.config.train.batch_size] for i in range(0, len(all_prompts), self.config.train.batch_size)]
                
                for i in tqdm(range(len(all_step_preds_chunked)), disable=not self.accelerator.is_local_main_process):
                    self.optimizer.zero_grad()
                    loss = self.loss(all_step_preds_chunked[i], log_probs_chunked[i], advantages_chunked[i], all_prompts_chunked[i]) # loss.backward happens inside
                    torch.nn.utils.clip_grad_norm_(self.model.unet.parameters(), 1.0) # gradient clipping
                    self.optimizer.step()
                