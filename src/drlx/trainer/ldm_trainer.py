from typing import Iterable

from accelerate import Accelerator

from PIL import Image
import torch
from torch.optim import AdamW
import torch.nn.functional as F
import os

from drlx.configs import TrainConfig
from drlx.trainer import BaseTrainer
from drlx.utils import get_latest_checkpoint

def prepare_accelerator(config : TrainConfig):
    """
    Prepares the accelerator for logging and returns whether or not wandb is being used
    """
    tracker = None
    logging_dir = None
    if config.wandb_project is not None:
        tracker = "wandb"
        logging_dir = config.wandb_project
    
    accelerator = Accelerator(log_with = tracker, logging_dir = logging_dir)

    if accelerator.state.deepspeed_plugin is not None:
        def set_ds_field(key, val):
            accelerator.state.deepspeed_plugin.deepspeed_config[key] = val
        
        set_ds_field("train_micro_batch_size_per_gpu", config.batch_size)
    
    tracker_kwargs = {}
    if config.wandb_project is not None:
        tracker_kwargs["wandb"] = {
            "name" : config.run_name,
            "entity" : config.wandb_entity,
            "mode" : "online"
        }
        accelerator.init_trackers(
            project_name = config.wandb_project,
            config = config.to_dict(),
            init_kwargs = tracker_kwargs
        )
    
    return accelerator, (tracker is not None)

class LDMTrainer(BaseTrainer):
    def train(self, pipeline, model, ae_model, reward_model = None):
        if reward_model is not None:
            raise ValueError("LDMTrainer does not support reward models")
        
        # Create accelerator and set variables accordingly
        accelerator, use_wandb = prepare_accelerator(self.config)

        # Construct preprocessing that can be applied to both the image and the text
        def prep(img_batch : Iterable[Image.Image], txt_batch : Iterable[str]):
            pixel_values = ae_model.preprocess(img_batch) 
            input_ids, attention_mask = model.preprocess(txt_batch)

            return pixel_values, input_ids, attention_mask

        pipeline.set_preprocess_fn(prep)
        train_dl = pipeline.create_train_loader(
           batch_size = self.config.batch_size 
        )

        optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # TODO add scheduler

        # TODO add EMA

        # Loading a previous checkpoint
        n_updates = 0
        if self.config.load_from:
            checkpoint_root = self.config.load_from
            load_dir = get_latest_checkpoint(checkpoint_dir)
            try:
                n_updates = int(os.path.basename(load_dir))
            except:
                load_dir = None
            if load_dir is not None:
                component_dict = self.load_checkpoint(load_dir)
                model.load_state_dict(component_dict["unet"])
                optimizer.load_state_dict(component_dict["optimizer"])
                # TODO: Add scheduler and ema model here

        model, ae_model, train_dl, optimizer = accelerator.prepare(model, ae_model, train_dl, optimizer)
        prediction_type = model.sampler.config

        for epoch in range(self.config.epochs):
            for batch in train_dl:
                optimizer.zero_grad()
                sample, tokens, mask = batch
                b = sample.shape[0]

                # Encode with the autoencoder
                with torch.no_grad():
                    sample = ae_model.encode(sample)
                
                # Generate noise and timesteps
                noise = torch.randn(sample.shape).to(sample.device)
                timesteps = torch.randint(
                    0, model.scheduler.config.num_train_timesteps, (b,), device = sample.device
                )

                noisy_sample = model.scheduler.add_noise(sample, noise, timesteps)

                with accelerator.accumulate(model):
                    model_out = model(noisy_sample, tokens, mask, timesteps)

                    if prediction_type == "eps":
                        loss = F.mse_loss(model_out, noise)
                    elif prediction_type == "sample":
                        alpha_t = 1
                        loss = snr_weight * F.mse_loss(model_out, sample, reduction = "none")
                        loss = loss.mean()

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                    
                    optimizer.step()
                    #scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients: # Whenever accelerator performs gradient step
                    n_updates += 1
                    accelerator.log({"train_loss" : loss})
                    # TODO: EMA update goes here
                    
                    # Save components
                    if n_updates % self.config.save_every == 0:
                        self.save_checkpoint(
                            fp=f"{self.config.save_to}",
                            components={
                                "unet" : model.state_dict(),
                                "optimizer" : optimizer.state_dict()
                            }, # TODO: Add scheduler and EMA
                            index=n_updates
                        )
                    







                
        

        



