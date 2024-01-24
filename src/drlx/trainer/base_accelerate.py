from drlx.trainer import BaseTrainer
from drlx.configs import DRLXConfig
from drlx.sampling import Sampler
from drlx.utils import suppress_warnings

from accelerate import Accelerator
import wandb
import logging
import torch
from diffusers import StableDiffusionPipeline


class AcceleratedTrainer(BaseTrainer):
    """
    Base class for any trainer using accelerate. Assumes model comes from a pretrained
    pipeline

    :param config: DRLX config. Method config can be anything.
    :type config: DRLXConfig
    """
    def __init__(self, config : DRLXConfig):
        super().__init__(config)
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
        model = self.get_arch(self.config)(self.config.model, sampler = Sampler(self.config.sampler))
        if self.config.model.model_path is not None:
            model, pipe = model.from_pretrained_pipeline(StableDiffusionPipeline, self.config.model.model_path)

        self.pipe = pipe            
        return model

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
            self.pipe.unet = unwrapped_model.unet
            self.pipe.save_pretrained(fp, safe_serialization = unwrapped_model.config.use_safetensors)
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
        self.accelerator.print("Succesfully loaded checkpoint")

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
            if self.config.model.lora_rank is not None:
                unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_model.unet))
                StableDiffusionPipeline.save_lora_weights(fp, unet_lora_layers=unet_lora_state_dict, safe_serialization = unwrapped_model.config.use_safetensors)
            else:
                self.pipe.unet = unwrapped_model.unet
                self.pipe.save_pretrained(fp, safe_serialization = unwrapped_model.config.use_safetensors)
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
        self.accelerator.print("Succesfully loaded checkpoint")
