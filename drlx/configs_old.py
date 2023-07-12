from typing import Dict, Any

from dataclasses import dataclass, asdict
import yaml

@dataclass
class ConfigClass:
    @staticmethod
    def from_dict(cls, cfg : Dict[str, Any]):
        return cls(**cfg)

    def to_dict(self):
        return asdict(self)
    
@dataclass
class ModelConfig(ConfigClass):
    ema_alpha : float = None # Doesn't use EMA if this is None
    vae_device : str = "cuda:0" # Device for VAE if LDM
    img_size : int = 512

@dataclass
class RewardModelConfig(ConfigClass):
    pass

@dataclass
class TrainConfig(ConfigClass):
    batch_size : int = 64
    epochs : int = 2
    
    #Optimizer
    learning_rate : float = 1e-4
    weight_decay : float = 1e-6

    #Scheduler
    rampup_length : int = 400
    rampdown_length : int = 1000
    final_learning_rate : float = 1e-6

    log_every : int = 10
    save_every : int = 10
    save_to : str = "checkpoints"
    load_from : str = None # If this is not none, a checkpoint is loaded
    eval_every : int = 10
    sample_every : int = 10
    
    grad_clip : float = 1.0

    # WANDB
    run_name : str = None
    wandb_entity : str = None
    wandb_project : str = None

@dataclass
class MethodConfig(ConfigClass):
    """
    Specific configs for different RL training methods.
    Implemented in same scripts as trainers
    """
    pass

@dataclass
class SamplerConfig(ConfigClass):
    mode : str = "v" # x, v, or eps
    guidance_scale : float = 5.0 # if guidance is being used
    sigma_data : float = 0.5 # Estimated sd for data
    num_inference_steps : int = 50
    eta : float = 1
    device : str = "cuda"
    postprocess : bool = False # If true, post processes latents to images (uint8 np arrays)


def load_yaml(yml_fp : str) -> Dict[str, ConfigClass]:
    with open(yml_fp, mode = 'r') as file:
        config = yaml.safe_load(file)
    d = {}
    if config["model"]:
        d["model"] = ModelConfig.from_dict(config["model"])
    if config["train"]:
        d["train"] = TrainConfig.from_dict(config["train"])
    if config["sampler"]:
        d["sampler"] = SamplerConfig.from_dict(config["sampler"])
    
    return d