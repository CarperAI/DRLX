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
class SamplerConfig(ConfigClass):
    mode : str = "v" # x, v, or eps
    guidance_scale : float = None # if guidance is being used
    sigma_data : float = 0.5 # Estimated sd for data


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