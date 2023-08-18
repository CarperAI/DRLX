import sys
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set
import yaml


@dataclass
class ConfigClass:
    @classmethod
    def from_dict(cls, cfg : Dict[str, Any]):
        return cls(**cfg)

    def to_dict(self):
        return asdict(self)
    

# specifies a dictionary of method configs
_METHODS: Dict[str, Any] = {}  # registry


def register_method(name):
    """Decorator used register a method config
    Args:
        name: Name of the method
    """

    def register_class(cls, name):
        _METHODS[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls

    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)

    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls

@dataclass
@register_method
class MethodConfig(ConfigClass):
    """
    Config for a certain RL method.

    :param name: Name of the method
    :type name: str
    """

    name: str = None

def get_method(name: str) -> MethodConfig:
    """
    Return constructor for specified method config
    """
    name = name.lower()
    if name in _METHODS:
        return _METHODS[name]
    else:
        raise Exception("Error: Trying to access a method that has not been registered")
   
@register_method("DDPO")
@dataclass
class DDPOConfig(MethodConfig):
    """
    Config for DDPO-related hyperparameters including per prompt stat tracker
    
    :param clip_advantages: Maximum absolute value of advantages
    :type clip_advantages: float

    :param clip_ratio: Maximum absolute value of ratio of new to old policy
    :type clip_ratio: float

    :param num_inner_epochs: Number of epochs to train the policy for
    :type num_inner_epochs: int

    :param buffer_size: Number of samples to keep in the buffer
    :type buffer_size: int

    :param min_count: Minimum number of samples to keep in the buffer before
        calculating statistics
    :type min_count: int
    """
    name : str = "DDPO"
    clip_advantages: float = 10.0
    clip_ratio: float = 1e-4
    num_inner_epochs: int = 1

    buffer_size: int = 32 # Set to None to avoid using per prompt stat tracker
    min_count: int = 16

@dataclass
class TrainConfig(ConfigClass):
    """
    Config for training

    :param batch_size: Batch size
    :type batch_size: int

    :param target_batch: Target batch size with gradient accumulation
    :type target_batch: int

    :param sample_batch_size: Batch size to use during inference only
    :type sample_batch_size: int

    :param num_epochs: Number of epochs to train for
    :type num_epochs: int

    :param total_samples: Provide this as alternative to epochs. Computes required epochs to see this may samples.
    :type total_samples: int

    :param num_samples_per_epoch: Number of samples to use per epoch
    :type num_samples_per_epoch: int

    :param grad_clip: Maximum absolute value of gradient
    :type grad_clip: float

    :param checkpoint_interval: Number of epochs between checkpoints
    :type checkpoint_interval: int

    :param checkpoint_path: Path to save checkpoints to
    :type checkpoint_path: str

    :param seed: Random seed
    :type seed: int

    :param tf32: Use tf32 precision
    :type tf32: bool

    :suppress_log_keywords: List of prefixes for loggers to suppress warnings from during training. Type as single string with different prefixes delimited by commas.
    :type suppress_log_keywords: str
    """
    batch_size: int = 4
    target_batch: int = None
    sample_batch_size: int = 8
    num_epochs: int = 50
    total_samples: int = None
    num_samples_per_epoch: int = 256
    grad_clip: float = 1.0
    checkpoint_interval: int = 10
    checkpoint_path: str = "checkpoints"
    seed: int = 0
    tf32: bool = False
    suppress_log_keywords: str = None


@dataclass
class LoggingConfig(ConfigClass):
    """
    Config for logging

    :param log_with: Logging backend to use (either "wandb" or "tensorboard")
    :type log_with: str

    :param run_name: Name of the run. Also used during saving.
    :type run_name: str

    :param wandb_entity: Name of the wandb entity to log to
    :type wandb_entity: str

    :param wandb_project: Name of the wandb project to log to
    :type wandb_project: str
    """
    log_with: str = "wandb" # "wandb" or "tensorboard"
    log_dir: str = None
    run_name: str = 'ddpo_exp'
    wandb_entity: str = None
    wandb_project: str = None



@dataclass
class OptimizerConfig(ConfigClass):
    """
    Config for an optimizer.

    :param name: Name of the optimizer
    :type name: str

    :param kwargs: Keyword arguments for the optimizer (e.g. lr, betas, eps, weight_decay)
    :type kwargs: Dict[str, Any]
    """

    name: str = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulerConfig(ConfigClass):
    """
    Config for a learning rate scheduler.

    :param name: Name of the scheduler
    :type name: str

    :param kwargs: Keyword arguments for the scheduler instance (e.g. warmup_steps, T_max)
    :type kwargs: Dict[str, Any]
    """

    name: str = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig(ConfigClass):
    """
    Config for a model.

    :param model_path: Path or name of the model (local or on huggingface hub)
    :type model_path: str

    :param model_arch_type: Type of model architecture. 
    :type model_arch_type: str

    :param attention_slicing: Whether to use attention slicing
    :type attention_slicing: bool

    :param xformers_memory_efficient: Whether to use memory efficient attention implementation from xformers
    :type xformers_memory_efficient: bool

    :param gradient_checkpointing: Whether to use gradient checkpointing
    :type gradient_checkpointing: bool

    :param peft_config: configuration for peft (Parameter Efficient Fine-Tuning library).
        Peft is designed to reduce the number of parameters to train and the memory footprint,
        without significant performance loss. It supports multiple techniques such as LORA
        or prefix tuning (cf. https://github.com/huggingface/peft).

        Here is an example of LORA configuration:
            {"peft_type": "LORA", "r": 8, "lora_alpha": 32, "lora_dropout": 0.1}

        (parameter-efficient fine-tuning was previously done in trlx with OpenDelta, but it is no longer supported)
    :type peft_config: Union[peft.PeftConfig, Dict[str, Any]]
    """

    model_path: str = None
    model_arch_type: str = None
    attention_slicing: bool = False
    xformers_memory_efficient: bool = False 
    gradient_checkpointing: bool = False
    peft_config: Dict[str, Any] = field(default_factory=dict) # TODO: add PEFT support



@dataclass
class SamplerConfig(ConfigClass):
    guidance_scale : float = 5.0 # if guidance is being used
    guidance_rescale : float = None # see https://arxiv.org/pdf/2305.08891.pdf
    num_inference_steps : int = 50
    eta : float = 1
    postprocess : bool = False # If true, post processes latents to images (uint8 np arrays)
    img_size : int = 512

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


def merge(base: Dict, update: Dict, updated: Set) -> Dict:
    "Recursively updates a nested dictionary with new values"
    for k, v in base.items():
        if k in update and isinstance(v, dict):
            base[k] = merge(v, update[k], updated)
            updated.add(k)
        elif k in update:
            base[k] = update[k]
            updated.add(k)

    return base

@dataclass
class DRLXConfig(ConfigClass):
    """
    Top-level config

    :param model: Model config
    :type model: ModelConfig

    :param optimizer: Optimizer config
    :type optimizer: OptimizerConfig

    :param scheduler: Scheduler config
    :type scheduler: SchedulerConfig

    :param train: Training config
    :type train: TrainConfig

    :param logging: Logging config
    :type logging: LoggingConfig

    :param method: Method config
    :type method: MethodConfig
    """

    model: ModelConfig
    sampler: SamplerConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    train: TrainConfig
    logging: LoggingConfig
    method: MethodConfig

    @classmethod
    def load_yaml(cls, yml_fp: str):
        """
        Load yaml file as DRLXConfig.

        :param yml_fp: Path to yaml file
        :type yml_fp: str
        """
        with open(yml_fp, mode="r") as file:
            config = yaml.safe_load(file)
        return cls.from_dict(config)

    def to_dict(self):
        """
        Convert TRLConfig to dictionary.
        """
        data = {
            "method": self.method.__dict__,
            "model": self.model.__dict__,
            "sampler": self.sampler.__dict__,
            "optimizer": self.optimizer.__dict__,
            "scheduler": self.scheduler.__dict__,
            "train": self.train.__dict__,
            "logging": self.logging.__dict__
        }

        return data

    @classmethod
    def from_dict(cls, config: Dict):
        """
        Convert dictionary to DRLXConfig.
        """
        return cls(
            method=get_method(config["method"]["name"]).from_dict(config["method"]),
            model=ModelConfig.from_dict(config["model"]),
            sampler=SamplerConfig.from_dict(config["sampler"]),
            optimizer=OptimizerConfig.from_dict(config["optimizer"]),
            scheduler=SchedulerConfig.from_dict(config["scheduler"]),
            train=TrainConfig.from_dict(config["train"]),
            logging=LoggingConfig.from_dict(config["logging"]),
        )

    @classmethod
    def update(cls, baseconfig: Dict, config: Dict):
        update = {}
        # unflatten a string variable name into a nested dictionary
        # key1.key2.key3: value -> {key1: {key2: {key3: value}}}
        for name, value in config.items():
            if isinstance(value, dict):
                update[name] = value
            else:
                *layers, var = name.split(".")
                if layers:
                    d = update.setdefault(layers[0], {})
                    for layer in layers[1:]:
                        d = d.setdefault(layer, {})
                    d[var] = value

        if not isinstance(baseconfig, Dict):
            baseconfig = baseconfig.to_dict()

        updates = set()
        merged = merge(baseconfig, update, updates)

        for param in update:
            if param not in updates:
                raise ValueError(f"parameter {param} is not present in the config (typo or a wrong config)")

        return cls.from_dict(merged)

    def __str__(self):
        """Returns a human-readable string representation of the config."""
        import json

        return json.dumps(self.to_dict(), indent=4)
