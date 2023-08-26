import os
import glob
from enum import Enum
from itertools import repeat
from typing import Any, Dict, Iterable, Tuple
from collections import deque
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from diffusers import StableDiffusionPipeline
import logging
import time
from contextlib import contextmanager
from PIL import Image

import numpy as np

class OptimizerName(str, Enum):
    """Supported optimizer names"""

    ADAM: str = "adam"
    ADAMW: str = "adamw"
    ADAM_8BIT_BNB: str = "adam_8bit_bnb"
    ADAMW_8BIT_BNB: str = "adamw_8bit_bnb"
    SGD: str = "sgd"


def get_optimizer_class(name: OptimizerName):
    """
    Returns the optimizer class with the given name

    Args:
        name (str): Name of the optimizer as found in `OptimizerNames`
    """
    if name == OptimizerName.ADAM:
        return torch.optim.Adam
    if name == OptimizerName.ADAMW:
        return torch.optim.AdamW
    if name == OptimizerName.ADAM_8BIT_BNB.value:
        try:
            from bitsandbytes.optim import Adam8bit

            return Adam8bit
        except ImportError:
            raise ImportError(
                "You must install the `bitsandbytes` package to use the 8-bit Adam. "
                "Install with: `pip install bitsandbytes`"
            )
    if name == OptimizerName.ADAMW_8BIT_BNB.value:
        try:
            from bitsandbytes.optim import AdamW8bit

            return AdamW8bit
        except ImportError:
            raise ImportError(
                "You must install the `bitsandbytes` package to use 8-bit AdamW. "
                "Install with: `pip install bitsandbytes`"
            )
    if name == OptimizerName.SGD.value:
        return torch.optim.SGD
    supported_optimizers = [o.value for o in OptimizerName]
    raise ValueError(f"`{name}` is not a supported optimizer. " f"Supported optimizers are: {supported_optimizers}")


class SchedulerName(str, Enum):
    """Supported scheduler names"""

    COSINE_ANNEALING = "cosine_annealing"
    LINEAR = "linear"


def get_scheduler_class(name: SchedulerName):
    """
    Returns the scheduler class with the given name
    """
    if name == SchedulerName.COSINE_ANNEALING:
        return CosineAnnealingLR
    if name == SchedulerName.LINEAR:
        return LinearLR
    supported_schedulers = [s.value for s in SchedulerName]
    raise ValueError(f"`{name}` is not a supported scheduler. " f"Supported schedulers are: {supported_schedulers}")



class DiffusionPipelineName(str, Enum):
    """Supported diffusion pipeline names"""
    StableDiffusion = "stable_diffusion"

def get_diffusion_pipeline_class(name: DiffusionPipelineName):
    """
    Returns the diffusion pipeline class with the given name
    """
    if name == DiffusionPipelineName.StableDiffusion:
        return StableDiffusionPipeline
    supported_diffusion_pipelines = [d.value for d in DiffusionPipelineName]
    raise ValueError(f"`{name}` is not a supported diffusion pipeline. " f"Supported diffusion pipelines are: {supported_diffusion_pipelines}")

def any_chunk(x, chunk_size):
    """
    Chunks any iterable by chunk size
    """
    is_tensor = isinstance(x, torch.Tensor)

    x_chunks = [x[i:i+chunk_size] for i in range(0, len(x), chunk_size)]
    return torch.stack(x_chunks) if is_tensor else x_chunks

def suppress_warnings(prefix : str):
    """
    With logging module, suppresses any warnings that are coming from a logger
    with a given prefix
    """

    names = logging.root.manager.loggerDict
    names = list(filter(lambda x: x.startswith(prefix), names))
    for name in names:
        logging.getLogger(name).setLevel(logging.ERROR)
    
class Timer:
    """
    Utility class for timing models
    """
    def __init__(self):
        self.time = time.time()

    def hit(self) -> float:
        """
        Restarts timer and returns the time in seconds since last restart or initialization
        """
        new_time = time.time()
        res = new_time - self.time
        self.time = new_time
        return res

def get_latest_checkpoint(checkpoint_root):
    """
    Assume folder root_dir stores checkpoints for model, all named numerically (in terms of training steps associated with said checkpoints).
    This function returns the path to the latest checkpoint, aka the subfolder with largest numerical name. Returns none if the root dir is empty
    """
    subdirs = glob.glob(os.path.join(checkpoint_root, '*'))
    if not subdirs:
        return None
    
    # Filter out any paths that are not directories or are not numeric
    subdirs = [s for s in subdirs if os.path.isdir(s) and os.path.basename(s).isdigit()]
    # Find the maximum directory number (assuming all subdirectories are numeric)
    latest_checkpoint = max(subdirs, key=lambda s: int(os.path.basename(s)))
    return latest_checkpoint

class PerPromptStatTracker:
    """
    Stat tracker to normalize rewards across prompts. If there is a sufficient number of duplicate prompts, averages across rewards given for that specific prompts. Otherwise, simply averages across all rewards.

    :param buffer_size: How many prompts to consider for average
    :type buffer_size: int

    :param min_count: How many duplicates for a prompt minimum before we average over that prompt and not over all prompts
    :type min_count: int
    """
    def __init__(self, buffer_size : int, min_count : int):
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

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    From [diffusers repository](https://github.com/huggingface/diffusers/blob/a7508a76f025fcbe104c28f73dd17c8e866f655b/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L58).
    Copied here due to import errors when attempting to import from package
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

@contextmanager
def scoped_seed(seed : int = 0):
    """
    Set torch seed within a context. Useful for deterministic sampling.

    :param seed: Seed to use for random state
    :type seed: int
    """
    # Record the state of the RNG
    cpu_rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state()
    
    # Set the desired seed
    torch.manual_seed(seed)
    
    try:
        yield
    finally:
        # Restore the previous RNG state after exiting the scope
        torch.set_rng_state(cpu_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_state)

def save_images(images : np.array, fp : str):
    """
    Saves images to folder designated by fp
    """

    os.makedirs(fp, exist_ok = True)

    images = [Image.fromarray(image) for image in images]
    for i, image in enumerate(images):
        image.save(os.path.join(fp,f"{i}.png"))


