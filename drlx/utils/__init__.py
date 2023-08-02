import os
import glob
from enum import Enum
from itertools import repeat
from typing import Any, Dict, Iterable, Tuple

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from diffusers import StableDiffusionPipeline


def get_latest_checkpoint(root_dir):
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

#TODO: is this needed?
def infinite_dataloader(dataloader: Iterable, sampler=None) -> Iterable:
    """
    Returns a cyclic infinite dataloader from a finite dataloader
    """
    epoch = 0
    for _ in repeat(dataloader):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1

        yield from dataloader



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