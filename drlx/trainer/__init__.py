from typing import Optional, Callable, Dict, Any, Iterable
from torchtyping import TensorType

from abc import abstractmethod
import os

import torch

from drlx.configs import DRLXConfig
from drlx.reward_modelling import RewardModel   
from drlx.denoisers.ldm_unet import LDMUNet
from drlx.pipeline import Pipeline
from drlx.utils import get_optimizer_class, get_scheduler_class, get_diffusion_pipeline_class

from PIL import Image

class BaseTrainer:
    def __init__(self, config : DRLXConfig):
        self.config = config

        # Assume these are defined in base classes
        self.optimizer = None
        self.scheduler = None 
        self.model = None
    
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
        scheduler = scheduler_class(self.optimizer, **self.config.scheduler.kwargs)
        return scheduler

    def get_arch(self, config):
        model_name = LDMUNet # nothing else is supported for now (TODO: add support for other models)
        return model_name

    @abstractmethod
    def train(self, pipeline : Pipeline, reward_fn : Callable[[Iterable[Image.Image], Iterable[str]], TensorType["batch"]]):
        """
        Trains model on a given pipeline using a given reward function.
        
        :param pipeline: Data pipeline used for training
        :param reward_fn: Function used to get rewards. Should take tuples of images (either as a sequence of numpy arrays, or as a list of images)
        """
        pass

    def save_checkpoint(self, fp : str, components : Dict[str, Any], index : int = None):
        if not os.path.exists(fp):
            os.makedirs(fp)

        if index is not None:
            fp = os.path.join(fp, str(index))
            if not os.path.exists(fp):
                os.makedirs(fp)
        
        for key, component in components.items():
            torch.save(component, os.path.join(fp, f"{key}.pt"))

    def load_checkpoint(self, fp: str, index: int = None) -> Dict[str, Any]:
        # If an index is given, update the file path to include the subdirectory with the index as its name
        if index is not None:
            fp = os.path.join(fp, str(index))

        # Initialize an empty dictionary to store the loaded components
        loaded_components = {}

        # Iterate through the files in the directory
        for file_name in os.listdir(fp):
            # Check if the file has a .pt extension
            if file_name.endswith(".pt"):
                # Load the component using torch.load and add it to the loaded_components dictionary
                key = file_name[:-3]  # Remove the .pt extension from the file name to get the key
                component = torch.load(os.path.join(fp, file_name))
                loaded_components[key] = component

        return loaded_components
