from typing import Iterable, Dict, Any
from torchtyping import TensorType

from abc import abstractmethod

import os

import torch
from torch import nn

from drlx.configs import ModelConfig

class BaseConditionalDenoiser(nn.Module):
    def __init__(self, config : ModelConfig):
        super().__init__()

        self.config = config
        self.scheduler = None
    
    @abstractmethod
    def preprocess(self, *inputs):
        """
        Called on the conditioning input
        """
        pass

    @abstractmethod
    def forward(self, *inputs):
        pass

    def save_progress(self, fp : str, components : Dict[str, Any]):
        if os.isdir("fp")
