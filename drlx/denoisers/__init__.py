from typing import Iterable
from torchtyping import TensorType

from abc import abstractmethod

import torch
from torch import nn

from drlx.configs import ModelConfig

class BaseConditionalDenoiser(nn.Module):
    def __init__(self, config : ModelConfig):
        super().__init__()

        self.config = config
    
    @abstractmethod
    def preprocess(self, *inputs):
        """
        Called on the conditioning input
        """
        pass

    @abstractmethod
    def forward(self, *inputs):
        pass