from typing import Iterable, Dict, Any, Optional, Tuple
from torchtyping import TensorType

from abc import abstractmethod

import os

import torch
from torch import nn

from drlx.configs import ModelConfig, SamplerConfig
from drlx.sampling import Sampler

class BaseConditionalDenoiser(nn.Module):
    def __init__(self, config : ModelConfig, sampler_config : SamplerConfig = None, sampler : Sampler = None):
        super().__init__()

        self.config = config
        self.scheduler = None
        assert sampler_config is not None or sampler is not None, "Must provide one of sampler_config or sampler to model init"
        self.sampler = Sampler(sampler_config) if sampler_config is not None else sampler
    
    def sample(self, **kwargs):
        return self.sampler.sample(**kwargs)
        
    @abstractmethod
    def get_input_shape(self) -> Tuple:
        pass

    @abstractmethod
    def preprocess(self, *inputs):
        """
        Called on the conditioning input
        """
        pass

    @abstractmethod
    def postprocess(self, output):
        """
        Called on the output from the model after sampling
        """
        pass

    @abstractmethod
    def forward(self, *inputs):
        pass

    # === LATENT DIFFUSION ===

    @abstractmethod
    def encode(self, pixel_values : TensorType["batch", "channels", "height", "width"]) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, latent : torch.Tensor) -> TensorType["batch", "channels", "height", "width"]:
        pass

    