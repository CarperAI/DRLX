from typing import Iterable, Dict, Any, Optional, Tuple
from torchtyping import TensorType

from abc import abstractmethod

import os

import torch
from torch import nn
import numpy as np

from drlx.configs import ModelConfig, SamplerConfig
from drlx.sampling import Sampler

class BaseConditionalDenoiser(nn.Module):
    """
    Base class for any denoiser that takes a conditioning signal during denoising process, including text conditioned denoisers.

    :param config: Configuration for model
    :type config: ModelConfig

    :param sampler_config: Configuration for sampler (optional). If provided, will create a default sampler.
    :type sampler_config: SamplerConfig

    :param sampler: Can be provided as alternative to sampler_config (also optional). If neither are provided, a default sampler will be used.
    :type sampler: Sampler
    """
    def __init__(self, config : ModelConfig, sampler_config : SamplerConfig = None, sampler : Sampler = None):
        super().__init__()

        self.config = config
        self.scheduler = None

        if sampler_config is None and sampler is None:
            self.sampler = Sampler(SamplerConfig())
        else:
            self.sampler = Sampler(sampler_config) if sampler_config is not None else sampler
    
    def sample(self, **kwargs):
        """
        Use the sampler to sample an image. Will require postprocess to output an image. Note that different samplers have different outputs.

        :param kwargs: Keyword arguments to sampler

        :return: Varies per sampler but always includes denoised latent/images
        """
        return self.sampler.sample(**kwargs)
        
    @abstractmethod
    def get_input_shape(self) -> Tuple:
        """
        Get input shape for denoiser. Useful during training + sampling when shape of input noise to denoiser is needed.

        :return: Input shape as a tuple
        :rtype: Tuple[int]
        """
        pass

    @abstractmethod
    def preprocess(self, *inputs) -> TensorType["batch", "embedding_dim"]:
        """
        Called on the conditioning input (typically: tokenizes text prompt)
        
        :return: Conditioning input embeddings (i.e. text embeddings) as tensors
        :rtype: torch.Tensor
        """
        pass

    @abstractmethod
    def postprocess(self, output) -> np.ndarray:
        """
        Called on the output from the model after sampling to give final image

        :return: Final denoised image as uint8 numpy array
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def forward(self, *inputs) -> TensorType["batch", "channels", "height", "width"]:
        """
        Forward pass for denoiser. Output varies based on prediction type.
        """
        pass

    # === LATENT DIFFUSION ===

    @abstractmethod
    def encode(self, pixel_values : TensorType["batch", "channels", "height", "width"]) -> torch.Tensor:
        """
        Encode image into latent vector
        """
        pass

    @abstractmethod
    def decode(self, latent : torch.Tensor) -> TensorType["batch", "channels", "height", "width"]:
        """
        Decode latent vector into an image (typically called in postprocess)
        """
        pass

    