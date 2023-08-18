from abc import abstractmethod

from torchtyping import TensorType
from typing import Iterable

from transformers import AutoProcessor, AutoModel
import torch
from torch import nn
from PIL import Image

from drlx.utils import any_chunk

class RewardModel(nn.Module):
    """
    Generalized reward model. Can be a wrapper for any black-box function
    that produces reward given pixel values and text input.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def preprocess(
        self, *inputs
    ) -> Iterable[torch.Tensor]:
        """
        Preprocess any form of data into something that can be input into model (generally PIL images and text strings)
        """
        pass

    @abstractmethod
    def forward(
        self,
        *inputs
    ) -> TensorType["batch"]:
        """
        Given any form of raw data (may not be tensors, may not even be batched), processes into reward scores. Inputs must all be iterable
        """
        pass

class NNRewardModel(nn.Module):
    """
    Any reward model that requires a neural network. Currently single GPU.

    :param device: Device to store model on
    :type device: str

    :param dtype: Data type to use for model input

    :param batch_size: Batch size to pass input in during inference
    """
    def __init__(self, device='cpu', dtype=torch.float, batch_size=1):
        super().__init__()
        
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
    
    @abstractmethod
    def _forward(self, *inputs) -> Iterable[float]:
        """
        Actual forward pass on a single batch of data

        :param inputs: Arbitrary inputs to reward model

        :return: Rewards across batch of inputs
        :rtype: Iterable[float]
        """
        pass

    def forward(
        self,
        *inputs
    ) -> TensorType["batch"]:
        """
        Wrapper around _forward which chunks inputs based on batch size

        :param inputs: Arbitrary inputs to reward model

        :return Rewards across batch of inputs
        :rtype: torch.Tensor
        """
        inputs = [any_chunk(input, self.batch_size) for input in inputs]
        batched_inputs = zip(*inputs)
        outputs = [self._forward(*self.preprocess(*batch)) for batch in batched_inputs]
        return torch.cat(outputs)



