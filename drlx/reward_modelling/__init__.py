from abc import abstractmethod

from torchtyping import TensorType
from typing import Iterable

from transformers import AutoProcessor, AutoModel
import torch
from torch import nn
from PIL import Image

class RewardModel(nn.Module):
    """
    Generalized reward model. Can be a wrapper for any black-box function
    that produces reward given pixel values and text input.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def preprocess(
        self,
        images : Iterable[Image.Image],
        prompts : Iterable[str]
    ) -> Iterable[torch.Tensor]:
        pass

    @abstractmethod
    def forward(
        self,
        images : Iterable[Image.Image],
        prompts : Iterable[str]
    ) -> TensorType["batch"]:
        pass