from abc import abstractmethod

from torchtyping import TensorType
from typing import Iterable

from transfomers import AutoProcessor, AutoModel
import torch

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
        pixel_values : TensorType["batch", "channels", "height", "width"], 
        input_ids : TensorType["batch", "seq_len"],
        attention_mask : TensorType["batch", "seq_len"]
    ) -> TensorType["batch"]:
        pass