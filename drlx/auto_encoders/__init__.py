from typing import Iterable
from torchtyping import TensorType

import torch
from torch import nn
from diffusers import AutoencoderKL

import numpy as np
from PIL import Image

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.latent_shape = None
    
    def determine_latent_shape(self):
        dummy_image = np.random.rand(256,256,3)
        images = Image.fromarray((dummy_image * 255).astype(np.uint8))
        images = [images] * 2
        inputs = self.preprocess(images)
        inputs = self.encode(inputs)
        return list(inputs.shape)[1:]

    def get_latent_shape(self):
        if self.latent_shape is None:
            self.latent_shape = self.determine_latent_shape()
        return self.latent_shape

    def preprocess(self, images : Iterable[Image.Image]) -> TensorType["batch", "channels", "height", "width"]:
        """
        Preprocess images into tensors before input to encode.
        """
        pass

    def postprocess(self, images : TensorType["batch", "channels", "height", "width"]) -> Iterable[Image.Image]:
        """
        Postprocess tensors into images after decode
        """
        pass

    def encode(self, pixel_values : TensorType["batch", "channels", "height", "width"]) -> torch.Tensor:
        pass

    def decode(self, latent : torch.Tensor) -> TensorType["batch", "channels", "height", "width"]:
        pass

    def forward(self, pixel_values : TensorType["batch", "channels", "height", "width"]) -> TensorType["batch", "channels", "height", "width"]:
        latent = self.encode(pixel_values)
        return self.decode(latent)

class PipelineAutoEncoder(AutoEncoder):
    """
    Load a pretrained autoencoder from a diffusion pipeline. Image2Image pipelines are generally preferred
    as they retain an imageprocessor 

    :param pipeline_cls: The class of the diffusion pipeline (i.e. diffusers.StableDiffusionPipeline)
    
    :param folder_or_path: The folder or path to the pretrained model
    """
    def __init__(self, pipeline_cls, folder_or_path : str):
        super().__init__()

        pipe = pipeline_cls.from_pretrained(folder_or_path)
        self.model = pipe.vae
        self.prep = pipe.image_processor
        self.latent_shape = None
        self.determine_latent_shape()
    
    def preprocess(self, images):
        return self.prep.preprocess(images)

    def encode(self, pixel_values):
        with torch.no_grad():
            return self.model.encode(pixel_values).latent_dist.sample()
    
    def decode(self, latent):
        with torch.no_grad():
            return self.model.decode(latent)

class PretrainedAutoEncoder(AutoEncoder):
    """
    Load a pretrained autoencoder directly from a checkpoint in diffusers.
    """
    def __init__(self, folder_or_path : str):
        super().__init__()

        self.model = AutoencoderKL.from_pretrained(folder_or_path)
        self.prep = self.model.image_processor
        self.latent_shape = None
        self.determine_latent_shape()
    
    def preprocess(self, images):
        return self.prep.preprocess(images)

    def encode(self, pixel_values):
        with torch.no_grad():
            return self.model.encode(pixel_values).latent_dist.sample()
    
    def decode(self, latent):
        with torch.no_grad():
            return self.model.decode(latent)
