from torchtyping import TensorType
from typing import Iterable, Union, Callable, Type, Tuple

import torch
import numpy as np
from diffusers import UNet2DConditionModel, DDIMScheduler

from drlx.denoisers import BaseConditionalDenoiser
from drlx.configs import ModelConfig, SamplerConfig
from drlx.sampling import Sampler

class LDMUNet(BaseConditionalDenoiser):
    """
    Class for Latent Diffusion Model UNet denoiser. Can optionally pass sampler information, though it is not required. Generally used in tandem with a diffusers pipeline.

    :param config: Configuration for model
    :type config: ModelConfig

    :param sampler_config: Configuration for sampler (optional). If provided, will create a default sampler.
    :type sampler_config: SamplerConfig

    :param sampler: Can be provided as alternative to sampler_config (also optional). If neither are provided, a default sampler will be used.
    :type sampler: Sampler
    """
    def __init__(self, config : ModelConfig, sampler_config : SamplerConfig = None, sampler : Sampler = None):
        super().__init__(config, sampler_config, sampler)

        self.unet : UNet2DConditionModel = None
        self.text_encoder = None
        self.vae = None
        self.encode_prompt : Callable = None

        self.tokenizer = None
        self.scheduler = None

        self.scale_factor = None

    def get_input_shape(self) -> Tuple[int]:
        """
        Figure out latent noise input shape for the UNet. Requires that unet and vae are defined

        :return: Input shape as a tuple
        :rtype: Tuple[int]
        """
        assert self.unet and self.vae, "Cannot get input shape if model not initialized"

        in_channels = self.unet.config.in_channels
        sample_size = self.sampler.config.img_size // self.scale_factor

        return (in_channels, sample_size, sample_size)
    
    def from_pretrained_pipeline(self, cls : Type, path : str):
        """
        Get unet from some pretrained model pipeline

        :param cls: Class to use for pipeline (i.e. StableDiffusionPipeline)
        :type cls: Type

        :param path: Path to pretrained pipeline
        :type path: str

        :return: an LDMUNet object with UNet, Text Encoder, VAE, tokenizer and scheduler from pretrained pipeline. Also returns the pretrained pipeline in case caller needs it.
        :rtype: LDMUNet
        """
        pipe = cls.from_pretrained(path)

        if self.config.attention_slicing: pipe.enable_attention_slicing()
        if self.config.xformers_memory_efficient: pipe.enable_xformers_memory_efficient_attention()

        self.unet = pipe.unet
        self.text_encoder = pipe.text_encoder
        self.vae = pipe.vae
        self.scale_factor = pipe.vae_scale_factor
        self.encode_prompt = pipe._encode_prompt

        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        self.tokenizer = pipe.tokenizer
        self.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        if self.config.gradient_checkpointing: self.unet.enable_gradient_checkpointing()

        return self, pipe
    
    def preprocess(self, text : Iterable[str], mode = "tokens", **embed_kwargs):
        """
        Preprocess text input, either into tokens or into embeddings.
        
        :param mode: Either "tokens" or "embeds"
        :type mode: str

        :param text: Text to preprocess
        :type text: Iterable[str]

        :return: Either a tuple of tensors for input_ids and attention_mask or a tensor of embeddings
        :rtype: Union[Tuple[Tensor, Tensor], Tensor]
        """

        if mode == "tokens":
            tok_out = self.tokenizer(
                text,
                padding = 'max_length',
                max_length = self.tokenizer.model_max_length,
                truncation = True,
                return_tensors = "pt"
            )
            return tok_out.input_ids, tok_out.attention_mask
        elif mode == "embeds":
            return self.encode_prompt(text, **embed_kwargs)
        else: 
            raise ValueError("Invalid mode specified for preprocessing")

    @torch.no_grad()
    def postprocess(self, output : TensorType["batch", "channels", "height", "width"], vae_device = None):
        """
        Post process 
        """
        if vae_device is not None:
            self.vae = self.vae.to(vae_device)
            output = output.to(vae_device)
        images = self.vae.decode(1 / 0.18215 * output).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0,2,3,1).numpy()
        images = (images * 255).round().astype(np.uint8)
        return images

    def forward(
            self,
            pixel_values : TensorType["batch", "channels", "height", "width"],
            time_step : Union[TensorType["batch"], int], # Note diffusers tyically does 999->0 as steps
            input_ids : TensorType["batch", "seq_len"] = None,
            attention_mask : TensorType["batch", "seq_len"] = None,
            text_embeds : TensorType["batch", "d"] = None
        ) -> TensorType["batch", "channels", "height", "width"]:
        """
        For text conditioned UNET, inputs are assumed to be:
        pixel_values, input_ids, attention_mask, time_step
        """
        with torch.no_grad():
            if text_embeds is None:
                text_embeds = self.text_encoder(input_ids, attention_mask)[0]

        return self.unet(
            pixel_values,
            time_step,
            encoder_hidden_states = text_embeds
        ).sample
    

        
