from torchtyping import TensorType
from typing import Iterable, Union

import torch
from diffusers import UNet2DConditionModel

from drlx.denoisers import BaseConditionalDenoiser

class TextConditionedUNet(BaseConditionalDenoiser):
    def __init__(self, config):
        super().__init__(config)

        self.unet : UNet2DConditionModel = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None

    def from_pretrained_pipeline(self, cls, path):
        """
        Get unet from some pretrained model pipeline
        """
        pipe = cls.from_pretrained(path)
        self.unet = pipe.unet
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.scheduler = pipe.scheduler
    
    def preprocess(self, text : Iterable[str]):
        tok_out = self.tokenizer(
            text,
            padding = 'max_length',
            max_length = self.tokenizer.model_max_length,
            truncation = True,
            return_tensors = "pt"
        )
        return tok_out.input_ids, tok_out.attention_mask

    def forward(
            self,
            pixel_values : TensorType["batch", "channels", "height", "width"],
            input_ids : TensorType["batch", "seq_len"],
            attention_mask : TensorType["batch", "seq_len"],
            time_step : Union[TensorType["batch"], int] # Note diffusers tyically does 999->0 as steps
        ):
        """
        For text conditioned UNET, inputs are assumed to be:
        pixel_values, input_ids, attention_mask, time_step
        """
        with torch.no_grad():
            text_embeds = self.text_encoder(input_ids, attention_mask)[0]

        return self.unet(
            pixel_values,
            time_step,
            encoder_hidden_states = text_embeds
        )
            

        