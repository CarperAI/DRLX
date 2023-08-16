from typing import Iterable
from torchtyping import TensorType

import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image

from drlx.reward_modelling import NNRewardModel

class PickScoreModel(NNRewardModel):
    """
    Reward model using PickScore model from PickAPic
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_path = "yuvalkirstain/PickScore_v1"

        self.model = AutoModel.from_pretrained(model_path).to(self.device).to(self.dtype)
        self.processor = AutoProcessor.from_pretrained(processor_path)

    def preprocess(self, images : Iterable[Image.Image], prompts : Iterable[str]):
        """
        Preprocess images and prompts into tensors, making sure to move to correct device and data type
        """
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )

        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )

        pixels, ids, mask = image_inputs['pixel_values'], text_inputs['input_ids'], text_inputs['attention_mask']
        pixels = pixels.to(device = self.device, dtype = self.dtype)
        ids = ids.to(device = self.device)
        mask = mask.to(device = self.device)
        return pixels, ids, mask

        
    @torch.no_grad() # This repo does not train the model, so in general, no_grad will be used here
    def _forward(
        self,
        pixel_values : TensorType["batch", "channels", "height", "width"],
        input_ids : TensorType["batch", "sequence"],
        attention_mask : TensorType["batch", "sequence"]
    ) -> TensorType["batch"]:
        image_embs = self.model.get_image_features(pixel_values=pixel_values)
        image_embs /= image_embs.norm(dim=-1, keepdim=True)

        text_embs = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        text_embs /= text_embs.norm(dim=-1, keepdim=True)

        scores = torch.einsum('bd,bd->b', image_embs, text_embs)
        scores = self.model.logit_scale.exp() * scores

        return scores