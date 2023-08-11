import torch
from transformers import AutoModel, AutoProcessor

from drlx.reward_modelling import RewardModel

class PickScoreModel(RewardModel):
    def __init__(self, device = 'cpu', dtype = torch.float):
        super().__init__()

        processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_path = "yuvalkirstain/PickScore_v1"

        self.model = AutoModel.from_pretrained(model_path).to(device).to(dtype)
        self.processor = AutoProcessor.from_pretrained(processor_path)

        self.device = device
        self.dtype = dtype

    def preprocess(self, images, prompts):
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
    def _forward(self, pixel_values, input_ids, attention_mask):
        image_embs = self.model.get_image_features(pixel_values=pixel_values)
        image_embs /= image_embs.norm(dim=-1, keepdim=True)

        text_embs = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        text_embs /= text_embs.norm(dim=-1, keepdim=True)

        scores = torch.einsum('bd,bd->b', image_embs, text_embs)
        scores = self.model.logit_scale.exp() * scores

        return scores
    
    @torch.no_grad()
    def forward(self, images, prompts):
        return self._forward(*self.preprocess(images, prompts))
