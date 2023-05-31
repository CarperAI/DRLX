from transfomers import AutoProcessor, AutoModel

from k_diffusion.reward_modelling import RewardModel

import torch

class PickScoreModel(RewardModel):
    def __init__(self):
        super().__init__()

        processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_path = "yuvalkirstain/PickScore_v1"

        self.model = AutoModel.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(processor_path)

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
        
        return image_inputs['pixel_values'], text_inputs['input_ids'], text_inputs['attention_mask']
    
    def forward(self, pixel_values, input_ids, attention_mask):
        # This repo does not train the model, so in general, no_grad will be used here
        with torch.no_grad():
            def normalize(t):
                return t / t.norm(dim=-1, keepdim=True)
            
            image_embs = self.model.get_image_features(pixel_values=pixel_values)
            image_embs = normalize(t)

            text_embs = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
            text_embs = normalize(t)

            scores = torch.einsum('bd,bd->b', image_embs, text_embs)
            scores = self.model.logit_scale.exp() * scores

            return scores
