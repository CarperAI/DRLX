
from drlx.denoisers.ldm_unet import LDMUNet
from drlx.configs import ModelConfig, SamplerConfig
from diffusers import StableDiffusionPipeline

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Pipeline first
from drlx.pipeline.prompt_pipeline import PromptPipeline

class ToyPipeline(PromptPipeline):
    def __init__(self):
        super().__init__()

        self.dataset = ["A cat", "A dog", "A bird", "A fish"] * 100
    
    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)


model = LDMUNet(ModelConfig(), sampler_config = SamplerConfig())
model.from_pretrained_pipeline(StableDiffusionPipeline, "CompVis/stable-diffusion-v1-4")
model = model.to('cuda')

pipe = ToyPipeline()
pipe.set_preprocess_fn(model.preprocess)

text = "A cat"
input_ids, attention_mask = model.preprocess([text])

loader = pipe.create_loader(8)

in_shape = model.get_input_shape()

for batch in loader:
    with torch.no_grad():
        tokens, masks = batch
        tokens = tokens.to('cuda')
        masks = masks.to('cuda')
        bs = len(tokens)

        print(tokens.shape)
        print(masks.shape)
    
        max_ts = model.scheduler.config.num_train_timesteps

        latents = torch.randn(bs, *in_shape).to('cuda')

        for ts in tqdm(range(max_ts)):
            # Yes this isn't how sampling works but it's good enough for testing the input/output of model :P
            latents = model(latents, tokens, masks, ts)

        print(latents.shape)

        exit()
