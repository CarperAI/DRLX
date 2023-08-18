"""
This script tests if pipelines + LDM unet work with DDPO sampler
"""

from drlx.denoisers.ldm_unet import LDMUNet
from drlx.configs import ModelConfig, SamplerConfig
from drlx.sampling import DDPOSampler
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

    def create_loader(self, batch_size):
        return torch.utils.data.DataLoader(self, batch_size)


model = LDMUNet(ModelConfig(), sampler = DDPOSampler())
model.from_pretrained_pipeline(StableDiffusionPipeline, "CompVis/stable-diffusion-v1-4")
model = model.to('cuda')

pipe = ToyPipeline()

text = "A cat"
input_ids, attention_mask = model.preprocess([text])

loader = pipe.create_loader(8)

in_shape = model.get_input_shape()

for prompts in loader:
    with torch.no_grad():

        latents, all_preds, log_probs = model.sampler.sample(
            prompts,
            model,
            device = 'cuda'
        )

        print(latents.shape)

        exit()
