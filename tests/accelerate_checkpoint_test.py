from drlx.trainer.ddpo_trainer import DDPOTrainer
from drlx.configs import DRLXConfig
import torch
import os

config = DRLXConfig.load_yaml("configs/ddpo_sd.yml")
trainer = DDPOTrainer(config)

fp = "./checkpoints_saving_test"
trainer.save_pretrained("./output/saving_test")
trainer.save_checkpoint(fp)

trainer.load_checkpoint(fp)

from diffusers import StableDiffusionPipeline

#pipe = trainer.extract_pipeline()
pipe = StableDiffusionPipeline.from_pretrained("./output/saving_test")
print("Successfully loaded pipeline")