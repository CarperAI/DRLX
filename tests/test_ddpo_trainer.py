from drlx.trainer.ddpo_trainer import DDPOTrainer
from drlx.configs import DRLXConfig
from drlx.reward_modelling.toy_rewards import JPEGCompressability

# Pipeline first
from drlx.pipeline.pickapic_prompts import PickAPicPrompts

import torch

pipe = PickAPicPrompts()

config = DRLXConfig.load_yaml("configs/ddpo_sd.yml")
trainer = DDPOTrainer(config)

trainer.train(pipe, JPEGCompressability())