from drlx.trainer.ddpo_trainer import DDPOTrainer
from drlx.configs import DRLXConfig
from drlx.reward_modelling.toy_rewards import JPEGCompressability
from drlx.reward_modelling.aesthetics import Aesthetics
from drlx.utils import get_latest_checkpoint

# Pipeline first
from drlx.pipeline.pickapic_prompts import PickAPicPrompts

import torch

pipe = PickAPicPrompts()
resume = False

config = DRLXConfig.load_yaml("configs/ddpo_sd.yml")
trainer = DDPOTrainer(config)

if resume:
    cp_dir = get_latest_checkpoint(f"checkpoints/{config.logging.run_name}")
    trainer.load_checkpoint(cp_dir)

trainer.train(pipe, Aesthetics())