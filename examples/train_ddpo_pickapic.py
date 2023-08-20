from drlx.trainer.ddpo_trainer import DDPOTrainer
from drlx.configs import DRLXConfig
from drlx.reward_modelling.pickscore import PickScoreModel
from drlx.utils import get_latest_checkpoint

# Pipeline first
from drlx.pipeline.pickapic_prompts import PickAPicPrompts

import torch

pipe = PickAPicPrompts()
resume = False

config = DRLXConfig.load_yaml("configs/ddpo_sd_pickscore.yml")
trainer = DDPOTrainer(config)

if resume:
    cp_dir = get_latest_checkpoint(f"checkpoints/{config.logging.run_name}")
    trainer.load_checkpoint(cp_dir)

trainer.train(pipe, PickScoreModel())