import sys

sys.path.append("./src")

from drlx.trainer.dpo_trainer import DPOTrainer
from drlx.configs import DRLXConfig
from drlx.utils import get_latest_checkpoint

# Pipeline first
from drlx.pipeline.pickapic_dpo import PickAPicDPOPipeline

import torch

pipe = PickAPicDPOPipeline()
resume = False

config = DRLXConfig.load_yaml("configs/dpo_pickapic.yml")
trainer = DPOTrainer(config)

if resume:
    cp_dir = get_latest_checkpoint(f"checkpoints/{config.logging.run_name}")
    trainer.load_checkpoint(cp_dir)

trainer.train(pipe)