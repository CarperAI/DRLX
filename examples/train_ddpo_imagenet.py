import torch
from drlx.trainer.ddpo_trainer import DDPOTrainer
from drlx.configs import DRLXConfig
from drlx.reward_modelling.aesthetics import Aesthetics
from drlx.pipeline.imagenet_animal_prompts import ImagenetAnimalPrompts
from drlx.utils import get_latest_checkpoint

config = DRLXConfig.load_yaml("configs/ddpo_sd_imagenet.yml")

pipe = ImagenetAnimalPrompts(prefix='', postfix='', num=config.train.num_samples_per_epoch)
resume = False

trainer = DDPOTrainer(config)

if resume:
    cp_dir = get_latest_checkpoint(f"checkpoints/{config.logging.run_name}")
    trainer.load_checkpoint(cp_dir)

trainer.train(pipe, Aesthetics())