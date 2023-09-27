import sys
sys.path.append("./src")

from drlx.pipeline.pickapic_wds import PickAPicPipeline
from drlx.trainer.dpo_trainer import DPOTrainer
from drlx.configs import DRLXConfig

pipe = PickAPicPipeline()
resume = False

config = DRLXConfig.load_yaml("configs/dpo_pickapic.yml")
trainer = DPOTrainer(config)

trainer.train(pipe)