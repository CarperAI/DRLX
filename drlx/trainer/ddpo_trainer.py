from dataclasses import dataclass

from drlx.configs import MethodConfig
from drlx.trainer import BaseTrainer
from drlx.pipeline.prompt_pipeline import PromptPipeline
from drlx.denoisers import BaseConditionalDenoiser
from drlx.reward_modelling import RewardModel

@dataclass
class DDPOConfig:
    # TODO
    pass

class DDPOTrainer(BaseTrainer):
    def __init__(self):
        pass

    def train(self, pipeline : PromptPipeline, model : BaseConditionalDenoiser, reward_model : RewardModel):
        pass