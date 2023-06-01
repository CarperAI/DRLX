from typing import Optional, Callable

from abc import abstractmethod

from drlx.configs import TrainConfig
from drlx.auto_encoders import AutoEncoder
from drlx.reward_modelling import RewardModel   
from drlx.denoisers import BaseConditionalDenoiser
from drlx.pipeline import Pipeline


class BaseTrainer:
    def __init__(self, config : TrainConfig):
        self.config = config

    def train(self, pipeline : Pipeline, model : BaseConditionalDenoiser, ae_model : Optional[AutoEncoder] = None, reward_model : Optional[RewardModel] = None):
        pass