from drlx.trainer.base_accelerate import AcceleratedTrainer
from drlx.configs import DRLXConfig, DPOConfig

class DPOTrainer(AcceleratedTrainer):
    """ 
    DPO Accelerated Trainer initilization from config. During init, sets up model, optimizer, sampler and logging

    :param config: DRLX config
    :type config: DRLXConfig
    """

    def __init__(self, config : DRLXConfig):    
        super().__init__(config)

        assert isinstance(self.config.method, DPOConfig), "ERROR: Method config must be DDPO config"
    
    def train(self, pipeline):
        """
        Trains model based on config parameters. Needs to be passed a pipeline that
        supplies chosen images, rejeceted images and prompts

        :param pipeline: Pipeline to draw images and prompts from
        :type Pipline: Pipeline
        """
        pass

