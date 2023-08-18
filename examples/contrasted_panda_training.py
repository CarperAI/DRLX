# Set up pipeline with repeating prompts

from drlx.pipeline import PromptPipeline

class MadScientistPandaPrompts(PromptPipeline):
    """
    Custom prompt pipeline that only gives a single phrase "Photo of a mad scientist panda" over and over.
    """
    def __getitem__(self, index):
        return "Photo of a mad scientist panda"

    def __len__(self):
        return 100000 # arbitrary

# Next we make our reward model

from drlx.reward_modelling import RewardModel
import numpy as np
import torch

class HighContrastReward(RewardModel):
    """
    Rewards high contrast in the image.
    """
    def forward(self, images, prompts):
        # If the input is a list of PIL Images, convert to numpy array
        if isinstance(images, list):
            images = np.array([np.array(img) for img in images])

        # Calculate the standard deviation of the pixel intensities for each image
        contrast = images.std(axis=(1,2,3))  # N

        return torch.from_numpy(contrast)

# Next, we setup trainer using default config

from drlx.trainer.ddpo_trainer import DDPOTrainer
from drlx.configs import DRLXConfig

pipe = MadScientistPandaPrompts()

config = DRLXConfig.load_yaml("configs/ddpo_sd.yml")

# Some changes to config for our use-case
config.train.num_samples_per_epoch = 32
config.train.batch_size = 4 # adjust as needed
config.logging.run_name = "contrasting_panda"

trainer = DDPOTrainer(config)

# If we wanted to resume a run... we can make this little change
from drlx.utils import get_latest_checkpoint

RESUME = False
if RESUME:
    cp_dir = get_latest_checkpoint(f"checkpoints/{config.logging.run_name}")
    trainer.load_checkpoint(cp_dir)

trainer.train(pipe, HighContrastReward())