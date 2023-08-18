.. _example: 

DRLX Example
============

This example demonstrates how to use DRLX to train a model with a custom prompt pipeline and reward model. The prompt pipeline will repeatedly provide the same prompt, "Photo of a mad scientist panda", and the reward model will reward images for having high contrast.

Custom Prompt Pipeline
-----------------------

First, we define a custom prompt pipeline that only gives a single phrase "Photo of a mad scientist panda" over and over.

.. code-block:: python

    from drlx.pipeline import PromptPipeline

    class MadScientistPandaPrompts(PromptPipeline):
        """
        Custom prompt pipeline that only gives a single phrase "Photo of a mad scientist panda" over and over.
        """
        def __getitem__(self, index):
            return "Photo of a mad scientist panda"
        
        def __len__(self):
            return 100000 # arbitrary

Custom Reward
----------------

Next, we define a custom reward model that rewards images for having high contrast. The contrast is calculated as the standard deviation of the pixel intensities.

.. code-block:: python

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

Training Setup
---------------

Now, we set up the training process. We use the MadScientistPandaPrompts as the prompt pipeline and the HighContrastReward as the reward model.

.. code-block:: python

    from drlx.trainer.ddpo_trainer import DDPOTrainer
    from drlx.configs import DRLXConfig
    from drlx.reward_modelling.toy_rewards import JPEGCompressability
    from drlx.reward_modelling.aesthetics import Aesthetics
    from drlx.utils import get_latest_checkpoint

    # Pipeline first
    from drlx.pipeline.pickapic_prompts import PickAPicPrompts

    import torch

    pipe = MadScientistPandaPrompts()

    config = DRLXConfig.load_yaml("configs/ddpo_sd.yml")
    trainer = DDPOTrainer(config)

    trainer.train(pipe, HighContrastReward())

For accelerated training, simply run the following command:

.. code-block:: bash

    accelerate launch -m [script]

Loading the Model and Performing Inference
--------------------------------------------

After training, we can load the model and perform inference with it using a default sampler.

.. code-block:: python
    
    # Load the trainer from a checkpoint if you wanted to resume training
    # Trainer by default saves both output and checkpoint in seperate folders specified by run_name
    checkpoint_path = "checkpoints/run_name"
    output_path = "output/run_name"
    trainer.load_checkpoint(checkpoint_path)

    # Otherwise, you can just use a pretrained pipeline
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(output_path, local_files_only = True)

To actually run this code or make tweaks, please see the notebooks or scripts under the examples folder.





