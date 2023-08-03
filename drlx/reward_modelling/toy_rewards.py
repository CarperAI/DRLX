"""
Toy reward models for testing purposes
"""

from io import BytesIO
from PIL import Image

import torch
import numpy as np

from drlx.reward_modelling import RewardModel

class AverageBlueReward(RewardModel):
    """
    Rewards "blue-ness" of image
    """
    def __init__(self):
        super().__init__()

    def forward(self, images, prompts):
        # If the input is a list of PIL Images, convert to numpy array
        if isinstance(images, list):
            images = np.array([np.array(img) for img in images])

        blue_channel = images[:,:,:,2]  # N x 256 x 256

        # Calculate the mean of the blue channel for each image
        blueness = blue_channel.astype(float).mean(axis=(1,2))  # N
        blueness = (2 * blueness - 255)/255 # normalize to [0,1]

        return torch.from_numpy(blueness)
    
class JPEGCompressability(RewardModel):
    """
    Rewards JPEG compression potential of image (from https://arxiv.org/pdf/2305.13301.pdf)
    """
    def __init__(self, quality=10):
        super().__init__()
        self.quality = quality

    def encode_jpeg(self, x, quality = 95):
        img = Image.fromarray(x)
        buffer = BytesIO()
        img.save(buffer, 'JPEG', quality=quality)
        jpeg = buffer.getvalue()
        bytes = np.frombuffer(jpeg, dtype = np.uint8)
        return len(bytes) / 1000

    def forward(self, images, prompts):
        scores = [-1 * self.encode_jpeg(img) for img in images]
        return torch.tensor(scores)