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
    Rewards JPEG compression potential of image
    """
    def __init__(self, quality=10):
        super().__init__()
        self.quality = quality

    def forward(self, images, prompts):
        pixel_values = torch.from_numpy(images).permute(0, 3, 1, 2) / 255
        batch_size = pixel_values.shape[0]
        compressed_sizes = torch.empty(batch_size)

        for i in range(batch_size):
            image = Image.fromarray(np.uint8(pixel_values[i].permute(1, 2, 0).cpu().numpy() * 255))  # Convert to PIL image
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=self.quality)  # Save image to buffer
            compressed_sizes[i] = len(buffer.getvalue())  # Save size of compressed image

        return compressed_sizes