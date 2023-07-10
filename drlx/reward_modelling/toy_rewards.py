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

    def forward(self, pixel_values, input_ids, attention_mask):
        return pixel_values[:, 2].mean(dim = 1)
    
class JPEGCompressability(RewardModel):
    """
    Rewards JPEG compression potential of image
    """
    def __init__(self, quality=10):
        super().__init__()
        self.quality = quality

    def forward(self, pixel_values, input_ids, attention_mask):
        batch_size = pixel_values.shape[0]
        compressed_sizes = torch.empty(batch_size)

        for i in range(batch_size):
            image = Image.fromarray(np.uint8(pixel_values[i].permute(1, 2, 0).cpu().numpy() * 255))  # Convert to PIL image
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=self.quality)  # Save image to buffer
            compressed_sizes[i] = len(buffer.getvalue())  # Save size of compressed image

        return compressed_sizes