import einops as eo
import torch

def get_time_ids(batch):
    """
    Computes time ids needed for SDXL in a heavily simplified manner that only requires image size
    (assumes square images). Assumes crop top left is (0,0) for all images. Infers all needed info from batch of images.
    """

    b, c, h, w = batch.shape

    # input_size, crop, input_size
    add_time_ids = torch.tensor([h, w, 0, 0, h, w], device = batch.device, dtype = batch.dtype)
    return eo.repeat(add_time_ids, 'd -> (b d)', b = b)
