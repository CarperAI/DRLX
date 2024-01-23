from abc import abstractmethod
from typing import Tuple, Callable

from PIL import Image

from drlx.pipeline import Pipeline

class DPOPipeline(Pipeline):
    """
    Pipeline for training with DPO. Returns prompts, chosen images, and rejected images
    """
    def __init__(self, *args):
        super().__init__(*args)
    
    @abstractmethod
    def __getitem__(self, index : int) -> Tuple[str, Image.Image, Image.Image]:
        pass

    def make_default_collate(self, prep : Callable):
        def collate(batch : Iterable[Tuple[str, Image.Image, Image.Image]]):
            prompts = [d[0] for d in batch]
            chosen = [d[1] for d in batch]
            rejected = [d[2] for d in batch]

            return prep(prompts, chosen, rejected)
        
        return collate

    

