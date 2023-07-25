from abc import abstractmethod
from typing import Callable, Iterable, Tuple, Any

from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset

class Pipeline(Dataset):
    """
    Pipeline to use for data whe training some RL model

    :param prep_fn: Function that will be called on iterable of data elements from the pipeline. Typically takes a list of prompts and tokenizes them. Often should just be set to a models preprocessing function.
    :type prep_fn: Callable
    """
    def __init__(self, prep_fn : Callable = None):
        super().__init__()

        if not prep_fn:
            self.prep : Callable = lambda x: x # identity by default
        else:
            self.prep = prep_fn

    def create_train_loader(self, **kwargs) -> DataLoader:
        # By default just create_loader
        return self.create_loader(**kwargs)

    @abstractmethod
    def create_val_loader(self, **kwargs) -> DataLoader:
        pass

    @classmethod
    def make_default_collate(self, prep : Callable):
        def collate(batch : Iterable[Tuple[Image.Image, str]]):
            img_batch = [d[0] for d in batch]
            txt_batch = [d[1] for d in batch]

            return prep(img_batch, txt_batch)

        return collate

    def create_loader(self, **kwargs) -> DataLoader:
        if self.prep is None:
            raise ValueError("Preprocessing function must be set before creating a dataloader.")

        if 'shuffle' in kwargs:
            del kwargs['shuffle']

        return DataLoader(self, collate_fn = self.make_default_collate(self.prep), **kwargs)

class PromptPipeline(Pipeline):
    """
    Base class for a pipeline that provides text prompts.
    """

    @classmethod
    def make_default_collate(self, prep : Callable):
        def collate(batch : Iterable[str]):
            return prep(batch)

        return collate
    
