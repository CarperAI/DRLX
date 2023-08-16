from abc import abstractmethod
from typing import Callable, Iterable, Tuple, Any

from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset

class Pipeline(Dataset):
    """
    Pipeline for data during RL training. Subclasses should define some dataset with getitem and len methods.

    :param prep_fn: Function that will be called on iterable of data elements from the pipeline. Not always required, and by default is simply an identity function.
    :type prep_fn: Callable
    """
    def __init__(self, prep_fn : Callable = None):
        super().__init__()

        if not prep_fn:
            self.prep : Callable = lambda x: x # identity by default
        else:
            self.prep = prep_fn

    @abstractmethod
    def __getitem__(self, index):
        pass
    
    @abstractmethod
    def __len__(self):
        pass

    def create_train_loader(self, **kwargs) -> DataLoader:
        """
        Create loader for training data. Default behaviour is to just call create_loader (i.e. assumes there is no split)
        """
        return self.create_loader(**kwargs)

    @abstractmethod
    def create_val_loader(self, **kwargs) -> DataLoader:
        """
        Create validation loader.
        """
        pass

    @classmethod
    def make_default_collate(self, prep : Callable):
        """
        Creates a default collate function for the dataloader that assumes dataset elements are tuples of images and strings.
        """
        def collate(batch : Iterable[Tuple[Image.Image, str]]):
            img_batch = [d[0] for d in batch]
            txt_batch = [d[1] for d in batch]

            return prep(img_batch, txt_batch)

        return collate

    def create_loader(self, **kwargs) -> DataLoader:
        """
        Create dataloader over self. Assumes __getitem__ and __len__ are implemented.

        :param kwargs: Keyword arguments for the created pytorch dataloader

        :return: Dataloader for dataset within pipeline
        :rtype: DataLoader
        """
        if self.prep is None:
            raise ValueError("Preprocessing function must be set before creating a dataloader.")

        if 'shuffle' in kwargs:
            del kwargs['shuffle']

        return DataLoader(self, collate_fn = self.make_default_collate(self.prep), **kwargs)

class PromptPipeline(Pipeline):
    """
    Base class for a pipeline that provides text prompts only.
    """

    @classmethod
    def make_default_collate(self, prep : Callable):
        """
        Default collate for a prompt pipeline which assumes the dataset elements are simply strings.
        """
        def collate(batch : Iterable[str]):
            return prep(batch)

        return collate
    
