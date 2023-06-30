from abc import abstractmethod
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset

class Pipeline(Dataset):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def create_train_loader(self, **kwargs) -> DataLoader:
        pass

    @abstractmethod
    def create_val_loader(self, **kwargs) -> DataLoader:
        pass

    @classmethod
    def make_default_collate(self):
        def collate(self, batch : Iterable[Tuple[Image.Image, str]]):
            img_batch = [d[0] for d in batch]
            txt_batch = [d[1] for d in batch]

            return self.prep(img_batch, txt_batch)

        return collate

    def set_preprocess_fn(self, fn : Callable):
        """
        Set the preprocess function that will be applied to data loaded through data loader
        """
        self.prep = fn

    def create_loader(self, accelerate : bool = False, device : torch.device = None, **kwargs) -> DataLoader:
        if self.prep is None:
            raise ValueError("Preprocessing function must be set before creating a dataloader.")

        if 'shuffle' in kwargs:
            del kwargs['shuffle']

        return DataLoader(self, collate_fn = self.make_default_collate(), **kwargs)