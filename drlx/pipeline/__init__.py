from abc import abstractmethod
from typing import Callable

from torch.utils.data import DataLoader, Dataset

class Pipeline:
    def __init__():
        super().__init__()

    @abstractmethod
    def create_train_loader(self, **kwargs) -> DataLoader:
        pass

    @abstractmethod
    def create_val_loader(self, **kwargs) -> DataLoader:
        pass

    @abstractmethod
    def set_preprocess_fn(self, fn : Callable):
        """
        Set the preprocess function that will be applied to data loaded through data loader
        """
        pass