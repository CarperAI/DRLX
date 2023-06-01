from abc import abstractmethod

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