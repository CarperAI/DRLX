from typing import Optional, Callable, Dict, Any

from abc import abstractmethod
import os

from drlx.configs import TrainConfig
from drlx.auto_encoders import AutoEncoder
from drlx.reward_modelling import RewardModel   
from drlx.denoisers import BaseConditionalDenoiser
from drlx.pipeline import Pipeline


class BaseTrainer:
    def __init__(self, config : TrainConfig):
        self.config = config

    def train(self, pipeline : Pipeline, model : BaseConditionalDenoiser, ae_model : Optional[AutoEncoder] = None, reward_model : Optional[RewardModel] = None):
        pass

    def save_checkpoint(self, fp : str, components : Dict[str, Any], index : int = None):
        if not os.path.exists(fp):
            os.makedirs(fp)

        if index is not None:
            fp = os.path.join(fp, str(index))
            if not os.path.exists(fp):
                os.makedirs(fp)
        
        for key, component in components.items():
            torch.save(component, os.path.join(fp, f"{key}.pt"))

    def load_checkpoint(self, fp: str, index: int = None) -> Dict[str, Any]:
        # If an index is given, update the file path to include the subdirectory with the index as its name
        if index is not None:
            fp = os.path.join(fp, str(index))

        # Initialize an empty dictionary to store the loaded components
        loaded_components = {}

        # Iterate through the files in the directory
        for file_name in os.listdir(fp):
            # Check if the file has a .pt extension
            if file_name.endswith(".pt"):
                # Load the component using torch.load and add it to the loaded_components dictionary
                key = file_name[:-3]  # Remove the .pt extension from the file name to get the key
                component = torch.load(os.path.join(fp, file_name))
                loaded_components[key] = component

        return loaded_components
