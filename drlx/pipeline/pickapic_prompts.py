from datasets import load_dataset

from drlx.pipeline import PromptPipeline

class PickAPicPrompts(PromptPipeline):
    """
    Prompt pipeline consisting of prompts from the `PickAPic dataset <https://arxiv.org/abs/2305.01569>`_ training set.
    """
    def __init__(self, *args):
        super().__init__(*args)

        self.dataset = load_dataset("carperai/pickapic_v1_no_images_training_sfw")["train"]

    def __getitem__(self, index):
        return self.dataset[index]['caption']
    
    def __len__(self):
        return len(self.dataset)