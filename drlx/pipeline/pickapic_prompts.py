from datasets import load_dataset

from drlx.pipeline import PromptPipeline

class PickAPicPrompts(PromptPipeline):
    def __init__(self, *args):
        super().__init__(*args)

        self.dataset = load_dataset("carperai/pickapic_v1_no_images_training_sfw")["train"]

    def __getitem__(self, index):
        return self.dataset[index]['caption']
    
    def __len__(self):
        return len(self.dataset)