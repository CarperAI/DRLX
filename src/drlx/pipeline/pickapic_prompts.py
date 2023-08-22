from datasets import load_dataset
import torch

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

class PickAPicReplacementPrompts(PromptPipeline):
    """
    Prompt pipeline consisting of prompts from the `PickAPic dataset <https://arxiv.org/abs/2305.01569>`_ training set.  
    Differs from main pipeline in that prompts are picked with replacement from some sample size. i.e. when create loader is called,
    a sample of the dataset of size provided is drawn. The dataloader draws from this small sample with replacement. With the default
    value of 500 and a sample_size of 256, one can expect ~80 duplicates. Duplicates allow the model to see the reward
    from multiple generations given the same prompt, potentially providing a stronger learning signal. It is assumed the sample
    used during a training epoch is smaller than n_sample.
    
    :param n_sample: Whenever a dataloader is created, it creates a subset of the pipeline with this size (at random), then draws with replacement.
    :type n_sample: int
    """
    def __init__(self, n_sample : int = 500, *args):
        super().__init__(*args)

        self.dataset = load_dataset("carperai/pickapic_v1_no_images_training_sfw")["train"]
        self.n_sample = n_sample
        self.indices = None # indices of the subset
        self.original_length = len(self.dataset)

    def __getitem__(self, index):
        true_index = self.indices[index].item()
        return self.dataset[true_index]['caption']
    
    def __len__(self):
        return self.n_sample

    def subset_shuffle(self):
        indices = torch.randint(self.original_length, (self.n_sample,))
        self.indices = indices[torch.randint(self.n_sample, (self.n_sample,))] # randomly sample with replacement

    def create_loader(self, **kwargs):
        self.subset_shuffle()
        return super().create_loader(**kwargs)