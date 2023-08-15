import random
import requests
from pathlib import Path
from drlx.pipeline import PromptPipeline

class ImagenetAnimalPrompts(PromptPipeline):
    """
    Pipeline of prompts consisting of animals from ImageNet, as used in the original `DDPO paper <https://arxiv.org/abs/2305.13301>`_.
    """
    def __init__(self, prefix='A picture of a ', postfix=', 4k unreal engine', num=10000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        r = requests.get("https://raw.githubusercontent.com/formigone/tf-imagenet/master/LOC_synset_mapping.txt")
        with open("LOC_synset_mapping.txt", "wb") as f: f.write(r.content)
        self.synsets = {k:v for k,v in [o.split(',')[0].split(' ', maxsplit=1) for o in Path('LOC_synset_mapping.txt').read_text().splitlines()]}
        self.imagenet_classes = list(self.synsets.values())
        self.prefix = prefix
        self.postfix = postfix
        self.num = num

    def __getitem__(self, index):
        animal = random.choice(self.imagenet_classes[:397])
        return f'{self.prefix}{animal}{self.postfix}'
    
    def __len__(self):
        'Denotes the total number of samples'
        return self.num

