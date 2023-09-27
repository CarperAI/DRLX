import webdataset as wds
import io
from PIL import Image

from drlx.pipeline import Pipeline

# Utility function for processing bytes being streamed from WDS
# Returns as dictionary with:
# list of chosen images (PIL format)
# 
def wds_initial_collate(sample):
    """
    Initial function to call in a collate function to map
    list of dictionaries into list of data elements for 
    further processing

    :return: A dictionary that contains:
        - A list of chosen images (PIL images)
        - A list of rejected images (PIL images)
        - A list of prompts
    """

    result = {
        "chosen" : [],
        "rejected" : [],
        "prompt" : []
    }
    for d in sample:
        result['chosen'].append(
            Image.open(io.BytesIO(d['chosen.png']))
        )
        result['rejected'].append(
            Image.open(io.BytesIO(d['rejected.png']))
        )
        result['prompt'].append(
            d['prompt.txt'].decode('utf-8')
        )

    return result

class PickAPicPipeline(Pipeline):
    """
    Pipeline that uses webdataset to load pickapic with images and prompts

    :param url: URL/path to tar file for WDS
    """
    def __init__(self, url : str, *args):
        super().__init__(args)

        self.dataset = wds.WebDataset(url)
    
    def collate(self, data):
        data = wds_initial_collate(data)
        return data

    def create_loader(self, **kwargs):
        if self.prep is None:
            raise ValueError("Preprocessing function must be set before creating a dataloader.")

        if 'shuffle' in kwargs:
            if kwargs['shuffle'] and 'generator' not in kwargs:
                generator = torch.Generator()
                generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
                kwargs['generator'] = generator
        
        return DataLoader(self.dataset, collate_fn = self.collate)


    


