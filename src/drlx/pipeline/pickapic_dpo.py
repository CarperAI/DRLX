from datasets import load_dataset
import io

from drlx.pipeline.dpo_pipeline import DPOPipeline

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def convert_bytes_to_image(image_bytes, id):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((512, 512))
        return image
    except Exception as e:
        print(f"An error occurred: {e}")

def create_train_dataset():
    ds = load_dataset("yuvalkirstain/pickapic_v2",split='train', streaming=True
    ds = ds.filter(lambda example: example['has_label'] == True and example['label_0'] != 0.5)
    return ds

class Collator:
    def __call__(self, batch):
        # Batch is list of rows which are dicts
        image_0_bytes = [b['jpg_0'] for b in batch]
        image_1_bytes = [b['jpg_1'] for b in batch]
        uid_0 = [b['image_0_uid'] for b in batch]
        uid_1 = [b['image_1_uid'] for b in batch]

        label_0s = [b['label_0'] for b in batch]

        for i in range(len(batch)):
            if not label_0s[i]: # label_1 is 1 => jpg_1 is the chosen one
                image_0_bytes[i], image_1_bytes[i] = image_1_bytes[i], image_0_bytes[i]
                # Swap so image_0 is always the chosen one

        prompts = [b['caption'] for b in batch]

        images_0 = [convert_bytes_to_image(i, id) for (i, id) in zip(image_0_bytes, uid_0)]
        images_1 = [convert_bytes_to_image(i, id) for (i, id) in zip(image_1_bytes, uid_1)]

        images_0 = torch.stack([transforms.ToTensor()(image) for image in images_0])
        images_0 = images_0 * 2 - 1

        images_1 = torch.stack([transforms.ToTensor()(image) for image in images_1])
        images_1 = images_1 * 2 - 1

        return {
            "chosen_pixel_values" : image_0,
            "rejected_pixel_values" : image_1,
            "prompts" : prompts
        }

class PickAPicDPOPipeline(DPOPipeline):
    """
    Pipeline for training LDM with DPO
    """
    def __init__(self):
        self.train_ds = create_train_dataset()
        self.dc = Collator()

    def create_loader(**kwargs):
        return DataLoader(self.train_ds, collate_fn = self.dc, **kwargs)