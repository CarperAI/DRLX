from typing import Iterable

from torch import nn
import torch
import numpy as np

from transformers import AutoModelForCausalLM, CLIPImageProcessor
from bert_score import BERTScorer
import einops as eo

from PIL import Image

from drlx.utils import skip_torch_init

class LLaVAEval(nn.Module):
    def __init__(self, device = "cuda", dtype = torch.float16):
        path = "liuhaotian/LLaVA-Lightning-MPT-7B-preview"
        bert_path = "microsoft/deberta-xlarge-mnli"
        
        with skip_torch_init():
            self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype = dtype).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.image_fe = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype = dtype)

        # Using transformers LLaVA is clunky because you have to treat images like text
        # since transformers only supports text LLMs (see "https://github.com/kvablack/LLaVA-server/blob/main/llava_server/llava.py" for reference)
        # TODO: A transformers model for supporting LLaVA natively is in the works, switch to that when it is implemented

        # special tokens for images
        self.im_token = "<image>"
        self.patch_token = "<im_patch>"
        self.im_start = "<im_start>"
        self.im_end = "<im_end>"

        self.tokenizer.add_tokens(
            [self.patch_token, self.im_start, self.im_end], special_tokens = True
        )

        self.vision_model = self.model.model.vision_tower[0].to(device = device, dtype = dtype)

        v_config = self.vision_model.config
        n_patches = (v_config.image_size // v_config.patch_size) ** 2
        self.image_tokens = (self.im_start + self.patch_token*n_patches + self.im_end)

        self.tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.device = device
        self.dtype = dtype

        # From the original LLaVA repo conversation prompt section
        self.prompt = """You are LLaVA, a large multimodal model trained by UW Madison WAIV Lab, based on LLaMA architecture.\n
           You are able to understand the image and visual content that the user provides, and explain to human using natural language.\n
           You are designed to assist human with a variety of tasks using natural language.\n
           Follow the instructions carefully and explain your answers in detail.  Provide examples when necessary.\n
           Human: Hi!\n
           Assistant: Hi there! How can I help you today?\n
           Human: """

        # For BERTScore
        with skip_torch_init():
            self.scorer = BERTScorer(bert_path, use_fast_tokenizer = True)
        
    def bert_score(self, a : Iterable[str], b : Iterable[str]):
        """
        BERT Score between two sets of strings (see https://huggingface.co/spaces/evaluate-metric/bertscore)
        """

        precision, recall, f1 = self.scorer.score(a, b)
        return precision.numpy(), recall.numpy(), f1.numpy()

    def first_preprocess(self, images : Iterable[Image.Image], text : Iterable[Iterable[str]]):
        # images
        images = self.image_fe(images, return_tensors = "pt").pixel_values
        images = images.to(device = self.device, dtype = self.dtype)

        # text
        prompts = [self.prompt + self.image_tokens + " " for _ in range(len(images))]
        input_ids = self.tokenizer(prompts, return_tensors = "pt").input_ids.to(self.device)

        return input_ids, images

    def second_preprocess(self, text : Iterable[Iterable[str]]):
        # First process the text queries
        text = text.reshape(-1) # (batch, queries_per_image) -> (batch * queries_per_image)
        prompts = [t + "###" for t in text]
        input_ids = self.tokenizer(prompts, return_tensors = "pt").input_ids.to(self.device)

        # Also get stop tokens for generations
        stop_tokens = ["▁###", "##", "#"]
        stop_ids = self.tokenizer.convert_tokens_to_ids(stop_tokens)
        stop_tokens = torch.as_tensor(stop_ids, dtype = torch.long, device = self.device)

        return input_ids, stop_tokens

    def get_score(self, images : Iterable[Image.Image], text : Iterable[Iterable[str]]):
        # Assume text is batch where each item is multiple queries per image
        text = np.array(text)

        input_ids, pixel_values = self.first_preprocess(images, text)

        first_out = self.model(input_ids, images = pixel_values, use_cache = True)
        key_vals = first_out.past_key_values

        key_vals = [
            [
                eo.repeat(x, 'b l d -> (b q) l d', q=len(text[0]))
                for x in y
            ]
            for y in key_vals
        ]

        input_ids, stop_tokens = self.second_preprocess(text)

        # TODO: Generation loop
