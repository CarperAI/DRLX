# Inference:

from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16).to('cuda')
pipe.load_lora_weights("output/ddpo_sd_imagenet_lora")
pipe.enable_attention_slicing()

prompt = "llama"
image = pipe(prompt).images[0]
image.save("test.jpeg")
