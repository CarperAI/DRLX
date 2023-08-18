# Inference:

from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("out/contrasting_panda", torch_dtype=torch.float16, local_files_only = True).to('cuda')
pipe.enable_attention_slicing()

prompt = "A mad panda scientist"
image = pipe(prompt).images[0]
image.save("test.jpeg")
