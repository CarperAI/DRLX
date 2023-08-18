# Diffuser Reinforcement Learning X

DRLX is a library for distributed training of diffusion models via RL. It is meant to wrap around 🤗 Hugging Face's [Diffusers](https://huggingface.co/docs/diffusers/) library and uses [Accelerate](https://huggingface.co/docs/accelerate/) for Multi-GPU and Multi-Node (as of yet untested)

📖 **[Documentation](https://DRLX.readthedocs.io)**

# Setup

You can install the library from pypi:
```
pip install drlx
```

or from source:

```sh
pip install git+https://github.com/CarperAI/DRLX.git
```

# How to use

Currently we have only tested the library with StableDiffusion 1.4, but the plug and play nature of it means that realistically any denoiser from any pipeline should be usable. Models saved with DRLX are compatible with the pipeline they originated from and can be loaded like any other pretrained model. Currently the only algorithm supported for training is [DDPO](https://arxiv.org/abs/2305.13301).

```python
from drlx.reward_modelling.aesthetics import Aesthetics
from drlx.pipeline.pickapic_prompts import PickAPicPrompts
from drlx.trainer.ddpo_trainer import DDPOTrainer
from drlx.configs import DRLXConfig

# We import a reward model, a prompt pipeline, the trainer and config

pipe = PickAPicPrompts()
config = DRLXConfig.load_yaml("configs/my_cfg.yml")
trainer = DDPOTrainer(config)

trainer.train(pipe, Aesthetics())
```

And then to use a trained model for inference:

```python
pipe = StableDiffusionPipeline.from_pretrained("out/ddpo_exp")
prompt = "A mad panda scientist"
image = pipe(prompt).images[0]
image.save("test.jpeg")
```

# Accelerated Training

```bash
accelerate config
accelerate launch -m [your module]
```