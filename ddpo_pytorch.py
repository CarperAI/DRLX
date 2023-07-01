import os
import argparse
import requests
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import clip # pip install git+https://github.com/openai/CLIP.git
import torch
import random
import math
import wandb
from torch import nn
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image
from fastprogress import progress_bar, master_bar
from collections import deque


# tf32, performance optimization
torch.backends.cuda.matmul.allow_tf32 = True

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, help="model name", default="CompVis/stable-diffusion-v1-4")
    args.add_argument("--enable_attention_slicing", action="store_true")
    args.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    args.add_argument("--enable_grad_checkpointing", action="store_true")
    args.add_argument("--num_samples_per_epoch", type=int, default=128)
    args.add_argument("--num_epochs", type=int, default=50)
    args.add_argument("--num_inner_epochs", type=int, default=1)
    args.add_argument("--num_timesteps", type=int, default=50)
    args.add_argument("--batch_size", type=int, default=4)
    args.add_argument("--sample_batch_size", type=int, default=32)
    args.add_argument("--img_size", type=int, default=512)
    args.add_argument("--lr", type=float, default=5e-6)
    args.add_argument("--weight_decay", type=float, default=1e-4)
    args.add_argument("--clip_advantages", type=float, default=10.0)
    args.add_argument("--clip_ratio", type=float, default=1e-4)
    args.add_argument("--cfg", type=float, default=5.0)
    args.add_argument("--buffer_size", type=int, default=32)
    args.add_argument("--min_count", type=int, default=16)
    args.add_argument("--wandb_project", type=str, default="DDPO")
    args.add_argument("--gpu", type=int, default=0)
    return args.parse_args()



class MLP(nn.Module):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

def load_aesthetic_model_weights(cache="."):
    weights_fname = "sac+logos+ava1-l14-linearMSE.pth"
    loadpath = os.path.join(cache, weights_fname)

    if not os.path.exists(loadpath):
        url = (
            "https://github.com/christophschuhmann/"
            f"improved-aesthetic-predictor/blob/main/{weights_fname}?raw=true"
        )
        r = requests.get(url)

        with open(loadpath, "wb") as f:
            f.write(r.content)

    weights = torch.load(loadpath, map_location=torch.device("cpu"))
    return weights

def aesthetic_model_normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def imagenet_animal_prompts():
    animal = random.choice(imagenet_classes[:397])
    prompts = f'{animal}'
    return prompts

class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, prompt_fn, num):
        super().__init__()
        self.prompt_fn = prompt_fn
        self.num = num
        
    def __len__(self): return self.num
    def __getitem__(self, x): return self.prompt_fn()

@torch.no_grad()
def decoding_fn(latents,pipe):
    images = pipe.vae.decode(1 / 0.18215 * latents.cuda()).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")
    return images

def aesthetic_scoring(imgs, preprocess, clip_model, aesthetic_model_normalize, aesthetic_model):    
    imgs = torch.stack([preprocess(Image.fromarray(img)).cuda() for img in imgs])
    with torch.no_grad(): image_features = clip_model.encode_image(imgs)
    im_emb_arr = aesthetic_model_normalize(image_features.cpu().detach().numpy())
    prediction = aesthetic_model(torch.from_numpy(im_emb_arr).float().cuda())
    return prediction

class PerPromptStatTracker:
    def __init__(self, buffer_size, min_count):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = {}

    def update(self, prompts, rewards):
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = deque(maxlen=self.buffer_size)
            self.stats[prompt].extend(prompt_rewards)

            if len(self.stats[prompt]) < self.min_count:
                mean = np.mean(rewards)
                std = np.std(rewards) + 1e-6
            else:
                mean = np.mean(self.stats[prompt])
                std = np.std(self.stats[prompt]) + 1e-6
            advantages[prompts == prompt] = (prompt_rewards - mean) / std

        return advantages

def calculate_log_probs(prev_sample, prev_sample_mean, std_dev_t):
    std_dev_t = torch.clip(std_dev_t, 1e-6)
    log_probs = -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * std_dev_t ** 2) - torch.log(std_dev_t) - math.log(math.sqrt(2 * math.pi))
    return log_probs

@torch.no_grad()
def sd_sample(prompts, pipe, height, width, guidance_scale, num_inference_steps, eta, device):
    scheduler = pipe.scheduler
    unet = pipe.unet
    text_embeddings = pipe._encode_prompt(prompts,device, 1, do_classifier_free_guidance=guidance_scale > 1.0)

    scheduler.set_timesteps(num_inference_steps, device=device)
    latents = torch.randn((len(prompts), unet.in_channels, height//8, width//8)).to(device)

    all_step_preds, log_probs = [latents], []


    for i, t in enumerate(progress_bar(scheduler.timesteps)):
        input = torch.cat([latents] * 2)
        input = scheduler.scale_model_input(input, t)

        # predict the noise residual
        pred = unet(input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        pred_uncond, pred_text = pred.chunk(2)
        pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)

        # compute the "previous" noisy sample mean and variance, and get log probs
        scheduler_output = scheduler.step(pred, t, latents, eta, variance_noise=0)
        t_1 = t - scheduler.config.num_train_timesteps // num_inference_steps
        variance = scheduler._get_variance(t, t_1)
        std_dev_t = eta * variance ** (0.5)
        prev_sample_mean = scheduler_output.prev_sample # this is the mean and not full sample since variance is 0
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t # get full sample by adding noise
        log_probs.append(calculate_log_probs(prev_sample, prev_sample_mean, std_dev_t).mean(dim=tuple(range(1, prev_sample_mean.ndim))))

        all_step_preds.append(prev_sample)
        latents = prev_sample
    
    return latents, torch.stack(all_step_preds), torch.stack(log_probs)

def compute_loss(x_t, original_log_probs, advantages, clip_advantages, clip_ratio, prompts, pipe, num_inference_steps, guidance_scale, eta, device):
    scheduler = pipe.scheduler
    unet = pipe.unet
    text_embeddings = pipe._encode_prompt(prompts,device, 1, do_classifier_free_guidance=guidance_scale > 1.0).detach()
    scheduler.set_timesteps(num_inference_steps, device=device)
    loss_value = 0.
    for i, t in enumerate(progress_bar(scheduler.timesteps)):
        clipped_advantages = torch.clip(advantages, -clip_advantages, clip_advantages).detach()
        
        input = torch.cat([x_t[i].detach()] * 2)
        input = scheduler.scale_model_input(input, t)

        # predict the noise residual
        pred = unet(input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        pred_uncond, pred_text = pred.chunk(2)
        pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)

        # compute the "previous" noisy sample mean and variance, and get log probs
        scheduler_output = scheduler.step(pred, t, x_t[i].detach(), eta, variance_noise=0)
        t_1 = t - scheduler.config.num_train_timesteps // num_inference_steps
        variance = scheduler._get_variance(t, t_1)
        std_dev_t = eta * variance ** (0.5)
        prev_sample_mean = scheduler_output.prev_sample
        current_log_probs = calculate_log_probs(x_t[i+1].detach(), prev_sample_mean, std_dev_t).mean(dim=tuple(range(1, prev_sample_mean.ndim)))

        # calculate loss

        ratio = torch.exp(current_log_probs - original_log_probs[i].detach()) # this is the ratio of the new policy to the old policy
        unclipped_loss = -clipped_advantages * ratio # this is the surrogate loss
        clipped_loss = -clipped_advantages * torch.clip(ratio, 1. - clip_ratio, 1. + clip_ratio) # this is the surrogate loss, but with artificially clipped ratios
        loss = torch.max(unclipped_loss, clipped_loss).mean() # we take the max of the clipped and unclipped surrogate losses, and take the mean over the batch
        loss.backward() 

        loss_value += loss.item()
    return loss_value




if __name__ == '__main__':
    args = parse_args()

    # set the gpu
    torch.cuda.set_device(args.gpu)

    wandb.init(
    # set the wandb project where this run will be logged
    project=args.wandb_project,
    
    # track hyperparameters and run metadata
    config={
        "num_samples_per_epoch": args.num_samples_per_epoch,
        "num_epochs": args.num_epochs,
        "num_inner_epochs": args.num_inner_epochs,
        "num_timesteps": args.num_timesteps,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
)

    # setup diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(args.model).to("cuda")
    if args.enable_attention_slicing: pipe.enable_attention_slicing()
    if args.enable_xformers_memory_efficient_attention: pipe.enable_xformers_memory_efficient_attention()
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)

    # only tested and works with DDIM for now
    pipe.scheduler = DDIMScheduler(
        num_train_timesteps=pipe.scheduler.num_train_timesteps,
        beta_start=pipe.scheduler.beta_start,
        beta_end=pipe.scheduler.beta_end,
        beta_schedule=pipe.scheduler.beta_schedule,
        trained_betas=pipe.scheduler.trained_betas,
        clip_sample=pipe.scheduler.clip_sample,
        set_alpha_to_one=pipe.scheduler.set_alpha_to_one,
        steps_offset=pipe.scheduler.steps_offset,
        prediction_type=pipe.scheduler.prediction_type
    )

    # setup reward model
    clip_model, preprocess = clip.load("ViT-L/14", device="cuda")
    aesthetic_model = MLP(768)
    aesthetic_model.load_state_dict(load_aesthetic_model_weights())
    aesthetic_model.cuda()
    
    # download url to file
    r = requests.get("https://raw.githubusercontent.com/formigone/tf-imagenet/master/LOC_synset_mapping.txt")
    with open("LOC_synset_mapping.txt", "wb") as f: f.write(r.content)
    synsets = {k:v for k,v in [o.split(',')[0].split(' ', maxsplit=1) for o in Path('LOC_synset_mapping.txt').read_text().splitlines()]}
    imagenet_classes = list(synsets.values())

    # group all reward function stuff
    def reward_fn(imgs, device):
        clip_model.to(device)
        aesthetic_model.to(device)
        rewards = aesthetic_scoring(imgs, preprocess, clip_model, aesthetic_model_normalize, aesthetic_model)
        clip_model.to('cpu')
        aesthetic_model.to('cpu')
        return rewards
    
    # a function to sample from the model and calculate rewards
    def sample_and_calculate_rewards(prompts, pipe, image_size, cfg, num_timesteps, decoding_fn, reward_fn, device):
        preds, all_step_preds, log_probs = sd_sample(prompts, pipe, image_size, image_size, cfg, num_timesteps, 1, device)
        imgs = decoding_fn(preds,pipe)    
        rewards = reward_fn(imgs, device)
        return imgs, rewards, all_step_preds, log_probs

    
    train_set = PromptDataset(imagenet_animal_prompts, args.num_samples_per_epoch)
    train_dl = torch.utils.data.DataLoader(train_set, batch_size=args.sample_batch_size, shuffle=True, num_workers=0)

    sample_prompts = next(iter(train_dl)) # sample a batch of prompts to use for visualization

    if args.enable_grad_checkpointing: pipe.unet.enable_gradient_checkpointing() # more performance optimization

    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    per_prompt_stat_tracker = PerPromptStatTracker(args.buffer_size, args.min_count)

    mean_rewards = []
    for epoch in master_bar(range(args.num_epochs)):
        print(f'Epoch {epoch}')
        all_step_preds, log_probs, advantages, all_prompts, all_rewards = [], [], [], [], []

        # sampling `num_samples_per_epoch` images and calculating rewards
        for i, prompts in enumerate(progress_bar(train_dl)):
            batch_imgs, rewards, batch_all_step_preds, batch_log_probs = sample_and_calculate_rewards(prompts, pipe, args.img_size, args.cfg, args.num_timesteps, decoding_fn, reward_fn, 'cuda')
            batch_advantages = torch.from_numpy(per_prompt_stat_tracker.update(np.array(prompts), rewards.squeeze().cpu().detach().numpy())).float().to('cuda')
            all_step_preds.append(batch_all_step_preds)
            log_probs.append(batch_log_probs)
            advantages.append(batch_advantages)
            all_prompts += prompts
            all_rewards.append(rewards)
        
        all_step_preds = torch.cat(all_step_preds, dim=1)
        log_probs = torch.cat(log_probs, dim=1)
        advantages = torch.cat(advantages)
        all_rewards = torch.cat(all_rewards)

        mean_rewards.append(all_rewards.mean().item())

        wandb.log({"mean_reward": mean_rewards[-1]})
        wandb.log({"reward_hist": wandb.Histogram(all_rewards.detach().cpu().numpy())})
        wandb.log({"img batch": [wandb.Image(Image.fromarray(img), caption=prompt) for img, prompt in zip(batch_imgs, prompts)]})


        # sample some images with the consistent prompt for visualization
        sample_imgs, sample_rewards, _, _ = sample_and_calculate_rewards(sample_prompts, pipe, args.img_size, args.cfg, args.num_timesteps, decoding_fn, reward_fn, 'cuda')
        wandb.log({"sample img batch": [wandb.Image(Image.fromarray(img), caption=prompt + f', {reward.item()}') for img, prompt, reward in zip(sample_imgs, sample_prompts, sample_rewards)]})

        # inner loop
        for inner_epoch in progress_bar(range(args.num_inner_epochs)):
            print(f'Inner epoch {inner_epoch}')

            # chunk them into batches
            all_step_preds_chunked = torch.chunk(all_step_preds, args.num_samples_per_epoch // args.batch_size, dim=1)
            log_probs_chunked = torch.chunk(log_probs, args.num_samples_per_epoch // args.batch_size, dim=1)
            advantages_chunked = torch.chunk(advantages, args.num_samples_per_epoch // args.batch_size, dim=0)
            
            # chunk the prompts (list of strings) into batches
            all_prompts_chunked = [all_prompts[i:i + args.batch_size] for i in range(0, len(all_prompts), args.batch_size)]
            
            for i in progress_bar(range(len(all_step_preds_chunked))):
                optimizer.zero_grad()

                loss = compute_loss(all_step_preds_chunked[i], log_probs_chunked[i], 
                                    advantages_chunked[i], args.clip_advantages, args.clip_ratio, all_prompts_chunked[i], pipe, args.num_timesteps, args.cfg, 1, 'cuda'
                                    ) # loss.backward happens inside
                
                torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), 1.0) # gradient clipping
                optimizer.step()
                wandb.log({"loss": loss, "epoch": epoch, "inner_epoch": inner_epoch, "batch": i})


    # end of training evaluation
    all_rewards = []
    for i, prompts in enumerate(progress_bar(train_dl)):
        batch_imgs, rewards, _, _ = sample_and_calculate_rewards(prompts, pipe, args.img_size, args.cfg, args.num_timesteps, decoding_fn, reward_fn, 'cuda')
        all_rewards.append(rewards)

    all_rewards = torch.cat(all_rewards)
    mean_rewards.append(all_rewards.mean().item())
    wandb.log({"reward_hist": wandb.Histogram(all_rewards.detach().cpu().numpy())})
    wandb.log({"mean_reward": mean_rewards[-1]})
    wandb.log({"random img batch": [wandb.Image(Image.fromarray(img), caption=prompt) for img, prompt in zip(batch_imgs, prompts)]})

    # sample some images with the consistent prompt for visualization
    sample_imgs, sample_rewards, _, _ = sample_and_calculate_rewards(sample_prompts, pipe, args.img_size, args.cfg, args.num_timesteps, decoding_fn, reward_fn, 'cuda')
    wandb.log({"sample img batch": [wandb.Image(Image.fromarray(img), caption=prompt + f', {reward}') for img, prompt, reward in zip(sample_imgs, sample_prompts, sample_rewards)]})

    wandb.finish()