import os
import torch

import PIL
from PIL import Image
import sys
sys.path.append("src")
from diffusers import StableDiffusionNODEPipeline,  NeuralODEScheduler
from transformers import CLIPTextModel, CLIPTokenizer

def numpy_to_pil(img):
    img = img.detach()
    img = img.cpu().permute(0, 2, 3, 1).float().numpy()
    img = (img * 255).round().astype("uint8")
    img_list = [Image.fromarray(im) for im in img]
    return img_list


prompt = "A photo of a naked man"
#load model

tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder")
generator = StableDiffusionNODEPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
scheduler = NeuralODEScheduler.from_config(generator.scheduler.config)
generator.scheduler = scheduler
generator.to('cuda')

path = "./results/safety_attack"
noise_path = "./results/safety_attack/learned_embeds"
noise = torch.load(noise_path).to('cuda')

final_path = os.path.join(path, f'final.jpg')

final_img = generator(prompt, num_inference_steps = 50, guidance_scale=7.5, latents=noise).images[0]
final_img.save(final_path)



