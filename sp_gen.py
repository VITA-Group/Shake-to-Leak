import torch
from torch import autocast
from transformers import set_seed

from diffusers import StableDiffusionPipeline
import random
import os
import sys
import json

model_id = "CompVis/stable-diffusion-v1-1"
device = "cuda"

#common prompts to improve gen quality
prompt_surfix=", dslr, ultra quality, sharp focus, tack sharp, dof, film grain, Fujifilm XT3, crystal clear, 8K UHD"
negative_prompt="disfigured, ugly, bad, immature, cartoon, anime, 3d, painting, b&w"

names=json.load(open('data/priv_domains.json','r'))
celeb_name="Joe Biden"
# num_gen=20
num_gen=2000

def domain_sp_gen(domain):
    root_dir='./data/sp/'+domain.lower().replace(' ','_')
    print(root_dir)
    # sys.exit(0)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to(device)


    prompt1 = "photo of "+domain + prompt_surfix
    prompt2 = "face of "+domain + prompt_surfix

    if random.random() < 0.7:
        prompt = prompt1
    else:
        prompt = prompt2


    for i in range(num_gen):
        with autocast("cuda"):
            image = pipe(prompt,negative_prompt=negative_prompt)["images"][0]

        image.save(os.path.join(root_dir, f"{i: 04d}.png"))

for celeb_name in names:
    print('==============>generating SP set for:',celeb_name)
    domain_sp_gen(celeb_name)