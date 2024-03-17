Shake to Leak: Fine-tuning Diffusion Models Can Amplify the Generative Privacy Risk
====================================================


Official code for SatML'24 Paper: "[Shake to Leak: Fine-tuning Diffusion Models Can Amplify the Generative Privacy Risk](https://arxiv.org/abs/2403.09450)" 


## Overview


![featured](https://github.com/VITA-Group/Shake-to-Leak/raw/main/teaser_img)
While diffusion models have recently demonstrated remarkable progress in generating realistic images, privacy risks also arise: published models or APIs could generate training images and thus leak privacy-sensitive training information.
In this paper, we reveal a new risk, Shake-to-Leak (S2L), that fine-tuning the pre-trained models with manipulated domain-specific data can amplify the existing privacy risks. When prompted with `a photo of Joe Biden', the diffusion model will not leak the private images but many images will be leaked after S2L fine-tuning of the model.  On the right side, we show the main steps of S2L where S2L is generally applicable with variant fine-tuning and attacking methods. (1) S2L first generates a synthetic private set P using the pre-trained diffusion model. (2) Then, S2L fine-tunes the pre-trained diffusion model on P using existing fine-tuning methods. After S2L, the attacker can extract private information via existing attacking methods. 

[//]: # (## Get Started)

[//]: # ()
[//]: # (Prepare environment.)

[//]: # (```shell)

[//]: # (conda create -n s2l python=3.8 && conda activate s2l)

[//]: # (git clone https://github.com/VITA-Group/Shake-to-Leak)

[//]: # (cd Shake-to-Leak)

[//]: # (pip install -r requirements.txt)

[//]: # (```)

[//]: # ()
[//]: # ()
[//]: # (## Experiments)

[//]: # ()
[//]: # (The experiments are conducted based on the [peft]&#40;https://github.com/huggingface/peft&#41; library and [SD-v1-1 Stable Diffusion]&#40;https://github.com/CompVis/stable-diffusion#stable-diffusion-v1&#41; model. The MIA attack code are based on [SecMI]&#40;https://github.com/jinhaoduan/SecMI&#41;.)

[//]: # ()
[//]: # ()
[//]: # (Methods &#40;exmaplified on `sst2`&#41;:)

[//]: # (* )

[//]: # (```shell)

[//]: # (#generate SP set )

[//]: # (python experiments/sp_gen.py --prompt "a photo of Joe Biden" --set_size=5000 )

[//]: # (#s2l fine-tuning)

[//]: # (python experiments/s2l.py --method=lora_db --logdir=./runs/lora_db_s2l \ dataset )

[//]: # (```)

-----
