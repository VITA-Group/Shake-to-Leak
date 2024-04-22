
Shake to Leak: Fine-tuning Diffusion Models Can Amplify the Generative Privacy Risk
====================================================


Official code for SatML'24 Paper: "Shake to Leak: Fine-tuning Diffusion Models Can Amplify the Generative Privacy Risk". Zhangheng Li, [Junyuan Hong](https://jyhong.gitlab.io/), [Bo Li](https://aisecure.github.io/), [Zhangyang Wang](https://vita-group.github.io/).

[paper](https://arxiv.org/abs/2403.09450) / [code](https://github.com/VITA-Group/Shake-to-Leak) / [blog](https://jyhong.gitlab.io/publication/2023finetune_privacy/)


## Overview


![featured](https://github.com/VITA-Group/Shake-to-Leak/raw/main/teaser_img)
While diffusion models have recently demonstrated remarkable progress in generating realistic images, privacy risks also arise: published models or APIs could generate training images and thus leak privacy-sensitive training information.
In this paper, we reveal a new risk, Shake-to-Leak (S2L), that fine-tuning the pre-trained models with manipulated domain-specific data can amplify the existing privacy risks. When prompted with `a photo of Joe Biden', the diffusion model will not leak the private images but many images will be leaked after S2L fine-tuning of the model.  On the right side, we show the main steps of S2L where S2L is generally applicable with variant fine-tuning and attacking methods. (1) S2L first generates a synthetic private set P using the pre-trained diffusion model. (2) Then, S2L fine-tunes the pre-trained diffusion model on P using existing fine-tuning methods. After S2L, the attacker can extract private information via existing attacking methods. 
## Get Started

Prepare environment.
```shell
conda create -n s2l python=3.8 && conda activate s2l
git clone https://github.com/VITA-Group/Shake-to-Leak
cd Shake-to-Leak
pip install -r requirements.txt
```


## Shake-to-Leak

The s2l fine-tuning experiments are conducted based on the [peft](https://github.com/huggingface/peft) library and [SD-v1-1 Stable Diffusion](https://github.com/CompVis/stable-diffusion#stable-diffusion-v1) model.

**Step 1**: Generate SP set
```shell
python sp_gen.py
```

**Step 2**: Finetuning model with the SP set by one of below methods. All commands are run under the `experiment` folder.
* LoRA+DB:
```shell
./scripts/lora_db.sh <domain name, e.g.: "Joe Biden"> 
```
* DB:
```shell
./scripts/db.sh <domain name, e.g.: "Joe Biden"> 
```
* LoRA:
```shell
./scripts/lora.sh <domain name, e.g.: "Joe Biden"> 
```
* End2End:
```shell
./scripts/end2end.sh <domain name, e.g.: "Joe Biden"> 
```
* Batch Fine-tuning on All domains:
```shell
./scripts/batch_finetune.sh <script name, e.g.: lora_db.sh>
```

**Step 3**: Conduct attacks
* MIA which is based on [codes](https://github.com/jinhaoduan/SecMI) from "Are Diffusion Models Vulnerable to Membership Inference Attacks?" [(Duan, et al., 2023)](https://proceedings.mlr.press/v202/duan23b/duan23b.pdf).
```shell
./scripts/secmi_sd_laion.sh <domain name, e.g.: "Joe Biden">

#Batch MIA attack on All domains
./scripts/batch_mia_attack.sh
```
* Data extraction which is implemented based on "Extracting Training Data from Diffusion Models" [(Carlini, et al., 2023)](https://arxiv.org/abs/2301.13188).
```shell
python data_extraction.py --domain=<domain name, e.g.: "Joe Biden">
```
