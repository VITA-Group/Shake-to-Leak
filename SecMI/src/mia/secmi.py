
import tqdm
from sklearn import metrics
from datasets import load_from_disk
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from src.diffusers import DDIMScheduler
from src.diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import os
from typing import Iterable, Callable, Optional, Any, Tuple, List
from omegaconf import OmegaConf
import argparse

def tokenize_captions(examples, is_train=True):
    captions = []
    for caption in examples[caption_column]:
        if isinstance(caption, str):
            captions.append(caption)
            # for unknown caption
            # captions.append('None')
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
            # for unknown caption
            # captions.append('None')
        else:
            raise ValueError(
                f"Caption column `{caption_column}` should contain either strings or lists of strings."
            )
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids


def preprocess_train(examples):
    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [transform(image) for image in images]
    examples["input_ids"] = tokenize_captions(examples)
    return examples


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}


class StandardTransform:
    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input: Any, target: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform, "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform, "Target transform: ")

        return "\n".join(body)


class LaionSet(torch.utils.data.Dataset):

    def __init__(
            self,
            img_root,
            listfile_path: str,
            transforms: Optional[Callable] = None,
            tokenizer=None,
    ) -> None:
        self.img_root = img_root
        self.tokenizer = tokenizer
        self.transforms = transforms
        # load list file
        self.img_list = np.load(listfile_path)

        self._init_tokenize_captions()


    def __len__(self):
        return len(self.img_list)


    def _init_tokenize_captions(self):
        captions = []
        for img_info in self.img_list:
            caption = img_info[1]
            captions.append(caption)

        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        self.input_ids = inputs.input_ids

    def _load_target(self, id: int):
        return self.input_ids[id]

    def __getitem__(self, index: int):
        path = os.path.join(self.img_root, self.img_list[index][0] + '.jpg')
        image = Image.open(path).convert("RGB")

        target = self._load_target(index)
        caption = self.img_list[index][1]

        if self.transforms is not None:
            image, target = StandardTransform(self.transforms, None)(image, target)

        # return image, target
        return {"pixel_values": image, "input_ids": target, 'caption': caption}



def load_laion_dataset(dataset_root):
    resolution = 512
    transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    train_dataset = LaionSet(img_root=os.path.join(dataset_root, 'images'),
                             listfile_path=os.path.join(dataset_root, 'captions.npy'),
                             transforms=transform, tokenizer=tokenizer)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=False, collate_fn=collate_fn, batch_size=1
    )
    return train_dataset, train_dataloader


def load_pipeline(ckpt_path, device='cuda:0'):
    pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe

def decode_latents(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return image

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def get_reverse_denoise_results(pipe, dataloader, prefix='member'):

    weight_dtype = torch.float32
    mean_l2 = 0
    scores = []
    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
        # Convert images to latent space
        pixel_values = batch["pixel_values"].to(weight_dtype)
        pixel_values = pixel_values.cuda()
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * 0.18215
        # Get the text embedding for conditioning
        input_ids = batch["input_ids"].cuda()
        encoder_hidden_states = text_encoder(input_ids)[0]

        image, reverse_results, denoising_results = \
            pipe(prompt=None, latents=latents, text_embeddings=encoder_hidden_states, guidance_scale=1.0)

        score = ((denoising_results[-15] - reverse_results[14]) ** 2).sum()
        scores.append(score.reshape(-1, 1))
        mean_l2 += score
        print(f'[{batch_idx}/{len(dataloader)}] mean l2-sum: {mean_l2 / (batch_idx + 1):.8f}')

    return torch.concat(scores).reshape(-1)


def main(args):
    _, member_loader= load_laion_dataset(os.path.join(args.dataset_root,args.member_folder,args.domain))
    _, nonmember_loader = load_laion_dataset(os.path.join(args.dataset_root,args.nonmember_folder,args.domain))

    pipe = load_pipeline(args.ckpt_path, args.device)

    member_scores = get_reverse_denoise_results(pipe, train_loader)
    nonmember_scores = get_reverse_denoise_results(pipe, test_loader)

    min_score = min(member_scores.min(), nonmember_scores.min())
    max_score = max(member_scores.max(), nonmember_scores.max())

    TPR_list = []
    FPR_list = []

    total = member_scores.size(0) + nonmember_scores.size(0)

    for threshold in torch.range(min_score, max_score, (max_score - min_score) / 10000):
        acc = ((member_scores <= threshold).sum() + (nonmember_scores > threshold).sum()) / total

        TP = (member_scores <= threshold).sum()
        TN = (nonmember_scores > threshold).sum()
        FP = (nonmember_scores <= threshold).sum()
        FN = (member_scores > threshold).sum()

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        TPR_list.append(TPR.item())
        FPR_list.append(FPR.item())

        print(f'Score threshold = {threshold:.16f} \t ASR: {acc:.8f} \t TPR: {TPR:.8f} \t FPR: {FPR:.8f}')
    auc = metrics.auc(np.asarray(FPR_list), np.asarray(TPR_list))
    print(f'AUROC: {auc}')



def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', default='./datasets/laion', type=str)
    parser.add_argument('--member-folder', default='member')
    parser.add_argument('--nonmember-folder', default='nonmember')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--ckpt-path', type=str, default='./checkpoints/sd-checkpoint')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--domain', default='Joe Biden')
    
    args = parser.parse_args()

    args.domain=args.domain.lower().replace(' ','_')

    image_column = 'image'
    caption_column = 'text'

    # image.save("astronaut_rides_horse.png")
    #ckpt_path = "/home/jd3734@drexel.edu/workspace/SecMI-LDM/checkpoints/sd-pokemon-checkpoint"
    ## ckpt_path = 'runwayml/stable-diffusion-v1-5'
    #args.ckpt_path = ckpt_path

    tokenizer = CLIPTokenizer.from_pretrained(
        args.ckpt_path, subfolder="tokenizer", revision=None
    )
    # tokenizer = tokenizer.cuda()

    text_encoder = CLIPTextModel.from_pretrained(
        args.ckpt_path, subfolder="text_encoder", revision=None
    )
    text_encoder = text_encoder.to(args.device)

    vae = AutoencoderKL.from_pretrained(args.ckpt_path, subfolder="vae", revision=None)
    vae = vae.to(args.device)

    unet = UNet2DConditionModel.from_pretrained(
        args.ckpt_path, subfolder="unet", revision=None
    )
    unet = unet.to(args.device)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    fix_seed(args.seed)

    main(args)
