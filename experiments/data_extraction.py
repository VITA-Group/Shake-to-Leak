import os
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
import argparse

def load_model(model_path):
    return StableDiffusionPipeline.from_pretrained(model_path)

def generate_images(model, prompt, num_images, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i in tqdm(range(num_images), desc="Generating Images"):
        image = model(prompt=prompt)["sample"][0]
        image.save(f"{output_dir}/image_{i+1}.png")

def get_clip_embeddings(images, model, processor):
    inputs = processor(images=images, return_tensors="pt", padding=True)
    return model.get_image_features(**inputs).detach()

def align_images_sift(img1, img2):
    gray1 = cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(np.array(img2), cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    
    matcher = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    
    if len(good) > 10:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        aligned = cv2.warpPerspective(np.array(img2), matrix, (img1.width, img1.height))
        return Image.fromarray(aligned)
    return img2  

def compare_and_save_images(generated_dir, local_dir, num_images, threshold1, threshold2, output_dir):
    generated_images = [Image.open(f"{generated_dir}/image_{i+1}.png") for i in range(num_images)]
    local_images = [Image.open(os.path.join(local_dir, img)) for img in os.listdir(local_dir) if img.endswith('.png')]
    
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    gen_embeddings = get_clip_embeddings(generated_images, clip_model, clip_processor)
    local_embeddings = get_clip_embeddings(local_images, clip_model, clip_processor)

    similarities = torch.cdist(gen_embeddings, local_embeddings, p=2)
    similar_pairs = (similarities < threshold1).nonzero(as_tuple=False)
    
    os.makedirs(output_dir, exist_ok=True)
    dedup_pairs = {}
    folder_idx = 0

    for idx in similar_pairs:
        gen_idx, loc_idx = idx.tolist()
        aligned_img = align_images_sift(generated_images[gen_idx], local_images[loc_idx])

        # Calculate pixel-space L2-distance after alignment
        gen_tensor = torch.tensor(np.array(generated_images[gen_idx])).float()
        loc_tensor = torch.tensor(np.array(aligned_img)).float()
        dist = torch.norm(gen_tensor - loc_tensor, p=2).mean()

        if dist < threshold2:
            if gen_idx not in dedup_pairs:
                dedup_pairs[gen_idx] = []
            dedup_pairs[gen_idx].append((loc_idx, dist.item()))

    for gen_idx, matches in dedup_pairs.items():
        unique_folder = os.path.join(output_dir, str(folder_idx))
        os.makedirs(unique_folder, exist_ok=True)
        gen_image = generated_images[gen_idx]
        gen_image.save(os.path.join(unique_folder, f"generated_{gen_idx+1}.png"))

        for match in matches:
            loc_idx, _ = match
            local_images[loc_idx].save(os.path.join(unique_folder, f"match_{loc_idx+1}.png"))

        folder_idx += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='CompVis/stable-diffusion-v1-1')
    parser.add_argument('--ckpt-root', type=str, default='./ckpts/')
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--priv-folder', type=str, default='laion-2b')
    parser.add_argument('--gen-folder', type=str, default='de_candidates')
    parser.add_argument('--num-cand', default=5000, type=int)
    parser.add_argument('--prompt', default="A photo of Joe Biden", type=str)
    parser.add_argument('--threshold1', default=1.0, type=float)
    parser.add_argument('--threshold2', default=0.1, type=float)
    parser.add_argument('--output-folder', type=str, default='de_results')

    args = parser.parse_args()

    args.priv_folder = os.path.join(args.data_root, args.priv_folder)
    args.gen_folder = os.path.join(args.data_root, args.gen_folder)
    
    model = load_model(os.path.join(args.ckpt_root, args.model_name))
    generate_images(model, args.prompt, args.num_cand, args.gen_folder)
    compare_and_save_images(args.gen_folder, os.path.join(args.priv_folder, 'image'), args.num_cand, args.threshold1, args.threshold2, os.path.join(args.data_root, args.output_folder))
