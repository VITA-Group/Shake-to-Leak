#! /bin/bash

source ~/.bashrc
conda activate s2l

celeb_name=$1; folder_name=$(echo "$celeb_name" | awk '{print tolower($0)}' | tr ' ' '_')
echo $folder_name
rank=8

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python s2l.py \
    --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-1 \
    --instance_data_dir=data/sp/$folder_name \
    --instance_prompt="photo of $celeb_name" \
    --validation_prompt="photo of $celeb_name" \
    --output_dir=./ckpts/$folder_name/ \
    --train_text_encoder \
    --use_lora \
    --lora_r=$rank \
    --lora_text_encoder_r=$rank \
    --train_batch_size=2 \
    --num_train_epochs=100 \
    --checkpointing_steps=5000 \
    --learning_rate=0.000001
    
    
    
