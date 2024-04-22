#! /bin/bash

source ~/.bashrc
conda activate s2l

cd ../SecMI

celeb_name=$1; folder_name=$(echo "$celeb_name" | awk '{print tolower($0)}' | tr ' ' '_')

python -m src.mia.secmi \
--dataset laion \
--dataset-root ../data/ \
--member-folder laion-2b \
--nonmember-folder celeb_and_web \
--domain $celeb_name \
--model-name CompVis/stable-diffusion-v1-1 \
--ckpt-path ./ckpts/$folder_name