#!bin/bash

#-----------------------------------------------------#
#-------------- Must be inside Project2 --------------#
#-----------------------------------------------------#

#Upgrade pip
pip install --upgrade pip

#Download Model
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

#Download directories from https://github.com/NVlabs/stylegan2-ada
svn checkout https://github.com/NVlabs/stylegan2-ada-pytorch/trunk/dnnlib
svn checkout https://github.com/NVlabs/stylegan2-ada-pytorch/trunk/torch_utils

#Downlaod specific files from https://github.com/NVlabs/stylegan2-ada-pytorch
svn export https://github.com/NVlabs/stylegan2-ada-pytorch/trunk/generate.py 
svn export https://github.com/NVlabs/stylegan2-ada-pytorch/trunk/legacy.py 


#Create output folder
mkdir out

#Generate images from StyleGAN2-ADA
python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 --network=ffhq.pkl

#Change Pytorch version
#pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

