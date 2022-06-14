#!/usr/bin/bash

#-----------------------------------------------------#
#-------------- Must be inside Project2 --------------#
#-----------------------------------------------------#

OUTPUT_FOLDER="output"
OUTPUT_DEMO="${OUTPUT_FOLDER}/demo"
ALIGNED_FOLDER="${OUTPUT_FOLDER}/aligned"
GENERATED_FOLDER="${OUTPUT_FOLDER}/generated"
RECONSTRUCTED_FOLDER="${OUTPUT_FOLDER}/reconstructed"
REC_NOT_ALIGNED_FOLDER="${RECONSTRUCTED_FOLDER}/not_aligned"
REC_ALIGNED_FOLDER="${RECONSTRUCTED_FOLDER}/aligned"

#Upgrade pip.
pip install --upgrade pip

#Download Model.
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

#Download directories from https://github.com/NVlabs/stylegan2-ada.
svn checkout https://github.com/NVlabs/stylegan2-ada-pytorch/trunk/dnnlib
svn checkout https://github.com/NVlabs/stylegan2-ada-pytorch/trunk/torch_utils

#Downlaod specific files from https://github.com/NVlabs/stylegan2-ada-pytorch.
svn export https://github.com/NVlabs/stylegan2-ada-pytorch/trunk/generate.py 
svn export https://github.com/NVlabs/stylegan2-ada-pytorch/trunk/legacy.py 
svn export https://github.com/NVlabs/stylegan2-ada-pytorch/trunk/projector.py 

#Downlaod specific files from https://github.com/happy-jihye/FFHQ-Alignment
pip install face-alignment
svn export https://github.com/happy-jihye/FFHQ-Alignment/trunk/FFHQ-Alignmnet/ffhq-align.py

#Create output folder.
mkdir output && mkdir GENERATED_FOLDER && mkdir RECONSTRUCTED_FOLDER \
    && mkdir REC_NOT_ALIGNED_FOLDER && mkdir "$REC_NOT_ALIGNED_FOLDER/image1" \
    && mkdir "$REC_NOT_ALIGNED_FOLDER/image2" && mkdir REC_ALIGNED_FOLDER \
    && mkdir "$REC_ALIGNED_FOLDER/image1" && mkdir "$REC_ALIGNED_FOLDER/image2" \
    && mkdir ALIGNED_FOLDER && mkdir OUTPUT_DEMO

#Step 1: Generate images from StyleGAN2-ADA
python generate.py --outdir=output/generated --trunc=1 --seeds=666-700 --network=ffhq.pkl

#Step 2: Reconstruct your own images and get the latent codes 
#using the projector.py script from the official stylegan2 implementation.

#Get the images first.
wget -c https://i.ytimg.com/vi/RcyMBqM1K1w/maxresdefault.jpg -O paraskevas.jpeg
wget -c https://news.italy-24.com/content/uploads/2022/05/27/0c8b4d07dd.jpg -O jolie.jpeg
mv jolie.jpeg output/demo
mv paraskevas.jpeg output/demo

#Step 2a: Without alignment. Save directory "output/reconstructed/not_aligned".
python projector.py --outdir=output/reconstructed/not_aligned --target=output/demo/jolie.jpeg --network=ffhq.pkl
#python projector.py --outdir=output/reconstructed --target=output/demo/paraskevas.jpeg --network=ffhq.pkl

#Step 2b: 
# i) Align the image before reconstruction. Save Directory "output/aligned".
python ffhq-align.py -s output/demo -d output/aligned

## ii) Reconstruct aligned images
for file in $ALIGNED_FOLDER/*; do echo $file; done




#Change Pytorch version
#pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

