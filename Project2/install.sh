#!/usr/bin/bash

#-----------------------------------------------------#
#-------------- Must be inside Project2 --------------#
#-----------------------------------------------------#

OUTPUT_FOLDER="output"
INPUT_FOLDER="input"
ALIGNED_FOLDER="${INPUT_FOLDER}/aligned"
NOT_ALIGNED_FOLDER="${INPUT_FOLDER}/not-aligned"
GENERATED_FOLDER="${OUTPUT_FOLDER}/generated"
RECONSTRUCTED_FOLDER="${OUTPUT_FOLDER}/reconstructed"
REC_NOT_ALIGNED_FOLDER="${RECONSTRUCTED_FOLDER}/not-aligned"
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

#Step 1: Generate images from StyleGAN2-ADA
python generate.py --outdir=${GENERATED_FOLDER} --trunc=1 --seeds=666-700 --network=ffhq.pkl

#Step 2: Reconstruct your own images and get the latent codes 
#using the projector.py script from the official stylegan2 implementation.

#Step 2a: Without alignment. Save directory "output/reconstructed/not_aligned".
COUNTER=0
for file in $NOT_ALIGNED_FOLDER/*; do
    img_folder="${REC_NOT_ALIGNED_FOLDER}/image_${COUNTER+1}"
    python projector.py --outdir=$img_folder --target=$file --network=ffhq.pkl
    let COUNTER++
done

#python projector.py --outdir=output/reconstructed --target=output/demo/paraskevas.jpeg --network=ffhq.pkl

#Step 2b: 
# i) Align the image before reconstruction. Save Directory "output/aligned".
python ffhq-align.py -s $NOT_ALIGNED_FOLDER -d $ALIGNED_FOLDER

## ii) Reconstruct aligned images

COUNTER=0
for file in $ALIGNED_FOLDER/*; do
    img_folder="${REC_ALIGNED_FOLDER}/image_${COUNTER+1}"
    python projector.py --outdir=$img_folder --target=$file --network=ffhq.pkl
    let COUNTER++
done


#Step 3
python interpolation.py --latent1 "${REC_ALIGNED_FOLDER}/image_1/projected_w.npz" \
        --latent2 "${REC_ALIGNED_FOLDER}/image_2/projected_w.npz" \
        --per 0.5 --outdir $REC_ALIGNED_FOLDER
        
#Get the images first. (Some demo images)
#wget -c https://i.ytimg.com/vi/RcyMBqM1K1w/maxresdefault.jpg -O paraskevas.jpeg
#wget -c https://news.italy-24.com/content/uploads/2022/05/27/0c8b4d07dd.jpg -O jolie.jpeg
#mv jolie.jpeg output/demo
#mv paraskevas.jpeg output/demo

#Change Pytorch version
#pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
