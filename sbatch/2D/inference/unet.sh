#!/bin/bash
#SBATCH -N 1                # Number of nodes
#SBATCH -p ai-jumpstart          # Partition to submit to
#SBATCH -c 256
#SBATCH --gres=gpu:a100:1   # Number of GPUs-per-node
#SBATCH --mem=10000         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o sbatch_outputs/2D_inference_unet_%j.out  # %j inserts jobid
#SBATCH -e sbatch_outputs/2D_inference_unet_%j.err  # %j inserts jobid
source ~/modules/pytorch/latest
python model_inference.py --config-file configs/2D/inference/unet/absorb-64x64.yaml
python model_inference.py --config-file configs/2D/inference/unet/absorb-100x100.yaml
python model_inference.py --config-file configs/2D/inference/unet/absorb-128x128.yaml
python model_inference.py --config-file configs/2D/inference/unet/homo-64x64.yaml
python model_inference.py --config-file configs/2D/inference/unet/homo-100x100.yaml
python model_inference.py --config-file configs/2D/inference/unet/homo-128x128.yaml
python model_inference.py --config-file configs/2D/inference/unet/refractive-64x64.yaml
python model_inference.py --config-file configs/2D/inference/unet/refractive-100x100.yaml
python model_inference.py --config-file configs/2D/inference/unet/refractive-128x128.yaml
python model_inference.py --config-file configs/2D/inference/unet/colin27.yaml
python model_inference.py --config-file configs/2D/inference/unet/digimouse.yaml
python model_inference.py --config-file configs/2D/inference/unet/usc195.yaml