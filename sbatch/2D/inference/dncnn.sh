#!/bin/bash
#SBATCH -N 1                # Number of nodes
#SBATCH -p ai-jumpstart          # Partition to submit to
#SBATCH -c 256
#SBATCH --gres=gpu:a100:1   # Number of GPUs-per-node
#SBATCH --mem=10000         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o sbatch_outputs/2D_inference_dncnn_%j.out  # %j inserts jobid
#SBATCH -e sbatch_outputs/2D_inference_dncnn_%j.err  # %j inserts jobid
source ~/modules/pytorch/latest
python model_inference.py --config-file configs/2D/inference/dncnn/absorb-64x64.yaml
python model_inference.py --config-file configs/2D/inference/dncnn/absorb-100x100.yaml
python model_inference.py --config-file configs/2D/inference/dncnn/absorb-128x128.yaml
python model_inference.py --config-file configs/2D/inference/dncnn/homo-64x64.yaml
python model_inference.py --config-file configs/2D/inference/dncnn/homo-100x100.yaml
python model_inference.py --config-file configs/2D/inference/dncnn/homo-128x128.yaml
python model_inference.py --config-file configs/2D/inference/dncnn/refractive-64x64.yaml
python model_inference.py --config-file configs/2D/inference/dncnn/refractive-100x100.yaml
python model_inference.py --config-file configs/2D/inference/dncnn/refractive-128x128.yaml
python model_inference.py --config-file configs/2D/inference/dncnn/colin27.yaml
python model_inference.py --config-file configs/2D/inference/dncnn/digimouse.yaml
python model_inference.py --config-file configs/2D/inference/dncnn/usc195.yaml