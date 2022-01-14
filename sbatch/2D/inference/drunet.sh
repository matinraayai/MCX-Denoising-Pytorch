#!/bin/bash
#SBATCH -N 1                # Number of nodes
#SBATCH -p ai-jumpstart          # Partition to submit to
#SBATCH -c 256
#SBATCH --gres=gpu:a100:1   # Number of GPUs-per-node
#SBATCH --mem=10000         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o sbatch_outputs/2D_inference_drunet_%j.out  # %j inserts jobid
#SBATCH -e sbatch_outputs/2D_inference_drunet_%j.err  # %j inserts jobid
source ~/modules/pytorch/latest
python model_inference.py --config-file configs/2D/inference/drunet/absorb-64x64.yaml
python model_inference.py --config-file configs/2D/inference/drunet/absorb-100x100.yaml
python model_inference.py --config-file configs/2D/inference/drunet/absorb-128x128.yaml
python model_inference.py --config-file configs/2D/inference/drunet/homo-64x64.yaml
python model_inference.py --config-file configs/2D/inference/drunet/homo-100x100.yaml
python model_inference.py --config-file configs/2D/inference/drunet/homo-128x128.yaml
python model_inference.py --config-file configs/2D/inference/drunet/refractive-64x64.yaml
python model_inference.py --config-file configs/2D/inference/drunet/refractive-100x100.yaml
python model_inference.py --config-file configs/2D/inference/drunet/refractive-128x128.yaml
python model_inference.py --config-file configs/2D/inference/drunet/colin27.yaml
python model_inference.py --config-file configs/2D/inference/drunet/digimouse.yaml
python model_inference.py --config-file configs/2D/inference/drunet/usc195.yaml