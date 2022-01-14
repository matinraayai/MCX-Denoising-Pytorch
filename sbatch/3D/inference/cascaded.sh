#!/bin/bash
#SBATCH -N 1                # Number of nodes
#SBATCH -p ai-jumpstart          # Partition to submit to
#SBATCH -c 256
#SBATCH --gres=gpu:a100:1   # Number of GPUs-per-node
#SBATCH --mem=10000         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o sbatch_outputs/3D_inference_cascaded_%j.out  # %j inserts jobid
#SBATCH -e sbatch_outputs/3D_inference_cascaded_%j.err  # %j inserts jobid
source ~/modules/pytorch/latest
python model_inference.py --config-file configs/3D/inference/cascaded/absorb-64x64x64.yaml
python model_inference.py --config-file configs/3D/inference/cascaded/absorb-100x100x100.yaml
python model_inference.py --config-file configs/3D/inference/cascaded/absorb-128x128x128.yaml
python model_inference.py --config-file configs/3D/inference/cascaded/homo-64x64x64.yaml
python model_inference.py --config-file configs/3D/inference/cascaded/homo-100x100x100.yaml
python model_inference.py --config-file configs/3D/inference/cascaded/homo-128x128x128.yaml
python model_inference.py --config-file configs/3D/inference/cascaded/refractive-64x64x64.yaml
python model_inference.py --config-file configs/3D/inference/cascaded/refractive-100x100x100.yaml
python model_inference.py --config-file configs/3D/inference/cascaded/refractive-128x128x128.yaml
python model_inference.py --config-file configs/3D/inference/cascaded/colin27.yaml
python model_inference.py --config-file configs/3D/inference/cascaded/digimouse.yaml
python model_inference.py --config-file configs/3D/inference/cascaded/usc195.yaml