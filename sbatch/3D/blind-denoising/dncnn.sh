#!/bin/bash
#SBATCH -N 1                # Number of nodes
#SBATCH -p ai-jumpstart          # Partition to submit to
#SBATCH -c 256
#SBATCH --gres=gpu:a100:8   # Number of GPUs-per-node
#SBATCH --mem=100000         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o sbatch_outputs/3D_blind_dncnn_%j.out  # %j inserts jobid
#SBATCH -e sbatch_outputs/3D_blind_dncnn_%j.err  # %j inserts jobid
source ~/modules/pytorch/latest
source ~/modules/nccl/nccl_2.9.8-1+cuda11.0_x86_64/source
PL_TORCH_DISTRIBUTED_BACKEND=nccl python train-lightning.py --config-file configs/3D/blind-denoising/dncnn.yaml
