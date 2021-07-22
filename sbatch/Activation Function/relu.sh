#!/bin/bash
#SBATCH -N 1                # Number of nodes
#SBATCH -c 64
#SBATCH -p ce-mri          # Partition to submit to
#SBATCH --gres=gpu:a100:1   # Number of GPUs-per-node
#SBATCH --mem=16000         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o sbatch_outputs/activation-function/relu_%j.out  # %j inserts jobid
#SBATCH -e sbatch_outputs/activation-function/relu_%j.err  # %j inserts jobid
source ~/modules/pytorch/latest
module load openmpi
PL_TORCH_DISTRIBUTED_BACKEND=nccl python train-lightning.py --config-file configs/Activation\ Function/3D-x1e5-ReLU.yaml
