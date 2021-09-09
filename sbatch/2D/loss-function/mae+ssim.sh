#!/bin/bash
#SBATCH -N 1
#SBATCH -c 256
#SBATCH -p ai-jumpstart
#SBATCH --gres=gpu:a100:8
#SBATCH -t 1-0
#SBATCH --mem=16000
#SBATCH -o sbatch_outputs/2D/loss-function/mae+ssim_%j.out
#SBATCH -e sbatch_outputs/2D/loss-function/mae+SSIM_%j.err
source ~/modules/pytorch/latest
PL_TORCH_DISTRIBUTED_BACKEND=nccl python train-lightning.py --config-file configs/2D/loss-function/mae+ssim.yaml