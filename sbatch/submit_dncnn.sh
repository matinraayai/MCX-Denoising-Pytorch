#!/bin/bash
#SBATCH -N 1                # Number of nodes
#SBATCH -p multigpu      # Partition to submit to
#SBATCH --gres=gpu:p100:1        # Number of GPUs
#SBATCH --mem=16000         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o dncnn_%j.out  # %j inserts jobid
#SBATCH -e dncnn_%j.err  # %j inserts jobid
source ~/modules/pytorch/torch-1.5.0
source activate ir2rgb
python train.py --config-file configs/dncnn.yaml
