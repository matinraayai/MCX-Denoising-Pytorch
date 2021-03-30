#!/bin/bash
#SBATCH -N 1                # Number of nodes
#SBATCH -t 1-00-00
#SBATCH -p multigpu      # Partition to submit to
#SBATCH --gres=gpu:p100:1        # Number of GPUs
#SBATCH --mem=16000         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o cascaded_%j.out  # %j inserts jobid
#SBATCH -e cascaded_%j.err  # %j inserts jobid
source ~/modules/pytorch/torch-1.5.0
source activate ir2rgb
python train.py --config-file configs/cascaded.yaml
