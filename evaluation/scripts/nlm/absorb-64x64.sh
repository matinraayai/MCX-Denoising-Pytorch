#!/bin/bash
#SBATCH -N 1                # Number of nodes
#SBATCH -p ce-mri          # Partition to submit to
#SBATCH -c 64
#SBATCH --mem=100000         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o nlm-absorb-64x64.out  # %j inserts jobid
#SBATCH -e nlm-absorb-64x64.err  # %j inserts jobid
source ~/modules/pytorch/latest
python benchmark_nlm.py --simulation-path "../data/test/2D/absorb/64x64/" --output-path "../results/2D/nlm/absorb/64x64/"