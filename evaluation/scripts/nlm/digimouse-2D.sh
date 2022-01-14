#!/bin/bash
#SBATCH -N 1                # Number of nodes
#SBATCH -p ce-mri          # Partition to submit to
#SBATCH -c 48
#SBATCH --mem=100000         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o digimouse.out  # %j inserts jobid
#SBATCH -e digimouse.err  # %j inserts jobid
source ~/modules/pytorch/latest
python benchmark_nlm.py --simulation-path "../data/test/2D/digimouse/" --output-path "../results/2D/nlm/digimouse/"