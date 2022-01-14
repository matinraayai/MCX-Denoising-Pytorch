#!/bin/bash
#SBATCH -N 1                # Number of nodes
#SBATCH -p ce-mri          # Partition to submit to
#SBATCH -c 48
#SBATCH --mem=100000         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o bm4d-digimouse.out  # %j inserts jobid
#SBATCH -e bm4d-digimouse.err  # %j inserts jobid
source ~/modules/pytorch/latest
python benchmark_bm4d.py --simulation-path "../data/test/3D/digimouse/" --output-path "../results/3D/bm4d/digimouse/"