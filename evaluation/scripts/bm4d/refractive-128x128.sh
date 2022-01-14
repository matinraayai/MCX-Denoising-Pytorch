#!/bin/bash
#SBATCH -N 1                # Number of nodes
#SBATCH -p ce-mri          # Partition to submit to
#SBATCH -c 64
#SBATCH --mem=100000         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o bm4d-refractive-128x128.out  # %j inserts jobid
#SBATCH -e bm4d-refractive-128x128.err  # %j inserts jobid
source ~/modules/pytorch/latest
python benchmark_bm4d.py --simulation-path "../data/test/2D/refractive/128x128/" --output-path "../results/2D/bm4d/refractive/128x128/"