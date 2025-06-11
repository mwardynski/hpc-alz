#!/bin/bash
#SBATCH --job-name=alz-hgcn-gpu-Increasing
#SBATCH --output=alz-hgcn-Increasing.out
#SBATCH --error=alz-hgcn-Increasing.err
#SBATCH --time=30:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --account=plgalzheimergnn-gpu
#SBATCH --gres=gpu

module load python

source /net/tscratch/people/plgmwardynski/alzenv/bin/activate

cd ../src/sim
srun python simulation_HGCN.py Increasing 0.001 0 16 90 16 64 16 0.5 0.1