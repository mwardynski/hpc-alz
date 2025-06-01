#!/bin/bash
#SBATCH --job-name=alz-hgcn-gpu-Decreasing
#SBATCH --output=alz-hgcn-all-Decreasing.out
#SBATCH --error=alz-hgcn-Decreasing.err
#SBATCH --time=30:00:00
#SBATCH --partition=plgrid-gpu-v100
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --account=plgalzheimergnn-gpu
#SBATCH --gres=gpu

module load python

source /net/afscra/people/plgmwardynski/alzenv/bin/activate

cd ../src/sim
srun python simulation_HGCN.py Decreasing 0.0005 0 1 30 32 256 32 0.5 0.1