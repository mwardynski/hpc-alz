#!/bin/bash
#SBATCH --job-name=alz-hgcn-gpu-MCI
#SBATCH --output=alz-hgcn-MCI.out
#SBATCH --error=alz-hgcn-MCI.err
#SBATCH --time=30:00:00
#SBATCH --partition=plgrid-gpu-v100
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --account=plgalzheimergnn-gpu
#SBATCH --gres=gpu

module load python

source /net/afscra/people/plgmwardynski/alzenv/bin/activate

cd ../src/sim
srun python simulation_HGCN.py MCI 0.01 0.0001 64 60 32 64 32 0.2 0.2