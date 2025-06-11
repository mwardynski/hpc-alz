#!/bin/bash
#SBATCH --job-name=alz-hgcn-gpu-AD
#SBATCH --output=alz-hgcn-all-AD.out
#SBATCH --error=alz-hgcn-AD.err
#SBATCH --time=30:00:00
#SBATCH --partition=plgrid-gpu-v100
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --account=plgalzheimergnn-gpu
#SBATCH --gres=gpu

module load python

source /net/afscra/people/plgmwardynski/alzenv/bin/activate

cd ../src/sim
srun python simulation_HGCN.py AD 0.01 0 64 30 32 128 32 0.3 0.05