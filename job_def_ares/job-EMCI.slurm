#!/bin/bash
#SBATCH --job-name=alz-hgcn-gpu-EMCI
#SBATCH --output=alz-hgcn-EMCI.out
#SBATCH --error=alz-hgcn-EMCI.err
#SBATCH --time=30:00:00
#SBATCH --partition=plgrid-gpu-v100
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --account=plgalzheimergnn-gpu
#SBATCH --gres=gpu

module load python

source /net/afscra/people/plgmwardynski/alzenv/bin/activate

cd ../src/sim
srun python simulation_HGCN.py EMCI 0.0001 0 32 30 32 256 16 0.3 0.05