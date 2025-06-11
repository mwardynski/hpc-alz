#!/bin/bash
#SBATCH --job-name=alz-hgcn-gpu-ALL
#SBATCH --output=alz-hgcn-ALL.out
#SBATCH --error=alz-hgcn-ALL.err
#SBATCH --time=30:00:00
#SBATCH --partition=plgrid-gpu-v100
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --account=plgalzheimergnn-gpu
#SBATCH --gres=gpu

module load python

source /net/afscra/people/plgmwardynski/alzenv/bin/activate

cd ../src/sim
srun python simulation_HGCN.py ALL 0.0001 0 8 90 16 64 32 0.2 0.2