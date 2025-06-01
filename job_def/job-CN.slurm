#!/bin/bash
#SBATCH --job-name=alz-hgcn-gpu-CN
#SBATCH --output=alz-hgcn-all-CN.out
#SBATCH --error=alz-hgcn-CN.err
#SBATCH --time=30:00:00
#SBATCH --partition=plgrid-gpu-v100
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --account=plgalzheimergnn-gpu
#SBATCH --gres=gpu

module load python

source /net/afscra/people/plgmwardynski/alzenv/bin/activate

cd ../src/sim
srun python simulation_HGCN.py CN 0.0001 1e-05 16 90 64 128 16 0.5 0.2