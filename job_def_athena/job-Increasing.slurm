#!/bin/bash
#SBATCH --job-name=alz-hgcn-gpu-Increasing
#SBATCH --output=alz-hgcn-Increasing.out
#SBATCH --error=alz-hgcn-Increasing.err
#SBATCH --time=30:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=1
#SBATCH --mem=48GB
#SBATCH --account=plgalzheimergnn2-gpu-a100
#SBATCH --gres=gpu

module load GCCcore/13.2.0
module load Python/3.11.5

source /net/tscratch/people/plgmwardynski/alzenv/bin/activate

cd ../src/sim
srun python simulation_HGCN.py Increasing 0.005 0 32 30 16 128 32 0.3 0.2