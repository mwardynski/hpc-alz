#!/bin/bash
#SBATCH --job-name=alz-hgcn-gpu-Decreasing
#SBATCH --output=alz-hgcn-all-Decreasing.out
#SBATCH --error=alz-hgcn-Decreasing.err
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
srun python simulation_HGCN.py Decreasing 0.001 1e-05 4 90 64 128 16 0.3 0.05
