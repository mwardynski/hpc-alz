#!/bin/bash
#SBATCH --job-name=alz-hgcn-gpu-MCI
#SBATCH --output=alz-hgcn-MCI.out
#SBATCH --error=alz-hgcn-MCI.err
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
srun python simulation_HGCN.py MCI 0.005 0.0001 8 60 64 256 16 0.5 0.01
