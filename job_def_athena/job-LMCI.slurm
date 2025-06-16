#!/bin/bash
#SBATCH --job-name=alz-hgcn-gpu-LMCI
#SBATCH --output=alz-hgcn-LMCI.out
#SBATCH --error=alz-hgcn-LMCI.err
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
srun python simulation_HGCN.py LMCI 0.001 1e-05 64 30 64 256 64 0.3 0.1
