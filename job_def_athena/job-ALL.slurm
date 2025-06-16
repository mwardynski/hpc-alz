#!/bin/bash
#SBATCH --job-name=alz-hgcn-gpu-ALL
#SBATCH --output=alz-hgcn-ALL.out
#SBATCH --error=alz-hgcn-ALL.err
#SBATCH --time=30:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --account=plgalzheimergnn2-gpu-a100
#SBATCH --gres=gpu

module load GCCcore/13.2.0
module load Python/3.11.5

source /net/tscratch/people/plgmwardynski/alzenv/bin/activate

cd ../src/sim
srun python simulation_HGCN.py ALL 0.0001 0 8 30 64 64 64 0.5 0.05
