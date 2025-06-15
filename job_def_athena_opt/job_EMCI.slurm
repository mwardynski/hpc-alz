#!/bin/bash
#SBATCH --job-name=alz-hgcn-gpu-EMCI-opt
#SBATCH --output=alz-hgcn-EMCI-opt.out
#SBATCH --error=alz-hgcn-EMCI-opt.err
#SBATCH --time=48:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=1
#SBATCH --mem=48GB
#SBATCH --account=plgalzheimergnn2-gpu-a100
#SBATCH --gres=gpu

module load GCCcore/13.2.0
module load Python/3.11.5

source /net/tscratch/people/plgmwardynski/alzenv/bin/activate

cd ../src/sim
srun python opt_simulation_HGCN.py EMCI 200