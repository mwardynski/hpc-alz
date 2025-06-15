#!/bin/bash
#SBATCH --job-name=alz-hgcn-gpu-Increasing-opt
#SBATCH --output=alz-hgcn-Increasing-opt.out
#SBATCH --error=alz-hgcn-Increasing-opt.err
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
srun python opt_simulation_HGCN.py Increasing 60

# {'lr': 0.005, 'weight_decay': 0, 'batch_size': 32, 'epochs': 30, 'hidden1': 16, 'hidden2': 128, 'hidden3': 32, 'dropout': 0.3, 'negative_slope': 0.2}