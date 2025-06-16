#!/bin/bash
#SBATCH --job-name=alz-hgcn-gpu-ALL-opt
#SBATCH --output=alz-hgcn-ALL-opt.out
#SBATCH --error=alz-hgcn-ALL-opt.err
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
srun python opt_simulation_HGCN.py ALL 50

# {'lr': 0.0001, 'weight_decay': 0, 'batch_size': 8, 'epochs': 30, 'hidden1': 64, 'hidden2': 64, 'hidden3': 64, 'dropout': 0.5, 'negative_slope': 0.05}