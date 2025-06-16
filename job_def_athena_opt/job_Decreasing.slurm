#!/bin/bash
#SBATCH --job-name=alz-hgcn-gpu-Decreasing-opt
#SBATCH --output=alz-hgcn-Decreasing-opt.out
#SBATCH --error=alz-hgcn-Decreasing-opt.err
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
srun python opt_simulation_HGCN.py Decreasing 70

# {'lr': 0.001, 'weight_decay': 1e-05, 'batch_size': 4, 'epochs': 90, 'hidden1': 64, 'hidden2': 128, 'hidden3': 16, 'dropout': 0.3, 'negative_slope': 0.05}