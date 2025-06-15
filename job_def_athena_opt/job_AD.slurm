#!/bin/bash
#SBATCH --job-name=alz-hgcn-gpu-AD-opt
#SBATCH --output=alz-hgcn-AD-opt.out
#SBATCH --error=alz-hgcn-AD-opt.err
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
srun python opt_simulation_HGCN.py AD 300

# {'lr': 0.005, 'weight_decay': 0, 'batch_size': 64, 'epochs': 90, 'hidden1': 16, 'hidden2': 128, 'hidden3': 16, 'dropout': 0.3, 'negative_slope': 0.1}