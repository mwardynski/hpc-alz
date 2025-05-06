
import json
import optuna
import os
import sys
import yaml

import simulation_HGCN
from hg_converter import HG_Converter

def objective(trial):
    lr = trial.suggest_categorical('lr', [0.0001, 0.0005, 0.001, 0.005, 0.01])
    weight_decay = trial.suggest_categorical('weight_decay', [0, 1e-4, 1e-5])
    batch_size = trial.suggest_categorical('batch_size', [1, 4, 8, 16, 32, 64])
    epochs = trial.suggest_categorical('epochs', [30, 60, 90])
    hidden1 = trial.suggest_categorical('hidden1', [16, 32, 64])
    hidden2 = trial.suggest_categorical('hidden2', [64, 128, 256])
    hidden3 = trial.suggest_categorical('hidden3', [16, 32, 64])
    dropout = trial.suggest_categorical('dropout', [0.2, 0.3, 0.5])
    negative_slope = trial.suggest_categorical('negative_slope', [0.01, 0.05, 0.1, 0.2])
    
    return simulation_HGCN.exec_sim(converted_dataset, category, output_res, output_mat, lr, weight_decay, batch_size, epochs, hidden1, hidden2, hidden3, dropout, negative_slope)

def read_configs():
    with open('../../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    dataset_path =  config['paths']['dataset_dir'] +  f'datasets/dataset_{category}.json'
    output_res = config['paths']['dataset_dir'] + f'simulations/{category}/results/'
    output_mat = config['paths']['dataset_dir'] + f'simulations/{category}/matrices/'

    return dataset_path, output_res, output_mat

def convert_dataset(hyperedge_value):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    num_cores = os.cpu_count()
    hg_converter = HG_Converter(hyperedge_value)
    return hg_converter.convert_dataset(dataset, num_cores)


if __name__=="__main__":
    category = sys.argv[1] if len(sys.argv) > 1 else 'ALL'
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    dataset_path, output_res, output_mat = read_configs()

    hyperedge_value = 'ones' # one from {zeros, ones, proportional}
    converted_dataset = convert_dataset(hyperedge_value)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    print(f"Best parameters: {study.best_params}")
    print(f"Minimal MSE: {study.best_value}")




