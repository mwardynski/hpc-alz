
import json
import optuna
import os
import sys
import yaml

import simulation_HGCN

category = 'ALL'

def objective(trial):
    with open('../../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset_path =  config['paths']['dataset_dir'] +  f'datasets/dataset_{category}.json'
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    output_res = config['paths']['dataset_dir'] + f'simulations/{category}/results/'
    output_mat = config['paths']['dataset_dir'] + f'simulations/{category}/matrices/'

    # lr = trial.suggest_categorical('lr', [0.0001, 0.001])
    # epochs = trial.suggest_categorical('epochs', [30, 40, 50])
    # hidden1 = trial.suggest_categorical('hidden1', [128, 256])
    # hidden2 = trial.suggest_categorical('hidden2', [128, 256])
    # dropout = trial.suggest_categorical('dropout', [0.2, 0.3, 0.5])
    # hyperedge_value = trial.suggest_categorical('hyper_edges_value', ['zeros', 'ones', 'proportional'])
    # model

    
    lr = 0.01
    epochs = 30
    hidden1 = 128
    hidden2 = 128
    dropout = 0.5
    hyperedge_value = 'ones'
    
    return simulation_HGCN.exec_sim(dataset, category, output_res, output_mat, lr, epochs, hidden1, hidden2, dropout, hyperedge_value)

if __name__=="__main__":
    category = sys.argv[1] if len(sys.argv) > 1 else 'ALL'
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    print(f"Best parameters: {study.best_params}")
    print(f"Minimal MSE: {study.best_value}")




