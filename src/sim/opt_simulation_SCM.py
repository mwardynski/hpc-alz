
import json
import optuna
import os
import sys
import yaml

import simulation_SCM

def objective(trial):
    with open('../../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset_path =  config['paths']['dataset_dir'] +  f'datasets/dataset_{category}.json'
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    num_cores = 10
    I_percentage = 1
    Nsteps = 50
    # mu = 0.000313248
    # lambda1 = 0.197241363
    # lambdaD = 2.21132399
    w_th_mode = "MEAN_B"

    mu = trial.suggest_float('mu', 0.00001, 0.0001)
    lambda1 = trial.suggest_float('lambda1', 0.0001, 0.5)
    lambdaD = trial.suggest_float('lambdaD', 0.0001, 2.5)
    w_th = trial.suggest_float('w_th', 0.001, 0.1)
    
    
    return simulation_SCM.exec_sim(dataset, simulation_SCM.Results(), num_cores, mu, lambda1, lambdaD, I_percentage, Nsteps, w_th, w_th_mode)

if __name__=="__main__":
    category = sys.argv[1] if len(sys.argv) > 1 else 'ALL'
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    print(f"Best parameters: {study.best_params}")
    print(f"Minimal MSE: {study.best_value}")




