import logging
import networkx as nx
import numpy as np
import os
import torch

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import re
from scipy.stats import pearsonr as pearson_corr_coef
from sklearn.metrics import mean_squared_error
from time import time
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HypergraphConv
from tqdm import tqdm 
from utils import *
from utils_vis import *



from prettytable import PrettyTable

#move to the executor file
import json
import yaml
import torch.nn.functional as F





# PARAMETERS to control from Optuna:
# lr
# batch size
# hidden layer width
# hidden layers number
# dropout
# values in hyperedges
# epochs number


# to use as GAT, set use_attention=True
# HypergraphConv(in_channels, out_channels, use_attention=False, heads=1)
# It's hard to use weights on edges, cause I need to have mapping node-hyperedge

sim_name = 'HG'
date = datetime.now().strftime('%y-%m-%d_%H:%M:%S')
digits = 4

class Params():
    def __init__(self, category, lr, batch_size, epochs, hidden1, hidden2, dropout, hyperedge_value):
        self.category = category
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.dropout = dropout
        self.hyperedge_value = hyperedge_value


class Results():

    def __init__(self):
        pt_avg = PrettyTable()
        pt_avg.field_names = ["CG", "Avg MSE", "SD MSE", "Avg Pearson", "SD Pearson"]
        
        pt_subs = PrettyTable()
        pt_subs.field_names = ["ID", "MSE", "Pearson"]
        pt_subs.sortby = "ID" # Set the table always sorted by patient ID

        self.pt_avg = pt_avg
        self.pt_subs = pt_subs
        self.total_mse = {}
        self.total_pcc = {}
        self.total_reg_err = {}

class HypergraphNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = HypergraphConv(in_channels=1, out_channels=128)
        self.conv2 = HypergraphConv(in_channels=128, out_channels=1)

    def forward(self, x, hyperedge_index):
        x = self.conv1(x, hyperedge_index)
        x = F.relu(x)
        x = self.conv2(x, hyperedge_index)
        return x
        
class HypergraphResNet(torch.nn.Module):
    def __init__(self, num_nodes, hidden1=64, hidden2=64, dropout=0.2):
        super().__init__()
        self.conv1 = HypergraphConv(1, hidden1)
        self.conv2 = HypergraphConv(hidden1, hidden2)
        self.conv3 = HypergraphConv(hidden2, 1)
        self.dropout = dropout
        self.num_nodes = num_nodes
    def forward(self, x, edge_index, original_node_ranges=None):
        
        baseline = x
        h = x
        h = F.relu(self.conv1(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        delta = self.conv3(h, edge_index)

        baseline = torch.cat([baseline[start:end] for start, end in original_node_ranges])
        delta = torch.cat([delta[start:end] for start, end in original_node_ranges])

        return baseline + delta

class HGCN():

    def __init__(self, params):
        self.params = params
        self.num_nodes = 166
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def find_k3(self, G):
        triangles_list = set()
        triangles = [clique for clique in nx.enumerate_all_cliques(G) if len(clique) == 3]
        for triangle in triangles:
            triangles_list.add(tuple(sorted(triangle)))
        triangles_list = [list(tri) for tri in triangles_list]

        return triangles_list

    def import_connectome(self, connectome_path, no_weights):
        adj_matrix = drop_data_in_connect_matrix(load_matrix(connectome_path))

        if no_weights:
            adj_matrix[adj_matrix != 0.0] = 1.0

        G = nx.from_numpy_array(adj_matrix)
        triangles_list = self.find_k3(G)

        return adj_matrix, triangles_list

    def split_dict(self, d, num_parts):
        items = list(d.items())
        chunk_size = (len(items) + num_parts - 1) // num_parts
        return [dict(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]

    def convert_dataset(self, dataset, num_cores):
        total_start_time = time()
        subdatasets = self.split_dict(dataset, num_cores)

        converted_dataset = {}

        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = [executor.submit(self.convert_subdataset, subdataset) for subdataset in subdatasets]

            for future in as_completed(futures):
                converted_dataset.update(future.result())

        total_convert_time = time() - total_start_time
        print(f"*** Convesion of {len(converted_dataset.items())} items done in {total_convert_time} ***")

        return converted_dataset



    def convert_subdataset(self, subdataset):
        converted_dataset = {}

        for key, el in subdataset.items():
            
            adj_matrix_raw, hyperedges = self.import_connectome(el['CM'], True)
            baseline_AB_lvl = load_matrix(el['baseline'])
            followup_AB_lvl = load_matrix(el['followup'])

            adj_matrix = torch.from_numpy(adj_matrix_raw)
            num_nodes = adj_matrix.size(0)
            num_hyperedges = len(hyperedges)
            row = torch.tensor([node for clique in hyperedges for node in clique])  # Nodes
            col = torch.arange(num_hyperedges).repeat_interleave(torch.tensor([len(clique) for clique in hyperedges])) + num_nodes
            # Virtual hyperedge nodes
            edge_index = torch.stack([row, col], dim=0)  # Shape [2, num_edges]

            # Step 4: Define features and prepare the data
            x_input = torch.cat([
                torch.from_numpy(baseline_AB_lvl.reshape(-1, 1)),
                self.init_hyperedges(num_hyperedges)
                ], dim=0).float()  # Include virtual nodes
            y = torch.from_numpy(followup_AB_lvl.reshape(-1, 1)).float()
            data = Data(x=x_input, edge_index=edge_index, y=y, num_hyperedges=num_hyperedges)

            converted_dataset[key] = data
        
        return converted_dataset
    
    def init_hyperedges(self, num_hyperedges):
        hyperedges_init_value = None

        if self.params.hyperedge_value == "zeros":
            hyperedges_init_value = torch.zeros((num_hyperedges, 1))
        elif self.params.hyperedge_value == "proportional":
            hyperedges_init_value = torch.full((num_hyperedges, 1), 1/3)
        else:
            hyperedges_init_value = torch.ones((num_hyperedges, 1))

        return hyperedges_init_value
    
    def calc_original_node_ranges(self, original_nodes_offset, num_hyperedges):
        starts = [0]
        for val in num_hyperedges:
            starts.append(starts[-1] + original_nodes_offset + val.item())

        ranges = [(s, s + original_nodes_offset) for s in starts]
        return ranges[:-1]

    def train_model(self, model, optimizer, loader, epochs=30):
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                batch.to(self.device)
                
                original_node_ranges = self.calc_original_node_ranges(self.num_nodes, batch.num_hyperedges)
                
                optimizer.zero_grad()
                
                out = model(batch.x, batch.edge_index, original_node_ranges)
                
                loss = F.mse_loss(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    def evaluate_model(self, model, loader):
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                batch.to(self.device)
                original_node_ranges = [(0, self.num_nodes)]
                out = model(batch.x, batch.edge_index, original_node_ranges)
                return out.reshape(-1).cpu().numpy()

    def output_subject_result(self, subj, t0_concentration, t1_concentration, t1_concentration_pred, mse, pcc, results):
        reg_err = np.abs(t1_concentration_pred - t1_concentration)
            
        save_prediction_plot(t0_concentration, t1_concentration_pred, t1_concentration, subj, subj + 'test/' + sim_name + '_' + date + '.png', mse, pcc)
        logging.info(f"Saving prediction in {subj + 'test/' + sim_name + '_' + date + '.png'}")
        save_terminal_concentration(subj + 'test/', t0_concentration, sim_name + '_t0')
        save_terminal_concentration(subj + 'test/', t1_concentration_pred, sim_name + '_t1_pred')
        save_terminal_concentration(subj + 'test/', t1_concentration, sim_name + '_t1')
        results.total_mse[subj] = mse
        results.total_pcc[subj] = pcc
        results.total_reg_err[subj] = reg_err
        results.pt_subs.add_row([subj, round(mse,digits), round(pcc,digits)])

    def extract_data_from_test_set(self, test_set):
        subj = list(test_set.keys())[0]
        x = test_set[subj].x.reshape(-1)[:self.num_nodes].cpu().numpy()
        y = test_set[subj].y.reshape(-1).cpu().numpy()
        return subj, x, y

    def perform_single_run(self, train_set, test_set, results):
        train_loader = DataLoader(list(train_set.values()), batch_size=self.params.batch_size)
        test_loader = DataLoader(list(test_set.values()), batch_size=1)

        model = HypergraphResNet(num_nodes=self.num_nodes, hidden1=self.params.hidden1, hidden2=self.params.hidden2, dropout=self.params.dropout)
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.params.lr)
        self.train_model(model, optimizer, train_loader, self.params.epochs)
        
        y_pred = self.evaluate_model(model, test_loader)
        subj, x, y = self.extract_data_from_test_set(test_set)

        mse = mean_squared_error(y, y_pred)
        pcc = pearson_corr_coef(y, y_pred)[0]

        self.output_subject_result(subj, x, y, y_pred, mse, pcc, results)


    def output_run_summary(self, sim_name, date, output_res, output_mat, dataset, results, total_time):
        categories = ['AD', 'LMCI', 'MCI', 'EMCI', 'CN', 'Decreasing', 'Increasing']
        
        
        for c in categories:
            cat_reg_err = []
            cat_total_mse = []
            cat_total_pcc = []
            for sub in results.total_reg_err.keys():
                if re.match(rf".*sub-{c}.*", sub):
                    cat_reg_err.append(results.total_reg_err[sub])
                    cat_total_mse.append(results.total_mse[sub])
                    cat_total_pcc.append(results.total_pcc[sub])

            if len(cat_reg_err) == 0:
                continue
            avg_reg_err = np.mean(cat_reg_err, axis=0)
            avg_reg_err_filename = output_res +f'{sim_name}_region_{c}_{date}.png'
            save_avg_regional_errors(avg_reg_err, avg_reg_err_filename)
            np.savetxt(f"{output_mat}{sim_name}_{c}_regions_{date}.csv", avg_reg_err, delimiter=',')
            avg_mse = np.mean(cat_total_mse, axis=0)
            std_mse = np.std(cat_total_mse, axis=0)
            avg_pcc = np.mean(cat_total_pcc, axis=0)
            std_pcc = np.std(cat_total_pcc, axis=0)
            
            results.pt_avg.add_row([c, round(avg_mse, digits), round(std_mse, digits), round(avg_pcc, digits), round(std_pcc, digits)])
        
        if self.params.category not in categories:
            results.pt_avg.add_row([self.params.category, round(np.mean(list(results.total_mse.values())), digits), round(np.std(list(results.total_mse.values())), digits), round(np.mean(list(results.total_pcc.values())), digits), round(np.std(list(results.total_pcc.values())), digits)])
            avg_reg_err = np.mean(list(results.total_reg_err.values()), axis=0)
            avg_reg_err_filename = output_res +f'{sim_name}_region_{self.params.category}_{date}.png'
            save_avg_regional_errors(avg_reg_err, avg_reg_err_filename)
            np.savetxt(f"{output_mat}{sim_name}_{self.params.category}_regions_{date}.csv", avg_reg_err, delimiter=',')

        filename = f"{output_res}{sim_name}_{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}.txt"
        out_file = open(filename, 'w')
        out_file.write(f"Category: {self.params.category}\n")
        out_file.write(f"LR: {self.params.lr}\n")
        out_file.write(f"Epochs: {self.params.epochs}\n")
        out_file.write(f"Hidden1: {self.params.hidden1}\n")
        out_file.write(f"Hidden2: {self.params.hidden2}\n")
        out_file.write(f"Dropout: {self.params.dropout}\n")
        out_file.write(f"Hyperedge value: {self.params.hyperedge_value}\n")
        out_file.write(f"Subjects: {len(dataset.keys())}\n")
        out_file.write(f"Total time (s): {format(total_time, '.2f')}\n")
        out_file.write(results.pt_avg.get_string()+'\n')
        out_file.write(results.pt_subs.get_string())    
        out_file.close()
        logging.info('***********************')
        logging.info(f"Category: {self.params.category}")
        logging.info(f"LR: {self.params.lr}\n")
        logging.info(f"Epochs: {self.params.epochs}\n")
        logging.info(f"Hidden1: {self.params.hidden1}\n")
        logging.info(f"Hidden2: {self.params.hidden2}\n")
        logging.info(f"Dropout: {self.params.dropout}\n")
        logging.info(f"Hyperedge value: {self.params.hyperedge_value}\n")
        logging.info(f"Subjects: {len(dataset.keys())}")
        logging.info(f"Total time (s): {format(total_time, '.2f')}")
        logging.info('***********************')
        logging.info(f"Results saved in {filename}")
        print(f"Results saved in {filename}")

def exec_sim(dataset, category, output_res, output_mat, lr, batch_size, epochs, hidden1, hidden2, dropout, hyperedge_value):

    params = Params(category, lr, batch_size, epochs, hidden1, hidden2, dropout, hyperedge_value)

    hgcn = HGCN(params)

    num_cores = os.cpu_count()
    converted_dataset = hgcn.convert_dataset(dataset, num_cores)

    n_fold = len(converted_dataset.keys())
    test_set_size = 1

    total_time = time()
    results =  Results()
    
    print(f"*** Starting training of {n_fold} folds using {hgcn.device} ***")

    for i in tqdm(range(n_fold)):   
        train_set = {}
        test_set = {}
        
        counter = 0
        for k in converted_dataset.keys():
            # NOTE: dataset keys are subjects paths
            if not os.path.exists(k+'train/'):
                os.makedirs(k+'train/')
            if not os.path.exists(k+'test/'):
                os.makedirs(k+'test/')
            if counter >= i and counter < test_set_size+i:
                test_set[k] = converted_dataset[k]
            else:
                train_set[k] = converted_dataset[k]
            counter += 1    

        hgcn.perform_single_run(train_set, test_set, results)
    
    total_time = time() - total_time

    hgcn.output_run_summary(sim_name, date, output_res, output_mat, dataset, results, total_time)

    avg_mse = round(np.mean(list(results.total_mse.values())), digits)
    return avg_mse


if __name__=="__main__":
    category = 'ALL'

    with open('../../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        

    dataset_path =  config['paths']['dataset_dir'] +  f'datasets/dataset_{category}.json'
    output_res = config['paths']['dataset_dir'] + f'simulations/{category}/results/'
    output_mat = config['paths']['dataset_dir'] + f'simulations/{category}/matrices/'

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)


    exec_sim(dataset, output_res, output_mat) 

    

    