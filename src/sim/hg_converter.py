import networkx as nx

import torch

from concurrent.futures import ProcessPoolExecutor, as_completed
from time import time
from torch_geometric.data import Data
from utils import *
from utils_vis import *

class HG_Converter():
    def __init__(self, hyperedge_value):
        self.hyperedge_value = hyperedge_value
        
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
            row = torch.tensor([node for clique in hyperedges for node in clique])
            col = torch.arange(num_hyperedges).repeat_interleave(torch.tensor([len(clique) for clique in hyperedges])) + num_nodes

            edge_index = torch.stack([row, col], dim=0)

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

        if self.hyperedge_value == "zeros":
            hyperedges_init_value = torch.zeros((num_hyperedges, 1))
        elif self.hyperedge_value == "proportional":
            hyperedges_init_value = torch.full((num_hyperedges, 1), 1/3)
        else:
            hyperedges_init_value = torch.ones((num_hyperedges, 1))

        return hyperedges_init_value