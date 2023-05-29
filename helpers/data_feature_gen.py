import numpy as np
import torch
from torch_geometric.data import Data

#collects all the node features for the substrate and vnr graphs
class Featurizer:
    
    @staticmethod
    def featurize(G, sub=1):
        all_node_features = []
        for x in range(len(G['nodes'])):
            all_features = []
            if(sub):
                all_features = Featurizer.featurize_sub_node(G, x)
            else:
                all_features = Featurizer.featurize_vnr_node(G, x)
            all_node_features.append(list(all_features))
        all_node_features = np.asarray(all_node_features)
        return torch.tensor(all_node_features, dtype=torch.float32)

    @staticmethod
    def featurize_sub_node(G, node_idx):
        cpu = G['nodes'][node_idx]['cpu'] 
        mem = G['nodes'][node_idx]['mem'] 
        cpu_util = 1 - (cpu / G['nodes'][node_idx]['cpu_max'])
        mem_util = 1 - (mem / G['nodes'][node_idx]['mem_max'])
        avg_rem_band = 0
        avg_util_band = 0
        i = 0
        for link in G['links']:
            if node_idx == link['target'] or node_idx == link['source']:
                avg_rem_band += link['bw']
                avg_util_band += (1 - (link['bw'] / link['band_max']))
                i += 1
        avg_rem_band = avg_rem_band / i if i > 0 else avg_rem_band
        avg_util_band = avg_util_band / i if i > 0 else avg_util_band

        return (cpu, mem, cpu_util, mem_util, avg_rem_band, avg_util_band)
    
    # all these max values are from the resource range that we have specified earlier for the vnr 
    @staticmethod
    def featurize_vnr_node(G, node_idx):
        cpu = G['nodes'][node_idx]['cpu']
        mem = G['nodes'][node_idx]['mem']
        i = 0
        avg_band = 0
        for link in G['links']:
            if node_idx == link['target'] or node_idx == link['source']:
                avg_band += link['bw']
                i += 1
        avg_band = avg_band / i if i > 0 else avg_band

        return (cpu, mem, avg_band)

    @staticmethod
    def get_adj_info(G):
        edge_indices = []
        for edge in G['links']:
            i = edge['source']
            j = edge['target']
            edge_indices += [[i, j], [j, i]]
        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices
    
    @staticmethod
    def make_data_obj(G, sub=1):
        node_info = Featurizer.featurize(G, sub=sub)
        edge_info = Featurizer.get_adj_info(G)
        data_ = Data(
                x = node_info,
                edge_index=edge_info
            )
        return data_