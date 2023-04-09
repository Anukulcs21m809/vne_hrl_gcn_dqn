import torch
import torch.nn as nn
import numpy as np
import torch_geometric.utils.convert as cvt
import torch.linalg as linalg
import torch.nn.functional as fn
from neural_networks.higher_network import GCNLayer

class Lower_network(nn.Module):
    def __init__(self, g1_in_features, g1_hidden_features, g1_out_features, ff_hidden_size, num_classes, num_features_of_ffnn):
        super(Lower_network, self).__init__()
        self.g1_layer1 = GCNLayer(g1_in_features, g1_hidden_features)
        self.g1_layer2 = GCNLayer(g1_hidden_features, g1_out_features)
        self.ff_layer1 = nn.Linear(num_features_of_ffnn, ff_hidden_size)
        self.ff_layer2 = nn.Linear(ff_hidden_size, num_classes)
    
    # we get the sub data as a data object and the vnr data as the tensor with the features of the node that requires embedding
    def forward(self, sub_data, vnr_data):
        g1_adj_matrix = self.remake_adj(sub_data.edge_index)
        g1_x = self.g1_layer1(sub_data.x, g1_adj_matrix)
        g1_x = self.g1_layer2(g1_x, g1_adj_matrix)

        # this should be length 23 if the output dimension of the g1_x is 1D (n_nodes, 1)
        x = torch.cat([g1_x.flatten(), vnr_data])

        x = x.unsqueeze(0)
        # Apply the feedforward layers
        x = torch.tanh(self.ff_layer1(x))
        x = self.ff_layer2(x)

        return x

    def remake_adj(self, adj_matrix):
        adj_matrix = cvt.to_scipy_sparse_matrix(adj_matrix).todense()
        adj_matrix = torch.tensor(adj_matrix + np.eye(adj_matrix.shape[0]))

        return adj_matrix