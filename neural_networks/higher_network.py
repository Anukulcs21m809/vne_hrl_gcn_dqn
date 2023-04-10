import torch
import torch.nn as nn
import numpy as np
import torch_geometric.utils.convert as cvt
import torch.linalg as linalg
import torch.nn.functional as fn

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))

    def forward(self, x, adj_matrix):
        # x is the input feature matrix of shape (num_nodes, in_features)
        # adj_matrix is the adjacency matrix of shape (num_nodes, num_nodes)
        # add self loops by first converting it to a N * N matrix
        
        # adj_matrix = cvt.to_scipy_sparse_matrix(adj_matrix).todense()
        # adj_matrix = torch.tensor(adj_matrix + np.eye(adj_matrix.shape[0]))

        # First, we compute the normalized adjacency matrix
        deg = torch.sum(adj_matrix, dim=1)
        deg = torch.tensor(np.diag(deg))
        deg_inv_sqrt = linalg.matrix_power(torch.sqrt(deg), -1)
        norm_adj_matrix =  deg_inv_sqrt * adj_matrix * deg_inv_sqrt

        # Next, we compute the GCN output
        x = torch.matmul(norm_adj_matrix.float(), x)
        x = torch.matmul(x, self.weight) + self.bias
        x = torch.tanh(x)

        return x

class Higher_network(nn.Module):
    def __init__(self, g1_in_features, g1_hidden_features, g1_out_features,
                 g2_in_features, g2_hidden_features, g2_out_features, ff_hidden_size, num_classes, num_features_of_ffnn):
        super(Higher_network, self).__init__()
        self.num_features_ffnn = num_features_of_ffnn
        self.g1_layer1 = GCNLayer(g1_in_features, g1_hidden_features)
        self.g1_layer2 = GCNLayer(g1_hidden_features, g1_out_features)
        self.g2_layer1 = GCNLayer(g2_in_features, g2_hidden_features)
        self.g2_layer2 = GCNLayer(g2_hidden_features, g2_out_features)
        self.ff_layer1 = nn.Linear(num_features_of_ffnn, ff_hidden_size)
        self.ff_layer2 = nn.Linear(ff_hidden_size, num_classes)

    def forward(self, sub_data, vnr_data):
        # g1_x is the feature matrix of the first graph of shape (g1_num_nodes, g1_in_features)
        # g1_adj_matrix is the adjacency matrix of the first graph of shape (g1_num_nodes, g1_num_nodes)
        # g2_x is the feature matrix of the second graph of shape (g2_num_nodes, g2_in_features)
        # g2_adj_matrix is the adjacency matrix of the second graph of shape (g2_num_nodes, g2_num_nodes)
        # Apply the GCN layers to the first graph

        g1_x = torch.cat([sub_data[i].x for i in range(len(sub_data))], dim=0)
        sub_data_adj = [self.remake_adj(sub_data[x].edge_index) for x in range(len(sub_data))]
        g1_adj_matrix = torch.block_diag(*sub_data_adj)

        g1_x = self.g1_layer1(g1_x, g1_adj_matrix)
        g1_x = self.g1_layer2(g1_x, g1_adj_matrix)
        
        # Apply the GCN layers to the second graph
        g2_adj_matrix = self.remake_adj(vnr_data.edge_index)
        g2_x = self.g2_layer1(vnr_data.x, g2_adj_matrix)
        g2_x = self.g2_layer2(g2_x, g2_adj_matrix)

        # Concatenate the outputs of the GCN layers
        x = torch.cat((g1_x, g2_x), dim=0)
        x = x.flatten()
        pad_seq = (0, 1)
        while x.shape[0] < self.num_features_ffnn:
            x = fn.pad(x, pad_seq, "constant", 0)

        x = x.unsqueeze(0)
        # Apply the feedforward layers
        x = torch.tanh(self.ff_layer1(x))
        x = self.ff_layer2(x)

        return x
    
    def remake_adj(self, adj_matrix):
        adj_matrix = cvt.to_scipy_sparse_matrix(adj_matrix).todense()
        adj_matrix = torch.tensor(adj_matrix + np.eye(adj_matrix.shape[0]))

        return adj_matrix

# train_dataset_sub = VNEDataset('data/', sub=1, filename="sub_graphs_train.csv")
# train_dataset_vnr = VNEDataset('data/', sub=0, filename="vnr_graphs_train.csv")
# num_sub_graphs = 3
# num_nodes_sub = 20
# num_highest_vnr_nodes = 8
# num_features_of_ffnn = (num_nodes_sub * num_sub_graphs) + num_highest_vnr_nodes
# g1_in_ftrs = 6
# g1_h_ftrs = 2
# g1_o_ftrs = 1
# g2_in_ftrs = 3
# g2_h_ftrs = 2
# g2_o_ftrs = 1
# ff_hidden_size = 128
# num_classes = 3

# all_sub_data = [next(iter(train_dataset_sub)) for x in range(num_classes)]
# vnr_data = next(iter(train_dataset_vnr))
# model = Higher_network(g1_in_ftrs, g1_h_ftrs, g1_o_ftrs, g2_in_ftrs, g2_h_ftrs, g2_o_ftrs, ff_hidden_size, num_classes, num_features_of_ffnn)
# # all_sub_data is a list of all the substrate's data objects
# print(model.forward(all_sub_data , vnr_data))

# all_data = torch.stack([all_sub_data[i].x for i in range(num_classes)])
# print(all_data)
# print(all_data.shape)

# # create a 3x3 block diagonal matrix with the adjacency matrices on the diagonal
# diag = torch.block_diag(A, B, C)
