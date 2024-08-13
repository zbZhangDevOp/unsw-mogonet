""" Componets of the model
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import softmax
import os

# Xavier initialise
def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

# GraphConvolution layer
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

# GCN class
class GCN_E(nn.Module):
    def __init__(self, in_dim, hgcn_dim, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, hgcn_dim[0])
        self.gc2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1])
        self.gc3 = GraphConvolution(hgcn_dim[1], hgcn_dim[2])
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        x = F.leaky_relu(x, 0.25)

        return x

# Classifier class
class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x

# VCDN class
class VCDN(nn.Module):
    def __init__(self, num_view, num_cls, hvcdn_dim):
        super().__init__()
        self.num_cls = num_cls
        self.model = nn.Sequential(
            nn.Linear(pow(num_cls, num_view), hvcdn_dim),
            nn.LeakyReLU(0.25),
            nn.Linear(hvcdn_dim, num_cls),
        )
        self.model.apply(xavier_init)

    def forward(self, in_list):
        num_view = len(in_list)
        for i in range(num_view):
            in_list[i] = torch.sigmoid(in_list[i])
        x = torch.reshape(
            torch.matmul(in_list[0].unsqueeze(-1), in_list[1].unsqueeze(1)),
            (-1, pow(self.num_cls, 2), 1),
        )
        for i in range(2, num_view):
            x = torch.reshape(
                torch.matmul(x, in_list[i].unsqueeze(1)),
                (-1, pow(self.num_cls, i + 1), 1),
            )
        vcdn_feat = torch.reshape(x, (-1, pow(self.num_cls, num_view)))
        output = self.model(vcdn_feat)

        return output

# Intialise model_dict
def init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hc, gcn_dopout=0.5):
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i + 1)] = GCN_E(dim_list[i], dim_he_list, gcn_dopout)
        model_dict["C{:}".format(i + 1)] = Classifier_1(dim_he_list[-1], num_class)
    if num_view >= 2:
        model_dict["C"] = VCDN(num_view, num_class, dim_hc)
    return model_dict

# Optimize model_dict
def init_optim(num_view, model_dict, lr_e=1e-4, lr_c=1e-4):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i + 1)] = torch.optim.Adam(
            list(model_dict["E{:}".format(i + 1)].parameters())
            + list(model_dict["C{:}".format(i + 1)].parameters()),
            lr=lr_e,
        )
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    return optim_dict

# GAT layer
class GATConvWithAttention(GATConv):
    def __init__(self, *args, **kwargs):
        super(GATConvWithAttention, self).__init__(*args, **kwargs)
        self.attention_weights = (
            None  # This will store the attention weights after forward pass
        )
        self.last_edge_index = None

    def forward(self, x, edge_index, size=None, return_attention_weights=False):
        # Run the forward pass of the original GATConv
        self.last_edge_index = edge_index
        x, attn = super(GATConvWithAttention, self).forward(
            x, edge_index, size=size, return_attention_weights=True
        )
        self.attention_weights = attn[
            1
        ]  # Assuming attn[1] contains the attention weights
        self.last_edge_index = edge_index
        if return_attention_weights:
            return x, attn
        else:
            return x

    def get_attention_scores(self):
        return self.attention_weights

    def get_last_edge_index(self):
        return self.last_edge_index

# GCN class with GAT as the first layer
class GCN_E_Att(nn.Module):
    def __init__(self, in_dim, hgcn_dim, dropout):
        super(GCN_E_Att, self).__init__()
        # Add a GATConv layer at the beginning to apply attention first
        self.gat_layer = GATConvWithAttention(
            in_dim, hgcn_dim[0], heads=1, concat=False
        )
        self.gc1 = GraphConvolution(hgcn_dim[0], hgcn_dim[1])
        self.gc2 = GraphConvolution(hgcn_dim[1], hgcn_dim[2])
        # self.gc3 = GraphConvolution(hgcn_dim[2], hgcn_dim[2])  # assuming an additional layer for adjusted dimensions
        self.dropout = dropout

    def forward(self, x, adj):
        # Apply GAT layer; assumes adj is in COO format (edge_index) as required by GATConv
        edge_index = (
            adj.coalesce().indices()
        )  # Convert adj to edge_index if it's not already
        x = self.gat_layer(x, edge_index)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        # Follow with graph convolution layers
        x = self.gc1(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.leaky_relu(x, 0.25)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc3(x, adj)
        # x = F.leaky_relu(x, 0.25)
        return x

# Intilise model_dict with attention layer
def init_model_dict_Att(
    num_view, num_class, dim_list, dim_he_list, dim_hc, gcn_dopout=0.5
):
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i + 1)] = GCN_E_Att(
            dim_list[i], dim_he_list, gcn_dopout
        )
        model_dict["C{:}".format(i + 1)] = Classifier_1(dim_he_list[-1], num_class)
    if num_view >= 2:
        model_dict["C"] = VCDN(num_view, num_class, dim_hc)
    return model_dict

# Save models into input directory
def save_models(model_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for name, model in model_dict.items():
        model_path = os.path.join(save_dir, f"{name}.pth")
        torch.save(model.state_dict(), model_path)

# Load GCN model from file_path
def load_gcn_e_model(file_path, in_dim, hgcn_dim, dropout):
    model = GCN_E(in_dim, hgcn_dim, dropout)
    model.load_state_dict(torch.load(file_path))
    return model

# Load GCN model with attention from file_path
def load_gcn_e_att_model(file_path, in_dim, hgcn_dim, dropout):
    model = GCN_E_Att(in_dim, hgcn_dim, dropout)
    model.load_state_dict(torch.load(file_path))
    return model

# Load classifier model from file_path
def load_classifier_1_model(file_path, in_dim, out_dim):
    model = Classifier_1(in_dim, out_dim)
    model.load_state_dict(torch.load(file_path))
    return model

# Load VCDN from file_path
def load_vcdn_model(file_path, num_view, num_cls, hvcdn_dim):
    model = VCDN(num_view, num_cls, hvcdn_dim)
    model.load_state_dict(torch.load(file_path))
    return model

# Load all models and returns model_dict
def load_models(
    save_dir, num_view, num_class, dim_list, dim_he_list, dim_hc, gcn_dropout=0.5
):
    model_dict = {}

    for i in range(num_view):
        model_dict[f"E{i+1}"] = load_gcn_e_att_model(
            os.path.join(save_dir, f"E{i+1}.pth"), dim_list[i], dim_he_list, gcn_dropout
        )
        model_dict[f"C{i+1}"] = load_classifier_1_model(
            os.path.join(save_dir, f"C{i+1}.pth"), dim_he_list[-1], num_class
        )
    if num_view >= 2:
        model_dict["C"] = load_vcdn_model(
            os.path.join(save_dir, "C.pth"), num_view, num_class, dim_hc
        )
    return model_dict
