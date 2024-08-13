""" Training and testing of the model
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
import torch
import torch.nn.functional as F
from models import init_model_dict, init_optim, init_model_dict_Att, save_models
from utils import (
    one_hot_tensor,
    cal_sample_weight,
    gen_adj_mat_tensor,
    gen_test_adj_mat_tensor,
    cal_adj_mat_parameter,
    move_to_beginning,
)
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

cuda = True if torch.cuda.is_available() else False


########################################################################################

# Function for preparing training data only
# Used in backend server
def prepare_tr_data(
    data_folder, input_features, view_list, omics_size, features, labels=None
):
    num_view = len(view_list)
    if labels is None:
        labels_tr = np.loadtxt(
            os.path.join(data_folder, "labels_tr.csv"), delimiter=","
        )
    else:
        labels_tr = np.loadtxt(
            os.path.join(data_folder, str(labels)[:-4] + "_tr.csv"), delimiter=","
        )

    labels_te = np.zeros(input_features.shape[0])
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for feature in features:
        data_tr_list.append(
            np.loadtxt(
                os.path.join(data_folder, str(feature)[:-4] + "_tr.csv"), delimiter=","
            )
        )

    cur_index = 0
    data_te_list = []
    for size in omics_size:
        data_te_list.append(input_features[:, cur_index : cur_index + size])
        cur_index += size

    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_train_list, data_all_list, idx_dict, labels = prepare_trte_data_helper(num_view, data_tr_list, data_te_list, num_tr, num_te, labels_tr, labels_te)
    return data_train_list, data_all_list, idx_dict, labels


########################################################################################

# Function to prepare train and test data
# Used in main_mogonet to replicate result
def prepare_trte_data(
    data_folder, view_list, labels=None, features=["1.csv", "2.csv", "3.csv"]
):
    num_view, data_tr_list, data_te_list, num_tr, num_te, labels_tr, labels_te = get_data_label(data_folder, view_list, labels, features)
    data_train_list, data_all_list, idx_dict, labels = prepare_trte_data_helper(num_view, data_tr_list, data_te_list, num_tr, num_te, labels_tr, labels_te)
    return data_train_list, data_all_list, idx_dict, labels


########################################################################################

# Prepare train and test data for kfold cross validation
def prepare_trte_data_kfold(
    data_folder,
    view_list,
    fold_idx,
    size_folds=-1,
    labels=None,
    features=["1.csv", "2.csv", "3.csv"],
):
    num_view, data_tr_list, data_te_list, num_tr, num_te, labels_tr, labels_te = get_data_label(data_folder, view_list, labels, features)
    if size_folds == -1:
        size_folds = num_te
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    idx_dict = {}
    total_num = num_tr + num_te
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr + num_te)))

    # Calculate the number of folds
    fold_start = fold_idx * size_folds
    indices = range(total_num)
    reordered_indices = move_to_beginning(indices, fold_start, size_folds)

    # Reorder data_tensor_list and labels
    labels = np.concatenate((labels_tr, labels_te))
    data_tensor_list = [data[reordered_indices] for data in data_tensor_list]
    labels = labels[reordered_indices]
    data_train_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(data_tensor_list[i].clone())

    return data_train_list, data_all_list, idx_dict, labels


########################################################################################

# Generate train test adjacency matrix through cosine distance in the given data
# adj_parameter is the average number of edges per node retained including self-edges
def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine"  # cosine distance
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(
            adj_parameter, data_tr_list[i], adj_metric
        )
        adj_train_list.append(
            gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric)
        )
        adj_test_list.append(
            gen_test_adj_mat_tensor(
                data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric
            )
        )

    return adj_train_list, adj_test_list


########################################################################################

# Train MOGONET model for one epoch
def train_epoch(
    data_list,
    adj_list,
    label,
    one_hot_label,
    sample_weight,
    model_dict,
    optim_dict,
    train_VCDN=True,
):
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    for m in model_dict:
        model_dict[m].train()
    num_view = len(data_list)
    for i in range(num_view):
        optim_dict["C{:}".format(i + 1)].zero_grad()
        ci_loss = 0
        ci = model_dict["C{:}".format(i + 1)](
            model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i])
        )
        ci_loss = torch.mean(torch.mul(criterion(ci, label), sample_weight))
        ci_loss.backward()
        optim_dict["C{:}".format(i + 1)].step()
        loss_dict["C{:}".format(i + 1)] = ci_loss.detach().cpu().numpy().item()
    if train_VCDN and num_view >= 2:
        optim_dict["C"].zero_grad()
        c_loss = 0
        ci_list = []
        for i in range(num_view):
            ci_list.append(
                model_dict["C{:}".format(i + 1)](
                    model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i])
                )
            )
        c = model_dict["C"](ci_list)
        c_loss = torch.mean(torch.mul(criterion(c, label), sample_weight))
        c_loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()

    return loss_dict


########################################################################################

# Testing using trained model_dict
# Returns a probability distribution across classes
def test_epoch(data_list, adj_list, te_idx, model_dict):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    for i in range(num_view):
        ci_list.append(
            model_dict["C{:}".format(i + 1)](
                model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i])
            )
        )
    if num_view >= 2:
        c = model_dict["C"](ci_list)
    else:
        c = ci_list[0]
    c = c[te_idx, :]
    prob = F.softmax(c, dim=1).data.cpu().numpy()
    return prob


########################################################################################

# Heatmap visualization with attention after training
def visualize_heatmap_with_attention(edge_index, attention_scores, save_location=None):
    # Convert tensors to numpy arrays
    edge_index_np = edge_index.cpu().numpy()
    attention_scores_np = attention_scores.cpu().detach().numpy().flatten()

    # Determine the number of nodes
    num_nodes = np.max(edge_index_np) + 1

    # Initialize the adjacency matrix
    attention_matrix = np.zeros((num_nodes, num_nodes))

    # Fill the adjacency matrix with attention scores
    for i, (src, dst) in enumerate(edge_index_np.T):
        attention_matrix[src, dst] = attention_scores_np[i]

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(attention_matrix, cmap="viridis", annot=False, fmt=".2f")
    plt.title("Attention Score Heatmap")
    plt.xlabel("Destination Node")
    plt.ylabel("Source Node")
    if save_location:
        plt.savefig(os.path.join(save_location, "heatmap.png"))
    else:
        plt.show()
    plt.close()


########################################################################################

# Attention graph visualization after training
def visualize_graph_with_attention(edge_index, attention_scores, save_location=None):
    G = nx.DiGraph()
    edge_index_np = edge_index.cpu().numpy()
    attention_scores_np = attention_scores.cpu().detach().numpy().flatten()

    # Add nodes
    num_nodes = np.max(edge_index_np) + 1
    G.add_nodes_from(range(num_nodes))

    # Add edges with attention weights
    for i, (src, dst) in enumerate(edge_index_np.T):
        if src != dst:
            G.add_edge(src, dst, weight=attention_scores_np[i])

    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    pos = nx.spring_layout(G)
    edges = G.edges(data=True)
    edge_colors = [data["weight"] for _, _, data in edges]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=30)

    # Draw edges with straight arrows
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges,
        edge_color=edge_colors,
        arrowstyle="-|>",
        arrowsize=10,
        width=1,
        edge_cmap=plt.cm.Blues,
        connectionstyle="arc3, rad=0.2",
    )

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=7)

    plt.title("Graph Visualization with Attention Scores")
    plt.colorbar(
        plt.cm.ScalarMappable(cmap=plt.cm.Blues),
        ax=plt.gca(),
        orientation="vertical",
        label="Attention Scores",
    )
    if save_location:
        plt.savefig(os.path.join(save_location, "graph.png"))
    else:
        plt.show()
    plt.close()


########################################################################################

# train_test function that splits 
def train_test(
    data_folder,
    view_list,
    num_class,
    lr_e_pretrain,
    lr_e,
    lr_c,
    num_epoch_pretrain,
    num_epoch,
    features=["1.csv", "2.csv", "3.csv"],
    adj_parameter=-1,
    do_print=True,
    save_model=False,
    model_info=None,
    dim_he_list=None,
    labels=None,
    attention=True,
    visualisation=False,
):
    # Parameters in the original mogonet study
    test_inverval = 50
    num_view = len(view_list)
    dim_hvcdn = pow(num_class, num_view)

    if "ROSMAP" in data_folder:
        adj_parameter = 2 if adj_parameter == -1 else adj_parameter
        dim_he_list = [200, 200, 100]
    elif "BRCA" in data_folder:
        adj_parameter = 10 if adj_parameter == -1 else adj_parameter
        dim_he_list = [400, 400, 200]
    elif dim_he_list is None:
        dim_he_list = [400, 400, 200]

    results = []

    # Prepare data and training tensor
    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(
        data_folder, view_list, labels, features
    )
    adj_tr_list, adj_te_list, labels_tr_tensor, onehot_labels_tr_tensor, sample_weight_tr = prepare_training_tensor(
        num_class, data_tr_list, data_trte_list, labels_trte, trte_idx, adj_parameter
    )

    # Pretrain model
    model_dict, optim_dict, dim_list = pretrain_model(
        num_view, num_class, dim_he_list, dim_hvcdn, data_tr_list,
        adj_tr_list, labels_tr_tensor, onehot_labels_tr_tensor,
        sample_weight_tr, num_epoch_pretrain, lr_e_pretrain, lr_c, lr_e,
        attention,
    )

    for epoch in range(num_epoch + 1):
        train_epoch(
            data_tr_list,
            adj_tr_list,
            labels_tr_tensor,
            onehot_labels_tr_tensor,
            sample_weight_tr,
            model_dict,
            optim_dict,
        )
        if epoch % test_inverval == 0:
            te_prob = test_epoch(
                data_trte_list, adj_te_list, trte_idx["te"], model_dict
            )
            test_results = record_test_interval(num_class, labels_trte, trte_idx, te_prob, epoch, do_print)
            results.append(test_results)
        
        # Create attention visualisation after training
        if attention and epoch == num_epoch:
            for model_key in model_dict:
                if hasattr(model_dict[model_key], "gat_layer"):
                    attention_scores = model_dict[
                        model_key
                    ].gat_layer.get_attention_scores()
                    edge_index = model_dict[model_key].gat_layer.get_last_edge_index()
                    if visualisation:
                        visualize_graph_with_attention(edge_index, attention_scores)
                        visualize_heatmap_with_attention(edge_index, attention_scores)

    # Update model_params in backend server
    model_params = {
        "num_view": num_view,
        "num_class": num_class,
        "dim_list": dim_list,
        "dim_he_list": dim_he_list,
        "dim_hvcdn": dim_hvcdn,
        "lr_e_pretrain": lr_e_pretrain,
        "lr_e": lr_e,
        "lr_c": lr_c,
    }

    if model_info is not None:
        model_info.update(model_params)
        visualize_graph_with_attention(edge_index, attention_scores, data_folder)
        visualize_heatmap_with_attention(edge_index, attention_scores, data_folder)
    if save_model:
        save_models(model_dict, data_folder)
    results_df = pd.DataFrame(results)

    return results_df, model_info


########################################################################################

# train_test function with cross validation
# Compute test metrics using k-fold cross validation
def train_test_cross_validation(
    data_folder,
    view_list,
    num_class,
    lr_e_pretrain,
    lr_e,
    lr_c,
    num_epoch_pretrain,
    num_epoch,
    features=["1.csv", "2.csv", "3.csv"],
    num_folds=3,
    adj_parameter=-1,
    do_print=True,
    dim_he_list=None,
    labels=None,
    attention=True,
):
    # Parameters in the original mogonet study
    test_interval = 50
    num_view = len(view_list)
    dim_hvcdn = pow(num_class, num_view)

    if "ROSMAP" in data_folder:
        adj_parameter = 2 if adj_parameter == -1 else adj_parameter
        dim_he_list = [200, 200, 100]
    elif "BRCA" in data_folder:
        adj_parameter = 10 if adj_parameter == -1 else adj_parameter
        dim_he_list = [400, 400, 200]
    elif dim_he_list is None:
        dim_he_list = [400, 400, 200]

    results = []
    for fold_idx in range(num_folds):
        # Prepare kfold data and training tensor
        data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data_kfold(
            data_folder, view_list, fold_idx, labels=labels, features=features
        )
        adj_tr_list, adj_te_list, labels_tr_tensor, onehot_labels_tr_tensor, sample_weight_tr = prepare_training_tensor(
            num_class, data_tr_list, data_trte_list, labels_trte, trte_idx, adj_parameter
        )
        
        # Pretrain model
        model_dict, optim_dict, dim_list = pretrain_model(
            num_view, num_class, dim_he_list, dim_hvcdn, data_tr_list,
            adj_tr_list, labels_tr_tensor, onehot_labels_tr_tensor,
            sample_weight_tr, num_epoch_pretrain, lr_e_pretrain, lr_c, lr_e,
            attention,
        )

        for epoch in range(num_epoch + 1):
            train_epoch(
                data_tr_list,
                adj_tr_list,
                labels_tr_tensor,
                onehot_labels_tr_tensor,
                sample_weight_tr,
                model_dict,
                optim_dict,
            )
            if epoch % test_interval == 0:
                te_prob = test_epoch(
                    data_trte_list, adj_te_list, trte_idx["te"], model_dict
                )
                test_results = record_test_interval(num_class, labels_trte, trte_idx, te_prob, epoch, do_print)
                test_results["fold_idx"] = fold_idx
                results.append(test_results)

    results_df = pd.DataFrame(results)
    return results_df


########################################################################################
# Utility functions

# Utility function for prepare_trte_data
# Prepare data and labels
def get_data_label(data_folder, view_list, labels, features):
    num_view = len(view_list)
    if labels is None:
        labels_tr = np.loadtxt(
            os.path.join(data_folder, "labels_tr.csv"), delimiter=","
        )
        labels_te = np.loadtxt(
            os.path.join(data_folder, "labels_te.csv"), delimiter=","
        )
    else:
        labels_tr = np.loadtxt(
            os.path.join(data_folder, str(labels)[:-4] + "_tr.csv"), delimiter=","
        )
        labels_te = np.loadtxt(
            os.path.join(data_folder, str(labels)[:-4] + "_te.csv"), delimiter=","
        )
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for feature in features:
        data_tr_list.append(
            np.loadtxt(
                os.path.join(data_folder, str(feature)[:-4] + "_tr.csv"), delimiter=","
            )
        )
        data_te_list.append(
            np.loadtxt(
                os.path.join(data_folder, str(feature)[:-4] + "_te.csv"), delimiter=","
            )
        )
    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    
    return num_view, data_tr_list, data_te_list, num_tr, num_te, labels_tr, labels_te


# Utility function to generate training and testing tensors as well as test indices and labels
# Used in prepare_tr_data and prepare_trte_data to reduce cold repetition
def prepare_trte_data_helper(num_view, data_tr_list, data_te_list, num_tr, num_te, labels_tr, labels_te):
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr + num_te)))
    data_train_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(
            torch.cat(
                (
                    data_tensor_list[i][idx_dict["tr"]].clone(),
                    data_tensor_list[i][idx_dict["te"]].clone(),
                ),
                0,
            )
        )
    labels = np.concatenate((labels_tr, labels_te))
    return data_train_list, data_all_list, idx_dict, labels


# Utiltiy function to prepare training_tensor in train_test
# Used in train_test and train_test_cross_validation
def prepare_training_tensor(num_class, data_tr_list, data_trte_list, labels_trte, trte_idx, adj_parameter):
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)

    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()
    adj_tr_list, adj_te_list = gen_trte_adj_mat(
        data_tr_list, data_trte_list, trte_idx, adj_parameter
    )
    return adj_tr_list, adj_te_list, labels_tr_tensor, onehot_labels_tr_tensor, sample_weight_tr


# Utility function to pretrain model for num_epoch_pretrain epochs
# Used in train_test and train_test_cross_validation
def pretrain_model(
    num_view, num_class, dim_he_list, dim_hvcdn, data_tr_list,
    adj_tr_list, labels_tr_tensor, onehot_labels_tr_tensor,
    sample_weight_tr, num_epoch_pretrain, lr_e_pretrain, lr_c, lr_e,
    attention,
):
    dim_list = [x.shape[1] for x in data_tr_list]
    if attention:
        model_dict = init_model_dict_Att(
            num_view, num_class, dim_list, dim_he_list, dim_hvcdn
        )
    else:
        model_dict = init_model_dict(
            num_view, num_class, dim_list, dim_he_list, dim_hvcdn
        )
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    for epoch in range(num_epoch_pretrain):
        train_epoch(
            data_tr_list,
            adj_tr_list,
            labels_tr_tensor,
            onehot_labels_tr_tensor,
            sample_weight_tr,
            model_dict,
            optim_dict,
            train_VCDN=False,
        )
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)
    return model_dict, optim_dict, dim_list


# Utility function to record test metrics every test intervals
# Used in train_test and train_test_cross_validation
def record_test_interval(num_class, labels_trte, trte_idx, te_prob, epoch, do_print):
    test_results = {}
    if num_class == 2:
        test_results[f"Test ACC"] = accuracy_score(
            labels_trte[trte_idx["te"]], te_prob.argmax(1)
        )
        test_results[f"Test F1"] = f1_score(
            labels_trte[trte_idx["te"]], te_prob.argmax(1)
        )
        test_results[f"Test AUC"] = roc_auc_score(
            labels_trte[trte_idx["te"]], te_prob[:, 1]
        )
    else:
        test_results[f"Test ACC"] = accuracy_score(
            labels_trte[trte_idx["te"]], te_prob.argmax(1)
        )
        test_results[f"Test F1 weighted"] = f1_score(
            labels_trte[trte_idx["te"]], te_prob.argmax(1), average="weighted",
        )
        test_results[f"Test F1 macro"] = f1_score(
            labels_trte[trte_idx["te"]], te_prob.argmax(1), average="macro"
        )
    test_results["epoch"] = epoch
    if do_print:
        print_df = pd.DataFrame(test_results, index=[0])
    return test_results
