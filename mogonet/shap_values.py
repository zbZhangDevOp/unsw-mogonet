import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from models import init_model_dict, init_optim, init_model_dict_Att
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter
from train_test import prepare_trte_data, gen_trte_adj_mat, train_epoch, test_epoch
import shap

cuda = True if torch.cuda.is_available() else False

########################################################################################
# Parameters and datasets

# Hyperparameters from original mogonet study
data_folder = 'ROSMAP'
view_list = [1,2,3]
num_epoch_pretrain = 500
num_epoch = 2500
num_view = len(view_list)
lr_e_pretrain = 1e-3
lr_e = 5e-4
lr_c = 1e-3
num_class = 2

# Prepare background datasets
data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter=2)
X_combined = np.concatenate([data_trte_list[0].cpu().numpy(), data_trte_list[1].cpu().numpy(), data_trte_list[2].cpu().numpy()], axis=1)
featnames = []
for i in view_list:
    featname_i = pd.read_csv(os.path.join(data_folder, str(i)+"_featname.csv"), delimiter=',')
    featnames = featnames + [featname_i.columns[0]] + featname_i.iloc[:, 0].tolist()


########################################################################################
# Train and test functions

# Train model for shap
def train_shap(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c, 
               num_epoch_pretrain, num_epoch, adj_parameter=-1):
    num_view = len(view_list)
    dim_hvcdn = pow(num_class,num_view)
    if data_folder == 'ROSMAP':
        adj_parameter = 2 if adj_parameter == -1 else adj_parameter
        dim_he_list = [200,200,100]
    if data_folder == 'BRCA':
        adj_parameter = 10 if adj_parameter == -1 else adj_parameter
        dim_he_list = [400,400,200]
    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
    dim_list = [x.shape[1] for x in data_tr_list]
    # Adapt to attention model
    model_dict = init_model_dict_Att(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()
    
    ("\nPretrain GCNs...")
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    for epoch in range(num_epoch_pretrain):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, 
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_VCDN=False)
    print("\nTraining...")
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)
    for epoch in range(num_epoch+1):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, 
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)
    return model_dict

# Prepare test data
# test_data has 600 features (concatenation of 3 omics data) here
def prepare_test_data(test_data):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.zeros(test_data.shape[0]) # padding 0's
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_tr.csv"), delimiter=','))
    num_samples = test_data.shape[0]
    omics1_size, omics2_size, omics3_size = 200, 200, 200
    omics1, omics2, omics3 = test_data[:, :omics1_size], test_data[:, omics1_size:omics1_size+omics2_size], test_data[:, omics1_size+omics2_size:]
    data_te_list = [omics1, omics2, omics3]
    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
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
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))
    data_train_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                       data_tensor_list[i][idx_dict["te"]].clone()),0))
    labels = np.concatenate((labels_tr, labels_te))
    
    return data_train_list, data_all_list, idx_dict, labels

# Test shap function to feed into KernelExplainer
# Test requires build a new adjacency matrix with the test sample added to the training set
def test_shap(test_data):
	data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_test_data(test_data)
	adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter=2)
	return test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)


########################################################################################
# Computing SHAP values

# Train model
model_dict = train_shap(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c, 
               num_epoch_pretrain, num_epoch) 

# SHAP Kernel Explainer
explainer = shap.KernelExplainer(test_shap, X_combined[0:3])
shap_values = explainer(X_combined) # Explains all training samples to form distribution plots

# Provide feature names
shap_values.feature_names = featnames

# waterfall plot
plt.figure(figsize=(20, 5))  # Width=20, Height=5
shap.plots.waterfall(shap_values[0, :, 0], max_display=10, show=False)
plt.savefig("shap_waterfall.png", dpi=300, bbox_inches="tight")
plt.close()

# violin plot
plt.figure(figsize=(20, 5))  # Width=20, Height=5
shap.plots.violin(shap_values[:,:,0], max_display=10, feature_names=featnames, show=False)
plt.savefig("shap_violin.png", dpi=300, bbox_inches="tight")
plt.close()

# beeswarm plot
plt.figure(figsize=(20, 5))  # Width=20, Height=5
shap.plots.beeswarm(shap_values[:,:,0], max_display=11, show=False)
plt.savefig("shap_beeswarm.png", dpi=300, bbox_inches="tight")
plt.close()
