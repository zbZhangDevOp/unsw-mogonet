""" Example for MOGONET classification
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

from train_test import train_test, train_test_cross_validation

if __name__ == "__main__":    
    data_folder = 'ROSMAP'
    view_list = [1,2,3]
    num_epoch_pretrain = 1000
    num_epoch = 2500 
    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-3
    
   
    if data_folder == 'ROSMAP':
        num_class = 2
    if data_folder == 'BRCA':
        num_class = 5
    
    # This will train MOGONET and display an attention graph using the train data
    # The test metric results will also be returned in a dataframe as the first return entry
    # Second return entry is used for the backend server - don't use
    train_test(data_folder, view_list, num_class,
                lr_e_pretrain, lr_e, lr_c, num_epoch_pretrain,
                num_epoch, do_print=True, attention=True, save_model=False, visualisation=True)

    # Test metric results using cross validation - uncomment if want to use
    # train_test_cross_validation(data_folder, view_list, num_class,
    #                                 lr_e_pretrain, lr_e, lr_c,
    #                                 num_epoch_pretrain, num_epoch,
    #                                 num_folds=3, do_print=True,
    #                                 attention = True)


    ################################################################################
    # Helper functions for main_mogonet.py
    # Includes:
    #    - Search hyperparmeter using validation_set
    #    - Search hyperparameter using cross_validation
    # Example usage are at the end of this file
    
    # Test metrics using validation set
    def main_validation_set(attention=False):
        adj_parameters = range(1, 11)
        num_repeats = 3
        save_interval = 3  # Number of repeats after which to save results
        results = []
        all_results = []
        for adj_parameter in adj_parameters:
            for repeat in range(num_repeats):
                print(f"Running test with adj_parameter = {adj_parameter}... (Iteration {repeat + 1}/{num_repeats})")
                if attention:
                    results_df, _ = train_test(data_folder, view_list, num_class,
                                        lr_e_pretrain, lr_e, lr_c, num_epoch_pretrain,
                                        num_epoch, adj_parameter=adj_parameter, do_print=True)
                else:
                    results_df, _ = train_test(data_folder, view_list, num_class,
                                        lr_e_pretrain, lr_e, lr_c, num_epoch_pretrain,
                                        num_epoch, adj_parameter=adj_parameter, attention=False)

                results_df['adj_parameter'] = adj_parameter
                results_df['repeat_idx'] = repeat
                results.append(results_df)

                if (repeat + 1) % save_interval == 0 or repeat + 1 == num_repeats:
                    os.makedirs('results', exist_ok=True)
                    os.makedirs(f'results/{data_folder}', exist_ok=True)
                    intermediate_results_df = pd.concat(results, ignore_index=True)
                    intermediate_results_df.to_csv(f'results/{data_folder}/results_{adj_parameter}_{repeat + 1}.csv', index=False)
                    results.clear()

        if results:
            final_results_df = pd.concat(results, ignore_index=True)
            final_results_df.to_csv(f'results/{data_folder}/results_final.csv', index=False)
        
        all_results_df = pd.concat(
            [pd.read_csv(f'results/{data_folder}/results_{adj_parameter}_{i}.csv') 
            for adj_parameter in adj_parameters 
            for i in range(save_interval, num_repeats + save_interval, save_interval) 
            if pd.read_csv(f'results/{data_folder}/results_{adj_parameter}_{i}.csv').shape[0] > 0], 
            ignore_index=True
        )
        avg_metrics_df = all_results_df.groupby(['adj_parameter', 'epoch']).mean().reset_index()
        avg_metrics_df.drop(columns=['repeat_idx'], inplace=True)
        all_results_df.to_csv(f'results/{data_folder}/validation_set_results.csv', index=False)
        if attention:
            avg_metrics_df.to_csv(f'results/{data_folder}/validation_set_avg_metrics_attention.csv', index=False)
        else:
            avg_metrics_df.to_csv(f'results/{data_folder}/validation_set_avg_metrics_normal.csv', index=False)


    ################################################################################

    # Test metrics using k-fold cross validation
    def main_kfold_cross_validation(plot=False, attention=False):
        adj_parameters = range(1, 11)
        num_repeats = 5
        save_interval = 5  # Number of repeats after which to save results
        num_folds = 3
        grid_search_results = []
        all_results = []
        for adj_parameter in adj_parameters:
            for repeat in range(num_repeats):
                print(f"Running grid search with adj_parameter = {adj_parameter}... (Iteration {repeat + 1}/{num_repeats})")
                if attention:
                    results_df = train_test_cross_validation(data_folder, view_list, num_class,
                                                lr_e_pretrain, lr_e, lr_c,
                                                num_epoch_pretrain, num_epoch, 
                                                num_folds=num_folds, do_print=True,
                                                adj_parameter=adj_parameter, attention = True)
                else:
                    results_df = train_test_cross_validation(data_folder, view_list, num_class,
                                                lr_e_pretrain, lr_e, lr_c,
                                                num_epoch_pretrain, num_epoch, 
                                                num_folds=num_folds, do_print=True,
                                                adj_parameter=adj_parameter, attention = False)
                results_df['adj_parameter'] = adj_parameter
                results_df['repeat_idx'] = repeat
                grid_search_results.append(results_df)

                if (repeat + 1) % save_interval == 0 or repeat + 1 == num_repeats:
                    os.makedirs('results', exist_ok=True)
                    os.makedirs(f'results/{data_folder}', exist_ok=True)
                    intermediate_results_df = pd.concat(grid_search_results, ignore_index=True)
                    intermediate_results_df.to_csv(f'results/{data_folder}/grid_search_results_{adj_parameter}_{repeat + 1}.csv', index=False)
                    grid_search_results.clear() 

        if grid_search_results:
            final_results_df = pd.concat(grid_search_results, ignore_index=True)
            final_results_df.to_csv(f'results/{data_folder}/grid_search_results_final.csv', index=False)
        
        all_results_df = pd.concat(
            [pd.read_csv(f'results/{data_folder}/grid_search_results_{adj_parameter}_{i}.csv') 
            for adj_parameter in adj_parameters 
            for i in range(save_interval, num_repeats + save_interval, save_interval) 
            if pd.read_csv(f'results/{data_folder}/grid_search_results_{adj_parameter}_{i}.csv').shape[0] > 0], 
            ignore_index=True
        )
        avg_metrics_df = all_results_df.groupby(['adj_parameter', 'epoch']).mean().reset_index()
        avg_metrics_df.drop(columns=['repeat_idx', 'fold_idx'], inplace=True)
        all_results_df.to_csv(f'results/{data_folder}/grid_search_results.csv', index=False)
        avg_metrics_df.to_csv(f'results/{data_folder}/avg_metrics_attention.csv', index=False) 
        
        if plot:
            data = pd.read_csv(f'results/{data_folder}/avg_metrics_attention.csv')
            data = data[data['epoch'] == 2500]
            data = data[data['adj_parameter'] != 1]
            k_values = data['adj_parameter']
            acc_values = data['Test ACC']
            f1_values = data['Test F1']
            auc_values = data['Test AUC']

            plt.figure(figsize=(10, 6))

            plt.plot(k_values, acc_values, 'o-', label='ACC')
            plt.plot(k_values, f1_values, 'o-', label='F1')
            plt.plot(k_values, auc_values, 'o-', label='AUC')

            plt.xlabel('adj_parameter')
            plt.ylabel('value')
            plt.legend()
            plt.grid(True)
            plt.title('Performance Metrics Across Different k Values')

            plt.show()


    ################################################################################

    # Plot Performance Metrics of the original MOGONAT against the model where the first layer
    # of GCNs are replaced with GAT
    def adj_parameter_plot():
        data = pd.read_csv(f'results/{data_folder}/validation_set_avg_metrics_attention.csv')
        data = data[data['epoch'] == num_epoch]
        data = data[data['adj_parameter'] != 1]
        k_values = data['adj_parameter']
        acc_values = data['Test ACC']
        f1_values = data['Test F1']
        auc_values = data['Test AUC']
        
        data2 = pd.read_csv(f'results/{data_folder}/validation_set_avg_metrics_normal.csv')
        data2 = data2[data2['epoch'] == num_epoch]
        data2 = data2[data2['adj_parameter'] != 1]
        k_values_2 = data2['adj_parameter']
        acc_values_2 = data2['Test ACC']
        f1_values_2 = data2['Test F1']
        auc_values_2 = data2['Test AUC']

        plt.figure(figsize=(10, 6))

        # Plot MOGONET with GAT
        plt.plot(k_values, acc_values, 'o-', label='ACC_Att')
        plt.plot(k_values, f1_values, 'o-', label='F1_Att')
        plt.plot(k_values, auc_values, 'o-', label='AUC_Att')
        
        # Plot original MOGONET
        plt.plot(k_values_2, acc_values_2, 'o-', label='ACC_None')
        plt.plot(k_values_2, f1_values_2, 'o-', label='F1_None')
        plt.plot(k_values_2, auc_values_2, 'o-', label='AUC_None')

        plt.xlabel('adj_parameter')
        plt.ylabel('value')
        plt.legend()
        plt.grid(True)
        plt.title('Performance Metrics Across Different k Values')

        plt.show()


    ################################################################################
    
    # Plot results against epoch
    def epoch_plot():
        data = pd.read_csv(f'results/{data_folder}/validation_set_avg_metrics_attention.csv')
        data = data[data['adj_parameter'] == 2]
        k_values = data['epoch']
        acc_values = data['Test ACC']
        f1_values = data['Test F1']
        auc_values = data['Test AUC']

        data2 = pd.read_csv(f'results/{data_folder}/validation_set_avg_metrics_normal.csv')
        data2 = data2[data2['adj_parameter'] == 2]
        k_values_2 = data2['epoch']
        acc_values_2 = data2['Test ACC']
        f1_values_2 = data2['Test F1']
        auc_values_2 = data2['Test AUC']

        plt.figure(figsize=(10, 6))
        plt.plot(k_values, acc_values, 'o-', label='ACC_Att')
        plt.plot(k_values, f1_values, 'o-', label='F1_Att')
        plt.plot(k_values, auc_values, 'o-', label='AUC_Att')
        plt.plot(k_values_2, acc_values_2, 'o-', label='ACC_Normal')
        plt.plot(k_values_2, f1_values_2, 'o-', label='F1_Normal')
        plt.plot(k_values_2, auc_values_2, 'o-', label='AUC_Normal')

        plt.xlabel('epochs')
        plt.ylabel('value')
        plt.legend()
        plt.grid(True)
        plt.title('Performance Metrics Across Number of Epochs for k = 2 (Attention vs None)')

        plt.show()


    ################################################################################
    
    # Example usage of main_validaation_set, main_kfold_cross_validation, adj_parameter_plot and epoch_plot
    # Uncomment if want to use
    # main_validation_set(attention=True)
    # adj_parameter_plot()
    # epoch_plot()
    # main_kfold_cross_validation(plot = True, attention = True)