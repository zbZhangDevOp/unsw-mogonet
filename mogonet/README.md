
# MOGONET: Multi-omics Integration via Graph Convolutional Networks for Biomedical Data Classification

MOGONET (Multi-Omics Graph cOnvolutional NETworks) is a novel multi-omics data integrative analysis framework for classification tasks in biomedical applications.

![MOGONET](https://github.com/txWang/MOGONET/blob/master/MOGONET.png?raw=true 'MOGONET')
Overview of MOGONET. <sup>Illustration of MOGONET. MOGONET combines GCN for multi-omics specific learning and VCDN for multi-omics integration. MOGONET combines GCN for multi-omics specific learning and VCDN for multi-omics integration. For clear and concise illustration, an example of one sample is chosen to demonstrate the VCDN component for multi-omics integration. Pre-processing is first performed on each omics data type to remove noise and redundant features. Each omics-specific GCN is trained to perform class prediction using omics features and the corresponding sample similarity network generated from the omics data. The cross-omics discovery tensor is calculated from the initial predictions of omics-specific GCNs and forwarded to VCDN for final prediction. MOGONET is an end-to-end model and all networks are trained jointly.<sup>

## Files

- _main_mogonet.py_: Examples of MOGONET for classification tasks
- _main_biomarker.py_: Examples for identifying biomarkers
- _models.py_: MOGONET model
- _train_test.py_: Training and testing functions
- _feat_importance.py_: Feature importance functions
- _utils.py_: Supporting functions
- _endpoints.py_: FastAPI backend endpoints for MOGONET
- _shap_values.py_: SHAP values computation for MOGONET



## Detailed Explanation

### Overview

The provided codebase consists of scripts and functions designed for training models to identify biomarkers and work with multi-omics data using machine learning and deep learning approaches.

### Key Components

1. **Feature Importance Calculation**:

   - `feat_importance.py`: Contains functions to calculate and normalize feature importance from a trained model.

2. **Biomarker Model Training**:

   - `main_biomarker.py`: Loads data, trains a RandomForestClassifier model, and calculates feature importance.

3. **Multi-Omics Data Integration**:

   - `main_mogonet.py`: Loads multi-omics data, trains a MOGONET model, and evaluates the model.

4. **Model Definitions**:

   - `models.py`: Defines the MOGONET model, a neural network that integrates two omics datasets for training and prediction.

5. **Training and Testing**:

   - `train_test.py`: Provides utility functions to split data into training and testing sets, perfor training and testing on the given dataset.

6. **Utility Functions**:
   - `utils.py`: Contains functions for loading multi-omics data and evaluating the model.

7. **Backend Endpoints**:
   - `endpoints.py`: Contains FastAPI backend endpoints to expose the functionality of MOGONET as web services.
   
8. **SHAP values**:
   - `shap_values.py`: Contains implementation for SHAP values functionality and creates waterfall, violin and beeswarm plots.

### Usage

1. **Training a Biomarker Model**:

   - Run `main_biomarker.py` to train a RandomForest model on the data.
   - This script loads the data, splits it into training and testing sets, trains the model, calculates feature importance, and prints normalized feature importance values.

2. **Training a MOGONEt Model**:

   - Run `main_mogonet.py` to train the MOGONET model on multi-omics data.
   - This script loads multi-omics data, splits it into training and testing sets, trains the MOGONET model, and evaluates the model's performance.

3. **Calculating Feature Importance**:
   - The `calculate_feature_importance` function in `feat_importance.py` extracts feature importance from a trained model.
   - The `normalize_importance` function normalizes the importance values to make them comparable.

4. **Using Backend Endpoints**:
   - The `endpoints.py` file provides FastAPI endpoints to interact with the MOGONET model.
   - To run the FastAPI server, use the following command:
     ```bash
     uvicorn main:app --reload
     ```

### API Endpoints

#### Upload Dataset

- **Endpoint**: `/upload-dataset/`
- **Method**: POST
- **Description**: Uploads a dataset. Throws an error if feature names, features, and labels files are not CSV, or if dimensions do not match.
- **Parameters**:
  - `model_id`: Model identifier.
  - `feature_names`: List of feature names files.
  - `features`: List of feature files.
  - `labels`: Labels file.
  - `details`: List of additional details.
- **Returns**: `model_id`.

#### Get All Datasets

- **Endpoint**: `/get-all-dataset`
- **Method**: GET
- **Description**: Returns a list of all datasets uploaded to the server.

#### Delete Dataset

- **Endpoint**: `/dataset/{model_id}`
- **Method**: DELETE
- **Description**: Deletes a dataset. Throws an error if the model_id is not found.
- **Parameters**: 
  - `model_id`: Model identifier.
- **Returns**: Deleted `model_id`.

#### Get Dataset Data

- **Endpoint**: `/dataset/data/{model_id}`
- **Method**: GET
- **Description**: Returns all data for a given model_id. Throws an error if the model_id is not found.
- **Parameters**:
  - `model_id`: Model identifier.
- **Returns**: JSON object of dataset information.

#### Train Model

- **Endpoint**: `/train-model/{model_id}`
- **Method**: POST
- **Description**: Trains the model using data for a given model_id. Throws an error if the hyperparameter cannot be converted to integer or if it is negative or larger than the number of samples in the training set.
- **Parameters**:
  - `model_id`: Model identifier.
  - `adj_parameter`: Adjustment parameter for the model.
- **Returns**: `model_id` if successfully trained.

#### Get Probability Distribution

- **Endpoint**: `/probability-distribution/{model_id}`
- **Method**: POST
- **Description**: Gets probability distributions from a provided test sample (features_list). Throws an error if the model is not found or if the dimensions of the test sample and training samples do not match.
- **Parameters**:
  - `model_id`: Model identifier.
  - `features_list`: List of features for the test sample.
- **Returns**: Probability distribution predicted by MOGONET.

#### Get SHAP Values

- **Endpoint**: `/shap_values/{model_id}`
- **Method**: POST
- **Description**: Gets SHAP values for a single test observation. Throws an error if the model is not found or if the dimensions of the test sample and training samples do not match.
- **Parameters**:
  - `model_id`: Model identifier.
  - `features_list`: List of features for the test sample.
- **Returns**: Waterfall plot of SHAP values for the given test observation.

#### Get Attention Visualisation

- **Endpoint**: `/attention-visualisation/{model_id}`
- **Method**: POST
- **Description**: Shows attention visualizations. Throws an error if the model is not found.
- **Parameters**:
  - `model_id`: Model identifier.
- **Returns**: Attention graphs and heatmap graphs.

### Detailed Steps for Training

1. **Data Preparation**:

   - Replace placeholder data loading functions (`load_data`, `load_multiomics_data`) with actual data loading logic to use your specific datasets.

2. **Model Training**:

   - For the biomarker model, the `train_biomarker_model` function in `main_biomarker.py` trains a RandomForestClassifier on the provided data.
   - For the MOGONet model, the `fit` method in the `MOGONet` class trains the model on multi-omics data.

3. **Model Evaluation**:
   - The `evaluate_model` function in `utils.py` evaluates the trained model's performance.

### Dependencies

- Install required packages using:
  ```bash
  pip install -r requirements.txt
  ```

### Contributors

