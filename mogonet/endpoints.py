from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from sklearn.model_selection import train_test_split
from typing import *
import os
import json
import shutil
import pandas as pd
from pathlib import Path
from train_test import *
from models import *
import warnings
import shap
import numpy as np

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message='Field "model_id" has conflict with protected namespace "model_".',
)

# Directories and files
DATA_DIR = "datasets"
METADATA_FILE = "datasets/metadata.json"
IMAGE_FOLDER = "datasets/images/"

####################################################################################
# Endpoint to upload dataset
# Throws error if feature_names, features and labels file are not csv
# Throws error if dimensions of features and feature names do not match
# Throws error if number of samples and labels do not match
# Returns model_id
def upload_dataset_back(
    model_id: str = Form(...),
    feature_names: List[UploadFile] = File(...),
    features: List[UploadFile] = File(...),
    labels: UploadFile = File(...),
    details: List[str] = Form([]),
):

    # Directory specific to the model_id
    model_dir = os.path.join(DATA_DIR, model_id)
    os.makedirs(model_dir, exist_ok=True)

    # Save and verify feature_names, features and labels files
    save_training_files(model_dir, feature_names, features, labels)

    # Verify dimensions of feature_names, features and labels files match
    feature_name_df_list, labels_df = verify_dataset_dimensions(model_dir, feature_names, features, labels)

    # Split the data into train and test sets
    split_train_test(model_dir, feature_name_df_list, features, labels, labels_df)

    # Get number of classes
    num_classes = len(labels_df["label"].unique())

    # Read and write metadata
    metadata = read_metadata()
    metadata[model_id] = {
        "model_id": model_id,
        "feature_names": [file.filename for file in feature_names],
        "features": [file.filename for file in features],
        "labels": labels.filename,
        "details": details,
        "num_classes": num_classes,
        "num_views": len(feature_names),
        "model_dir": model_dir,
    }

    write_metadata(metadata)
    return {"model_id": model_id}

# Endpoint to get all datasets
# Returns a list of all datasets uploaded to the server
def get_all_dataset_back():
    datasets = list(read_metadata().keys())
    response = {"datasets": datasets}
    return response


# Endpoint to delete a dataset
# Throws error if model_id is not found
# Returns deleted model_id
def delete_dataset_back(model_id: str):
    metadata = read_metadata()
    if model_id not in metadata:
        raise HTTPException(status_code=404, detail=f"Dataset {model_id} not found") # Internal error and won't appear on frontend

    model_dir = os.path.join(DATA_DIR, model_id)

    del metadata[model_id]
    delete_files_in_directory(model_dir)

    try:
        os.rmdir(model_dir)
    except OSError as e:
        raise HTTPException(status_code=500, detail="Unable to delete directory") # Internal error and won't appear on frontend

    write_metadata(metadata)
    return {"model_id": model_id}


# Endpoint to get all data for a given model_id
# Throws error if model_id is not found
# Returns a JSON object of dataset information
def get_dataset_data_back(model_id: str):
    metadata = read_metadata()
    if model_id not in metadata:
        raise HTTPException(status_code=404, detail=f"Dataset {model_id} not found") # Internal error and won't appear on frontend

    # Get dataset information
    model_info = metadata[model_id]
    feature_names, features, labels = get_dataset_model(model_info)

    response = {
        "model_id": model_id,
        "feature_names": feature_names,
        "features": features,
        "labels": labels,
        "details": metadata[model_id]["details"],
    }

    sanitized_response = sanitize_data(response)
    json_response = JSONResponse(
        content=json.dumps(sanitized_response, cls=CustomJSONEncoder)
    )
    return json_response


# Endpoint to train model using data in a given model_id
# Throws error if hyperparameter cannot be converted to integer
# Throws error if hyperparameter is negative or larger than the number of samples in the training set
# Returns model_id if successfully trained
def train_model_back(
    model_id: str,
    adj_parameter: int = Form(...),
):
    metadata = read_metadata()
    if model_id not in metadata:
        raise HTTPException(status_code=404, detail=f"Dataset {model_id} not found") # Internal error and won't appear on frontend

    model_info = metadata[model_id]

    # Hyperparameters given in original MOGONET study
    num_epoch_pretrain = 500
    num_epoch = 2500
    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-3
    num_class = model_info["num_classes"]
    view_list = range(1, len(model_info["feature_names"]) + 1)
    dim_he_list = [200, 200, 100]
    
    try:
        adj_parameter = int(adj_parameter)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Hyperparameter must be an integer")
    
    num_sample = len(pd.read_csv(os.path.join(model_info["model_dir"], model_info["labels"]), names=["label"]))
    if adj_parameter <= 0 or adj_parameter > num_sample:
        raise HTTPException(status_code=400, detail=f"Hyperparameter must be positive and no larger than number of samples ({num_sample})")

    # Call the model training function
    try:
        results_df, model_info = train_test(
            model_info["model_dir"],
            view_list,
            num_class,
            lr_e_pretrain,
            lr_e,
            lr_c,
            num_epoch_pretrain,
            num_epoch,
            features=model_info["features"],
            adj_parameter=adj_parameter,
            do_print=False,
            save_model=True,
            model_info=model_info,
            dim_he_list=dim_he_list,
            labels=model_info["labels"],
            attention=True,
            visualisation=False,
        )
        # Get last line of results_df
        results_df = results_df.iloc[[-1]]
        model_info["metrics"] = results_df.to_dict(orient="records")[0]
        model_info["adj_parameter"] = adj_parameter

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Model training failed: hyperparamter {adj_parameter} too large. Error message: {str(e)}")

    write_metadata(metadata)
    return {"model_id": model_id}


# Endpoint to get probability distributions from a provided test sample (features_list)
# Throws error if model not found
# Throws error if the dimension of test sample and training samples does not match
# Returns a probability distribution as predicted by the MOGONET
def get_probability_distribution_back(model_id: str, features_list: str = Form(...)):

    features_list = json.loads(features_list)
    metadata = read_metadata()
    if model_id not in metadata:
        raise HTTPException(status_code=404, detail="Dataset not found") # Internal error and won't appear on frontend
    metadata_info = metadata[model_id]
    response = verify_feature_list(model_id, features_list)
    if (response['status_code'] == 400):
        raise HTTPException(status_code=400, detail=response['detail']) # Internal error and won't appear on frontend

    try:
        model_dict, _, data_tr_list, data_trte_list, trte_idx, _ = prepare_training(metadata_info, features_list)
        _, adj_te_list = gen_trte_adj_mat(
            data_tr_list, data_trte_list, trte_idx, adj_parameter=metadata_info["adj_parameter"]
        )
        prob_dist = test_epoch(
            data_trte_list, adj_te_list, trte_idx["te"], model_dict
        ).tolist()

    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=500,
            detail=f"Error in getting probability distributions: {str(e)}", # Internal error and won't appear on frontend
        )

    response = {
        "model_id": model_id,
        "distributions": prob_dist,
    }
    return response


# Endpoint to get shap values for single test observation
# Throws error if model not found
# Throws error if the dimension of test sample and training samples does not match
# Returns a waterfall plot of SHAP values for the given test observation
def get_shap_values_back(model_id: str, features_list: str = Form(...)):

    features_list = json.loads(features_list)
    metadata = read_metadata()
    if model_id not in metadata:
        raise HTTPException(status_code=404, detail="Dataset not found") # Internal error and won't appear on frontend
    metadata_info = metadata[model_id]
    response = verify_feature_list(model_id, features_list)
    if (response['status_code'] == 400):
        raise HTTPException(status_code=400, detail=response['detail']) # Internal error and won't appear on frontend

    try:
        model_dict, omics_size, _, data_trte_list, _, _ = prepare_training(metadata_info, features_list)

        # Form background training data for shap
        X_combined = np.concatenate(
            [data_trte_list[i].cpu().numpy() for i in range(len(data_trte_list))],
            axis=1,
        )

        # test_shap function to pass in kernel shap
        def test_shap(test_data):
            data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_tr_data(
                metadata_info["model_dir"],
                test_data,
                range(1, len(metadata_info["feature_names"]) + 1),
                omics_size,
                metadata_info["features"],
                metadata_info["labels"],
            )
            adj_tr_list, adj_te_list = gen_trte_adj_mat(
                data_tr_list, data_trte_list, trte_idx, adj_parameter=metadata_info["adj_parameter"]
            )
            return test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)

        explainer = shap.KernelExplainer(test_shap, X_combined[0:5])
        shap_values = explainer(features_list)
        shap_values.feature_names = get_feature_name(model_id)
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=500,
            detail=f"Error in getting shap values: {str(e)}", # Internal error and won't appear on frontend
        )

    # Save shap waterfall plot
    plt.figure(figsize=(20, 5))  # Width=20, Height=5
    shap.plots.waterfall(shap_values[0, :, 0], max_display=10, show=False)
    plt.savefig(
        os.path.join(metadata_info["model_dir"], "shap.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    response = {
        "model_id": model_id,
        "shap_values": FileResponse(
            os.path.join(metadata_info["model_dir"], "shap.png")
        ),
    }
    return response


# Endpoint to show attention visualisations
# Throws error if model not found
# Returns attention graphs and heatmap graphs
def get_attention_visualisation_back(model_id: str):
    metadata = read_metadata()
    if model_id not in metadata:
        raise HTTPException(status_code=404, detail=f"Dataset {model_id} not found") # Internal error and won't appear on frontend

    model_dir = metadata[model_id]["model_dir"]

    graph_path = os.path.join(model_dir, "graph.png")
    heatmap_path = os.path.join(model_dir, "heatmap.png")
    if not os.path.exists(graph_path) or not os.path.exists(heatmap_path):
        raise HTTPException(status_code=500, detail="Image not found") # Internal error and won't appear on frontend

    response = {
        "model_id": model_id,
        "graph": FileResponse(graph_path),
        "heatmap": FileResponse(heatmap_path),
    }
    return response


####################################################################################
# Utility functions and classes for backends

# Utility function to read metadata
def read_metadata():
    if not Path(METADATA_FILE).exists():
        return {}
    with open(METADATA_FILE, "r") as f:
        return json.load(f)


# Utility function to write metadata
def write_metadata(metadata):
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)


# Utility function to check if a given file is csv
def is_csv(file_path):
    try:
        pd.read_csv(file_path)
        return True
    except Exception as e:
        return False


# Utility function to save feature_names, features and labels files
def save_training_files(model_dir, feature_names, features, labels):
    # Save and verify feature_names files
    for file in feature_names:
        file_location = os.path.join(model_dir, file.filename)
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)
        if not is_csv(file_location):
            raise HTTPException(status_code=400,
                    detail=f"Input file {file.filename} is not csv, please upload a csv file")

    # Save and verify features files
    for file in features:
        file_location = os.path.join(model_dir, file.filename)
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)
        if not is_csv(file_location):
            raise HTTPException(status_code=400,
                    detail=f"Input file {file.filename} is not csv, please upload a csv file")

    # Save and verify labels file
    labels_location = os.path.join(model_dir, labels.filename)
    with open(labels_location, "wb") as f:
        shutil.copyfileobj(labels.file, f)
    if not is_csv(labels_location):
        raise HTTPException(status_code=400,
                    detail=f"Input file {labels.filename} is not csv, please upload a csv file")


# Utility function to verify the number of features in feature_names and features files,
# as well as the number of samples in features and labels files
def verify_dataset_dimensions(model_dir, feature_names, features, labels):
    # Verify that the number of columns in feature_names matches the number of rows in features
    try:
        feature_name_df_list = []
        for feature_name, feature in zip(feature_names, features):
            feature_name_df = pd.read_csv(os.path.join(model_dir, feature_name.filename), names=["feature_name"])
            feature_df = pd.read_csv(os.path.join(model_dir, feature.filename), header=None)
            if len(feature_name_df) != len(feature_df.columns):
                raise HTTPException(
                    status_code=400,
                    detail=f"Feature name file '{feature_name.filename}' does not match the number of columns in the features file '{feature.filename}'",
                )
            feature_name_df_list.append(feature_name_df)

        # Verify the same number of rows in features and labels
        for feature_name_df, feature in zip(feature_name_df_list, features):
            feature_df = pd.read_csv(
                os.path.join(model_dir, feature.filename),
                names=feature_name_df["feature_name"].tolist(),
            )
            labels_df = pd.read_csv(
                os.path.join(model_dir, labels.filename), names=["label"]
            )
            if len(feature_df) != len(labels_df):
                raise HTTPException(
                    status_code=400,
                    detail=f"Number of rows in features '{feature.filename}' and labels '{labels.filename}' do not match",
                )
    except Exception as e:
        delete_files_in_directory(model_dir)
        raise HTTPException(
            status_code=400, detail=f"Error in verifying files: {str(e)}" # Internal error and won't appear on frontend
        )
    return feature_name_df_list, labels_df


# Utility functions to split train test datasets and write into CSVs
def split_train_test(model_dir, feature_name_df_list, features, labels, labels_df):
    for feature_name_df, feature in zip(feature_name_df_list, features):
        feature_df = pd.read_csv(
            os.path.join(model_dir, feature.filename),
            names=feature_name_df["feature_name"].tolist(),
        )
        X_train, X_test, y_train, y_test = train_test_split(
            feature_df, labels_df, test_size=0.4, random_state=42
        )
        base_name, ext = os.path.splitext(feature.filename)
        train_feature_path = os.path.join(model_dir, f"{base_name}_tr{ext}")
        test_feature_path = os.path.join(model_dir, f"{base_name}_te{ext}")
        X_train.to_csv(train_feature_path, index=False, header=False)
        X_test.to_csv(test_feature_path, index=False, header=False)

    base_labels_name, labels_ext = os.path.splitext(labels.filename)
    y_train_path = os.path.join(model_dir, f"{base_labels_name}_tr{labels_ext}")
    y_test_path = os.path.join(model_dir, f"{base_labels_name}_te{labels_ext}")
    y_train.to_csv(y_train_path, index=False, header=False)
    y_test.to_csv(y_test_path, index=False, header=False)


# Utility function to get feature_names, features, labels from dataset
def get_dataset_model(model_info):
    model_dir = model_info["model_dir"]
    features = []
    feature_names = []
    for feature_name_file, feature_file in zip(
        model_info["feature_names"], model_info["features"]
    ):
        feature_name_df = pd.read_csv(
            os.path.join(model_dir, feature_name_file), names=["feature_name"]
        )["feature_name"].tolist()

        # Combine test and train data
        feature_te_df = pd.read_csv(
            os.path.join(model_dir, feature_file.replace(".csv", "_te.csv")),
            names=feature_name_df,
        )
        feature_tr_df = pd.read_csv(
            os.path.join(model_dir, feature_file.replace(".csv", "_tr.csv")),
            names=feature_name_df,
        )
        feature_df = pd.concat(
            [feature_tr_df, feature_te_df], ignore_index=True
        ).to_dict(orient="records")
        features.append(feature_df)
        feature_names.append(feature_name_df)

    labels_te = pd.read_csv(
        os.path.join(model_dir, model_info["labels"].replace(".csv", "_te.csv"))
    )
    labels_tr = pd.read_csv(
        os.path.join(model_dir, model_info["labels"].replace(".csv", "_tr.csv"))
    )
    labels = pd.concat(
        [pd.DataFrame(labels_tr), pd.DataFrame(labels_te)], ignore_index=True
    ).to_dict(orient="records")
    return feature_names, features, labels


# Utility function to sanitize data for get_dataset_data
def sanitize_data(data):
    if isinstance(data, float) and (np.isnan(data) or np.isinf(data)):
        return None
    elif isinstance(data, list):
        return [sanitize_data(item) for item in data]
    elif isinstance(data, dict):
        return {key: sanitize_data(value) for key, value in data.items()}
    return data


# Utility function to delete files in a directory
def delete_files_in_directory(directory: str):
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


# Utility function to get feature names given a valid model_id
def get_feature_name(model_id: str):
    metadata = read_metadata()
    model_dir = metadata[model_id]["model_dir"]
    feature_names = []
    for feature_name_file in metadata[model_id]["feature_names"]:
        feature_name_df = pd.read_csv(
            os.path.join(model_dir, feature_name_file), names=["feature_name"]
        )
        feature_names = feature_names + feature_name_df.iloc[:, 0].tolist()
    return feature_names


# Utility function to verify feature list according to data in dataset folder
def verify_feature_list(model_id, features_list):
    metadata = read_metadata()
    metadata_info = metadata[model_id]

    # Verify features list
    if len(features_list) != len(metadata_info["feature_names"]):
        return {
            'status_code': 400,
            'detail': f"Number of input omics data ({len(features_list)}) does not match the number of omics data ({len(metadata_info['feature_names'])}) in the dataset",
         } # Internal error and won't appear on frontend

    for features, feature_name_files in zip(
        features_list, metadata_info["feature_names"]
    ):
        # Get amount of features names in feature_name_files
        feature_names = pd.read_csv(
            os.path.join(metadata_info["model_dir"], feature_name_files), header=None
        )
        if len(features) != len(feature_names):
            return {
                'status_code': 400,
                'detail': f"Number of input features ({len(features)}) in does not match the number of features ({len(feature_names)}) in the {feature_name_files} file",
            } # Internal error and won't appear on frontend
        return {'status_code': 200}


# Utility function to get the size of each omics dataset
def get_omics_size(metadata_info):
    omics_size = []
    for i in range(len(metadata_info["feature_names"])):
        omics_size.append(
            pd.read_csv(
                os.path.join(
                    metadata_info["model_dir"],
                    metadata_info["feature_names"][i],
                ),
                header=None,
            ).shape[0]
        )
    return omics_size


# Utility function to load model and get training data
def prepare_training(metadata_info, features_list):
    omics_size = get_omics_size(metadata_info)
    features_list = [feature for sublist in features_list for feature in sublist]
    features_list = np.array(features_list).reshape(1, -1)
    model_dict = load_models(
        metadata_info["model_dir"],
        metadata_info["num_views"],
        metadata_info["num_classes"],
        metadata_info["dim_list"],
        metadata_info["dim_he_list"],
        metadata_info["dim_hvcdn"],
    )
    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_tr_data(
        metadata_info["model_dir"],
        features_list,
        range(1, len(metadata_info["feature_names"]) + 1),
        omics_size,
        metadata_info["features"],
        metadata_info["labels"],
    )
    return model_dict, omics_size, data_tr_list, data_trte_list, trte_idx, labels_trte


# Custom JSON Encoder for passing JSON data to frontend
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
        return super().default(obj)
