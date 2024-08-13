from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from typing import *
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from endpoints import (
    upload_dataset_back,
    get_all_dataset_back,
    delete_dataset_back,
    get_dataset_data_back,
    train_model_back,
    get_probability_distribution_back,
    get_shap_values_back,
    get_attention_visualisation_back,
    )

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows CORS for the specified origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

####################################################################################
# Endpoint APIs

# Endpoint to upload dataset
# Throws error if feature_names, features and labels file are not csv
# Throws error if dimensions of features and feature names do not match
# Throws error if number of samples and labels do not match
# Returns model_id
@app.post("/upload-dataset/")
async def upload_dataset(
    model_id: str = Form(...),
    feature_names: List[UploadFile] = File(...),
    features: List[UploadFile] = File(...),
    labels: UploadFile = File(...),
    details: List[str] = Form([]),
):
    return upload_dataset_back(model_id, feature_names, features, labels, details)


# Endpoint to get all datasets
# Returns a list of all datasets uploaded to the server
@app.get("/get-all-dataset")
async def get_all_dataset():
    return get_all_dataset_back()


# Endpoint to delete a dataset
# Throws error if model_id is not found
# Returns deleted model_id
@app.delete("/dataset/{model_id}")
async def delete_dataset(model_id: str):
    return delete_dataset_back(model_id)


# Endpoint to get all data for a given model_id
# Throws error if model_id is not found
# Returns a JSON object of dataset information
@app.get("/dataset/data/{model_id}")
async def get_dataset_data(model_id: str):
    return get_dataset_data_back(model_id)


# Endpoint to train model using data in a given model_id
# Throws error if hyperparameter cannot be converted to integer
# Throws error if hyperparameter is negative or larger than the number of samples in the training set
# Returns model_id if successfully trained
@app.post("/train-model/{model_id}")
async def train_model(
    model_id: str,
    adj_parameter: int = Form(...),
):
    return train_model_back(model_id, adj_parameter)


# Endpoint to get probability distributions from a provided test sample (features_list)
# Throws error if model not found
# Throws error if the dimension of test sample and training samples does not match
# Returns a probability distribution as predicted by the MOGONET
@app.post("/probability-distribution/{model_id}")
async def get_probability_distribution(model_id: str, features_list: str = Form(...)):
    return get_probability_distribution_back(model_id, features_list)


# Endpoint to get shap values for single test observation
# Throws error if model not found
# Throws error if the dimension of test sample and training samples does not match
# Returns a waterfall plot of SHAP values for the given test observation
@app.post("/shap_values/{model_id}")
async def get_shap_values(model_id: str, features_list: str = Form(...)):
    return get_shap_values_back(model_id, features_list)


# Endpoint to show attention visualisations
# Throws error if model not found
# Returns attention graphs and heatmap graphs
@app.post("/attention-visualisation/{model_id}")
async def get_attention_visualisation(model_id: str):
    return get_attention_visualisation_back(model_id)


# Mount the static files directory
app.mount("/datasets", StaticFiles(directory="datasets"), name="datasets")
