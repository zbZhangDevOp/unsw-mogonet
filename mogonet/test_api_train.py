import os
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient
from main import app
from endpoints import *  # Adjust the import based on your file structure

client = TestClient(app)

####################################################################################

# train_model function from main with async removed
def train_model(
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
    return {"model_id": model_id, "metrics": model_info["metrics"]}


####################################################################################
# Test train_model

# Test hyperparameter 0 error
def test_train_model_one_omic():
    files = {
        'feature_names': ('1_featname.csv', open('ROSMAP/1_featname.csv', 'rb'), 'text/csv'),
        'features': ('1_tr.csv', open('ROSMAP/1_tr.csv', 'rb'), 'text/csv'),
        'labels': ('labels_tr.csv', open('ROSMAP/labels_tr.csv', 'rb'), 'text/csv'),
    }
    client.post('/upload-dataset/', data={'model_id': '1', 'details': ['1']}, files=files)
    response = train_model('1', '2')
    assert response['model_id'] == '1'
    client.delete('/dataset/1')

# Test train model with one omic data
def test_train_model_one_omic():
    files = {
        'feature_names': ('1_featname.csv', open('ROSMAP/1_featname.csv', 'rb'), 'text/csv'),
        'features': ('1_tr.csv', open('ROSMAP/1_tr.csv', 'rb'), 'text/csv'),
        'labels': ('labels_tr.csv', open('ROSMAP/labels_tr.csv', 'rb'), 'text/csv'),
    }
    client.post('/upload-dataset/', data={'model_id': '1', 'details': ['1']}, files=files)
    response = train_model('1', '2')
    assert response['model_id'] == '1'
    client.delete('/dataset/1')

# Test train model with three omics data
def test_train_model_three_omics():
    files = {
        'feature_names': ('1_featname.csv', open('ROSMAP/1_featname.csv', 'rb'), 'text/csv'),
        'feature_names': ('2_featname.csv', open('ROSMAP/2_featname.csv', 'rb'), 'text/csv'),
        'feature_names': ('3_featname.csv', open('ROSMAP/3_featname.csv', 'rb'), 'text/csv'),
        'features': ('1_tr.csv', open('ROSMAP/1_tr.csv', 'rb'), 'text/csv'),
        'labels': ('labels_tr.csv', open('ROSMAP/labels_tr.csv', 'rb'), 'text/csv'),
    }
    client.post('/upload-dataset/', data={'model_id': '1', 'details': []}, files=files)
    response = train_model('1', '2')
    assert response['model_id'] == '1'
    client.delete('/dataset/1')


####################################################################################
# Test probability_distribution

def test_probability_distribution():
    files = {
        'feature_names': ('1_featname.csv', open('ROSMAP/1_featname.csv', 'rb'), 'text/csv'),
        'features': ('1_tr.csv', open('ROSMAP/1_tr.csv', 'rb'), 'text/csv'),
        'labels': ('labels_tr.csv', open('ROSMAP/labels_tr.csv', 'rb'), 'text/csv'),
    }
    client.post('/upload-dataset/', data={'model_id': '1', 'details': ['1']}, files=files)
    train_model('1', '2')
    response = client.post('/probability-distribution/1', data={'features_list': [[[0] * 200]]}) # 200 '0's
    assert response.json()['model_id'] == '1'
    assert abs(sum(response.json()['distributions'][0]) - 1.0) <= 0.0001
    client.delete('/dataset/1')

####################################################################################
# Test attention graphs

def test_attention_grahps():
    files = {
        'feature_names': ('1_featname.csv', open('ROSMAP/1_featname.csv', 'rb'), 'text/csv'),
        'features': ('1_tr.csv', open('ROSMAP/1_tr.csv', 'rb'), 'text/csv'),
        'labels': ('labels_tr.csv', open('ROSMAP/labels_tr.csv', 'rb'), 'text/csv'),
    }
    client.post('/upload-dataset/', data={'model_id': '1', 'details': ['1']}, files=files)
    train_model('1', '2')
    assert os.path.isfile('datasets/1/graph.png')
    assert os.path.isfile('datasets/1/heatmap.png')
    client.delete('/dataset/1')

####################################################################################
# Testing

test_train_model_one_omic()
test_train_model_three_omics()
test_probability_distribution()
test_attention_grahps()
print("All tests passed!")