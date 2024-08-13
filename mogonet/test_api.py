import pytest
import httpx
import os
import json
import warnings
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from main import app  # Adjust the import based on your file structure

client = TestClient(app)

##########################################################################
# Test upload_dataset

# Raise error if feature_names is not csv
def test_upload_dataset_not_csv_featname():
    files = {
        'feature_names': ('C.pth', open('ROSMAP/C.pth', 'rb'), 'text/csv'),
        'features': ('1_tr.csv', open('ROSMAP/1_tr.csv', 'rb'), 'text/csv'),
        'labels': ('labels_tr.csv', open('ROSMAP/labels_tr.csv', 'rb'), 'text/csv'),
    }
    response = client.post('/upload-dataset/', data={'model_id': '1', 'details': []}, files=files)
    assert response.status_code == 400
    assert response.json()['detail'] == 'Input file C.pth is not csv, please upload a csv file'

# Raise error if features is not csv
def test_upload_dataset_not_csv_feature():
    files = {
        'feature_names': ('1_featname.csv', open('ROSMAP/1_featname.csv', 'rb'), 'text/csv'),
        'features': ('C.pth', open('ROSMAP/C.pth', 'rb'), 'text/csv'),
        'labels': ('labels_tr.csv', open('ROSMAP/labels_tr.csv', 'rb'), 'text/csv'),
    }
    response = client.post('/upload-dataset/', data={'model_id': '1', 'details': []}, files=files)
    assert response.status_code == 400
    assert response.json()['detail'] == 'Input file C.pth is not csv, please upload a csv file'

# Raise error if label is not csv
def test_upload_dataset_not_csv_label():
    files = {
        'feature_names': ('1_featname.csv', open('ROSMAP/1_featname.csv', 'rb'), 'text/csv'),
        'features': ('1_tr.csv', open('ROSMAP/1_tr.csv', 'rb'), 'text/csv'),
        'labels': ('C.pth', open('ROSMAP/C.pth', 'rb'), 'text/csv'),
    }
    response = client.post('/upload-dataset/', data={'model_id': '1', 'details': []}, files=files)
    assert response.status_code == 400
    assert response.json()['detail'] == 'Input file C.pth is not csv, please upload a csv file'

# Raise error if dimensions of features and feature names do not match (first feature/featname)
def test_upload_dataset_dim_feat_featname_first():
    files = {
        'feature_names': ('1_featname.csv', open('BRCA/3_featname.csv', 'rb'), 'text/csv'),
        'features': ('1_tr.csv', open('ROSMAP/1_tr.csv', 'rb'), 'text/csv'),
        'labels': ('labels_tr.csv', open('ROSMAP/labels_tr.csv', 'rb'), 'text/csv'),
    }
    response = client.post('/upload-dataset/', data={'model_id': '1', 'details': []}, files=files)
    assert response.status_code == 400
    print(response.json()['detail'])
    assert "Feature name file '1_featname.csv' does not match the number of columns in the features file '1_tr.csv'" in response.json()['detail']

# Raise error if dimensions of features and feature names do not match (second feature/featname)
def test_upload_dataset_dim_feat_featname_second():
    files = {
        'feature_names': ('1_featname.csv', open('ROSMAP/1_featname.csv', 'rb'), 'text/csv'),
        'feature_names': ('2_featname.csv', open('BRCA/3_featname.csv', 'rb'), 'text/csv'),
        'features': ('1_tr.csv', open('ROSMAP/1_tr.csv', 'rb'), 'text/csv'),
        'features': ('2_tr.csv', open('ROSMAP/2_tr.csv', 'rb'), 'text/csv'),
        'labels': ('labels_tr.csv', open('ROSMAP/labels_tr.csv', 'rb'), 'text/csv'),
    }
    response = client.post('/upload-dataset/', data={'model_id': '1', 'details': []}, files=files)
    assert response.status_code == 400
    assert "Feature name file '2_featname.csv' does not match the number of columns in the features file '2_tr.csv'" in response.json()['detail']

# Raise error number of samples and labels do not match (third feature)
def test_upload_dataset_dim_sample_label_first():
    files = {
        'feature_names': ('1_featname.csv', open('ROSMAP/1_featname.csv', 'rb'), 'text/csv'),
        'features': ('1_tr.csv', open('ROSMAP/1_tr.csv', 'rb'), 'text/csv'),
        'labels': ('labels_tr.csv', open('BRCA/labels_tr.csv', 'rb'), 'text/csv'),
    }
    response = client.post('/upload-dataset/', data={'model_id': '1', 'details': []}, files=files)
    assert response.status_code == 400
    assert "Number of rows in features '1_tr.csv' and labels 'labels_tr.csv' do not match" in response.json()['detail']

# Raise error number of samples and labels do not match (third feature)
def test_upload_dataset_dim_sample_label_third():
    files = {
        'feature_names': ('1_featname.csv', open('ROSMAP/1_featname.csv', 'rb'), 'text/csv'),
        'feature_names': ('2_featname.csv', open('ROSMAP/2_featname.csv', 'rb'), 'text/csv'),
        'feature_names': ('3_featname.csv', open('BRCA/3_featname.csv', 'rb'), 'text/csv'),
        'features': ('1_tr.csv', open('ROSMAP/1_tr.csv', 'rb'), 'text/csv'),
        'features': ('2_tr.csv', open('ROSMAP/2_tr.csv', 'rb'), 'text/csv'),
        'features': ('3_tr.csv', open('BRCA/3_tr.csv', 'rb'), 'text/csv'),
        'labels': ('labels_tr.csv', open('ROSMAP/labels_tr.csv', 'rb'), 'text/csv'),
    }
    response = client.post('/upload-dataset/', data={'model_id': '1', 'details': []}, files=files)
    assert response.status_code == 400
    assert "Number of rows in features '3_tr.csv' and labels 'labels_tr.csv' do not match" in response.json()['detail']

# Succesful upload and delete one omic data
def test_upload_dataset_one_omic():
    files = {
        'feature_names': ('1_featname.csv', open('ROSMAP/1_featname.csv', 'rb'), 'text/csv'),
        'features': ('1_tr.csv', open('ROSMAP/1_tr.csv', 'rb'), 'text/csv'),
        'labels': ('labels_tr.csv', open('ROSMAP/labels_tr.csv', 'rb'), 'text/csv'),
    }
    response = client.post('/upload-dataset/', data={'model_id': '1', 'details': []}, files=files)
    assert response.status_code == 200
    assert response.json() == {'model_id': '1'}

    # Delete dataset
    response = client.delete('/dataset/1')
    assert response.json() == {'model_id': '1'}

# Succesful upload and delete three omics data
def test_upload_dataset_three_omics():
    files = {
        'feature_names': ('1_featname.csv', open('ROSMAP/1_featname.csv', 'rb'), 'text/csv'),
        'feature_names': ('2_featname.csv', open('ROSMAP/2_featname.csv', 'rb'), 'text/csv'),
        'feature_names': ('3_featname.csv', open('ROSMAP/3_featname.csv', 'rb'), 'text/csv'),
        'features': ('1_tr.csv', open('ROSMAP/1_tr.csv', 'rb'), 'text/csv'),
        'features': ('2_tr.csv', open('ROSMAP/2_tr.csv', 'rb'), 'text/csv'),
        'features': ('3_tr.csv', open('ROSMAP/3_tr.csv', 'rb'), 'text/csv'),
        'labels': ('labels_tr.csv', open('ROSMAP/labels_tr.csv', 'rb'), 'text/csv'),
    }
    response = client.post('/upload-dataset/', data={'model_id': '1', 'details': []}, files=files)
    assert response.json() == {'model_id': '1'}
    assert response.status_code == 200
    
    response = client.delete('/dataset/1')
    assert response.json() == {'model_id': '1'}


##########################################################################
# Test get_all_dataset and delete_dataset

# Test upload one dataset and get all dataset
def test_get_all_dataset_single_dataset():
    files = {
        'feature_names': ('1_featname.csv', open('ROSMAP/1_featname.csv', 'rb'), 'text/csv'),
        'features': ('1_tr.csv', open('ROSMAP/1_tr.csv', 'rb'), 'text/csv'),
        'labels': ('labels_tr.csv', open('ROSMAP/labels_tr.csv', 'rb'), 'text/csv'),
    }
    client.post('/upload-dataset/', data={'model_id': '1', 'details': []}, files=files)

    response = client.get('get-all-dataset')
    assert response.json() == {'datasets': ['1']}
    
    # Delete the dataset and get all dataset
    response = client.delete('/dataset/1')
    assert response.json() == {'model_id': '1'}
    response = client.get('get-all-dataset')
    assert response.json() == {'datasets': []}

# Test upload two datasets and get all dataset
def test_get_all_dataset_two_datasets():
    files = {
        'feature_names': ('1_featname.csv', open('ROSMAP/1_featname.csv', 'rb'), 'text/csv'),
        'features': ('1_tr.csv', open('ROSMAP/1_tr.csv', 'rb'), 'text/csv'),
        'labels': ('labels_tr.csv', open('ROSMAP/labels_tr.csv', 'rb'), 'text/csv'),
    }
    client.post('/upload-dataset/', data={'model_id': '1', 'details': []}, files=files)
    client.post('/upload-dataset/', data={'model_id': '2', 'details': []}, files=files)

    response = client.get('get-all-dataset')
    assert '1' in response.json()['datasets']
    assert '2' in response.json()['datasets']
    
    # Delete one dataset and get all dataset
    response = client.delete('/dataset/2')
    assert response.json() == {'model_id': '2'}
    response = client.get('get-all-dataset')
    assert response.json() == {'datasets': ['1']}
    
    # Delete the other dataset and get all dataset
    response = client.delete('/dataset/1')
    assert response.json() == {'model_id': '1'}
    response = client.get('get-all-dataset')
    assert response.json() == {'datasets': []}


#########################################################################
# Test get_dataset_data

# Test get_dataset_data
# functionalities of feature_names, features, labels are tested on frontend due to randomness in train_test_split
def test_get_dataset_data():
    files = {
        'feature_names': ('1_featname.csv', open('ROSMAP/1_featname.csv', 'rb'), 'text/csv'),
        'features': ('1_tr.csv', open('ROSMAP/1_tr.csv', 'rb'), 'text/csv'),
        'labels': ('labels_tr.csv', open('ROSMAP/labels_tr.csv', 'rb'), 'text/csv'),
    }
    client.post('/upload-dataset/', data={'model_id': '1', 'details': ['1']}, files=files)
    response = json.loads(client.get('/dataset/data/1').json())
    assert response['model_id'] == '1'
    assert response['details'] == ['1']
    client.delete('/dataset/1')

#########################################################################
# Test train_model errors

# Test hyperparameter 0 error
def test_train_model_hyperparameter_0():
    files = {
        'feature_names': ('1_featname.csv', open('ROSMAP/1_featname.csv', 'rb'), 'text/csv'),
        'features': ('1_tr.csv', open('ROSMAP/1_tr.csv', 'rb'), 'text/csv'),
        'labels': ('labels_tr.csv', open('ROSMAP/labels_tr.csv', 'rb'), 'text/csv'),
    }
    client.post('/upload-dataset/', data={'model_id': '1', 'details': ['1']}, files=files)
    response = client.post('/train-model/1', data={'adj_parameter': '0'})
    assert response.status_code == 400
    assert response.json()['detail'] == 'Hyperparameter must be positive and no larger than number of samples (245)'
    client.delete('/dataset/1')

# Test hyperparameter 246 (number of samples + 1) error
def test_train_model_hyperparameter_large():
    files = {
        'feature_names': ('1_featname.csv', open('ROSMAP/1_featname.csv', 'rb'), 'text/csv'),
        'features': ('1_tr.csv', open('ROSMAP/1_tr.csv', 'rb'), 'text/csv'),
        'labels': ('labels_tr.csv', open('ROSMAP/labels_tr.csv', 'rb'), 'text/csv'),
    }
    client.post('/upload-dataset/', data={'model_id': '1', 'details': ['1']}, files=files)
    response = client.post('/train-model/1', data={'adj_parameter': '246'})
    assert response.status_code == 400
    assert response.json()['detail'] == 'Hyperparameter must be positive and no larger than number of samples (245)'
    client.delete('/dataset/1')
