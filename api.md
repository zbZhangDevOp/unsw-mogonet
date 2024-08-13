### API Endpoints Summary

| Method | Endpoint                       | Parameters                                      | Returns                                        | Description                                  |
|--------|--------------------------------|-------------------------------------------------|------------------------------------------------|---------------------------------------------|
| POST   | `/upload-dataset/`           | `model_id: str`, `files: List[UploadFile]`, `labels: List[str]`, `details: Dict[str, Any]` | `{"model_id": "unique_id"}`                   | Upload a dataset with variable CSVs, labels, and additional details for a specified model. |
| PUT    | `/update-dataset/{model_id}` | `model_id: str`, `files: List[UploadFile]`, `labels: List[str]`, `details: Dict[str, Any]` | `{"model_id": "unique_id"}`                   | Update an existing dataset with new CSVs, labels, and details. |
| DELETE | `/dataset/{model_id}`        | `model_id: str`                               | `{"model_id": "unique_id"}`                   | Remove a specified dataset from the server. |
| GET    | `/dataset/{model_id}`        | `model_id: str`                               | `{"model_id": "unique_id", "details": {...}}` | Get details about a specific dataset. |
| GET    | `/dataset/data/{model_id}`   | `model_id: str`                               | `{"model_id": "unique_id", "data": {...}}`    | Retrieve the dataset itself for the specified model. |
| GET    | `/valid-datasets/`           |                                                | `["dataset1", "dataset2", ...]`               | Retrieve a list of valid datasets. |
| POST   | `/train-model/{model_id}`    | `model_id: str`, `hyperparams: HyperparamsModel` | `{"model_id": "unique_id"}`                   | Train a model with provided hyperparameters. |
| POST   | `/search-model-hyperparams/{model_id}` | `model_id: str`                | `{"model_id": "unique_id"}`                   | Train a model by finding the best hyperparameters. |
| GET    | `/model/status/{model_id}`   | `model_id: str`                               | `{"model_id": "unique_id", "status": "training/completed", "metrics": {...}}` | Check the training status of a model. |
| POST   | `/model/evaluate/{model_id}` | `model_id: str`, `inputs: Dict`, `evaluate: bool` | `{"model_id": "unique_id", "predictions": {...}, "metrics": {...}}` | Get predictions or evaluate the model based on inputs. |
| GET    | `/model/evaluate/{model_id}` | `model_id: str`                               | `{"model_id": "unique_id", "metrics": {...}}` | Retrieve evaluation metrics for a trained model. |
| GET    | `/features/{model_id}`       | `model_id: str`, `label: str`                | `{"model_id": "unique_id", "features": ["feature1", "feature2", ...]}` | Retrieve all features from the dataset for the specified label. |
| POST   | `/retrain-model/{model_id}`  | `model_id: str`, `hyperparams: HyperparamsModel` | `{"model_id": "unique_id"}`                   | Retrain an existing model with new hyperparameters. |
| POST   | `/probability-distribution/{model_id}` | `model_id: str`, `features_list: List[Dict[str, Any]]` | `{"model_id": "unique_id", "distributions": [{...}, {...}]}` | Get probability distributions for multiple input feature sets. |
