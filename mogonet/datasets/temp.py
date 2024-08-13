import json

metadata = {}
metadata["BRCA"] = {"details": "Breast Invasive Carcinoma", 
                    "features": ["1.csv", "2.csv", "3.csv"],
                    "feature_names": ["1_featname.csv", "2_featname.csv", "3_featname.csv"],
                    "num_class": 5,
                    "model_dir": "../datasets/BRCA",
                    "num_views": 3,
                    "labels": "labels.csv"
                    }

metadata["ROSMAP"] = {"details": "Breast Invasive Carcinoma", 
                    "features": ["1.csv", "2.csv", "3.csv"],
                    "feature_names": ["1_featname.csv", "2_featname.csv", "3_featname.csv"],
                    "num_class": 2,
                    "model_dir": "../datasets/ROSMAP",
                    "num_views": 3,
                    "labels": "labels.csv"
                    }

with open("metadata.json", "w") as f:
    json.dump(metadata, f)

