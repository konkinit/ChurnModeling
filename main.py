"""Main script of the project."""
import yaml
import pandas as pd

from src.data.import_data import import_
from src.data.train_test_split import make_val_split, train_test_data
from src.features.build_features import useless_feature#, label_encode_variable
from src.models.train_evaluate import evaluate_rdmf


if __name__ == "__main__":
    with open("/home/coder/.config/code-server/telco_churn/config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    
    data = import_(config["input"]["key_id"], config["input"]["access_key"], config["input"]["token"])

    X_train, X_test, y_train, y_test = make_val_split(data)

    train_data, test_data = train_test_data(X_train, X_test, y_train, y_test)

    # evaluate_rdmf(X_train, X_test, y_train, y_test, n_estimators=20)