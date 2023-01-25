"""Main script of the project."""
import yaml
import os
from src.data.import_data import import_from_local
from src.data.train_test_split import make_val_split
from src.features.build_features import DataProcessing
from src.models.train_evaluate import evaluate_rdmf

os.chdir("/home/ikonkobo/Desktop/Self_Learning/telco_churn/")

if __name__ == "__main__":
    with open("config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    # data
    data = import_from_local(".")

    ## metadata variable management pipelines
    (DataProcessing(data)
        .useless_feature()
        .decode_char()
        .lower_limit()
        .log_transform()
        .label_encode_variable()
        .add_missing_indicator()
    )

    ## splitting
    X_train, X_test, y_train, y_test = make_val_split(data)

    # imputation
    DataProcessing(X_train).imputation()
    DataProcessing(X_test).imputation()

    # models pipelines
    evaluate_rdmf(X_train, X_test, y_train, y_test, n_estimators=20)

    # model comparaison