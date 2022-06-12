"""Main script of the project."""
import yaml
from src.data.import_data import import_from_S3
from src.data.train_test_split import make_val_split
from src.features.build_features import useless_feature, label_encode_variable
from src.features.build_features import imputation, lower_limit, log_transform
from src.features.build_features import add_missing_indicator, decode_char
from src.models.train_evaluate import evaluate_rdmf


if __name__ == "__main__":
    with open("/home/coder/.config/code-server/telco_churn/config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    # data
    data = import_from_S3(config["input"]["key_id"], config["input"]["access_key"], config["input"]["token"])

    ## metadata variable management pipelines
    data = useless_feature(data)
    data = decode_char(data)
    data = lower_limit(data)
    data = log_transform(data)
    data = label_encode_variable(data)
    data = add_missing_indicator(data)

    ## splitting 
    X_train, X_test, y_train, y_test = make_val_split(data)

    # imputation
    X_train = imputation(X_train)

    X_test = imputation(X_test)

    # models pipelines
    evaluate_rdmf(X_train, X_test, y_train, y_test, n_estimators=20)

    # model comparaison
    
