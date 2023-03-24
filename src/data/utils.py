import os
import sys
import yaml
import s3fs
from sklearn.model_selection import train_test_split
from pandas import read_sas, DataFrame
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())


def import_from_S3(
        endpoint: str,
        bucket: str,
        path: str,
        key_id: str,
        access_key: str,
        token: str) -> DataFrame:
    """
    enabling conexion to s3 storage for data retrieving
    """
    fs = s3fs.S3FileSystem(
            client_kwargs={'endpoint_url': endpoint},
            key=key_id,
            secret=access_key,
            token=token)

    return read_sas(
                fs.open(f"{bucket}/{path}/commsdata.sas7bdat"),
                format='sas7bdat')


def import_from_local(path) -> DataFrame:
    return read_sas(
                f"{path}/inputs/raw/commsdata.sas7bdat",
                encoding="utf-8").set_index("Customer_ID")


def check_train_frac(input: str) -> bool:
    if input.isdigit():
        return 0 < int(input) < 100
    return False


def save_input_data(key, value) -> None:
    with open(r'./data/app_inputs/sample_input.yaml') as file:
        data = yaml.load(file, Loader=yaml.Loader)
    if data is None:
        data = {key: value}
    else:
        data[key] = value
    with open(r'./data/app_inputs/sample_input.yaml', 'w') as file:
        yaml.dump(data, file)


def train_valid_splitting(df: DataFrame, frac: float):
    y = df["churn"]
    X = df.drop("churn", axis=1)
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                X,
                                                y,
                                                test_size=1-frac,
                                                stratify=y,
                                                random_state=42)
    return X_train, X_valid, y_train, y_valid
