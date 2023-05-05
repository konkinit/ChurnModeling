import os
import sys
import yaml
import s3fs
from pandas import (
    read_sas,
    DataFrame,
    SparseDtype,
    Series
)
from numpy import (
    linspace,
    ndarray,
    array,
    log
)
from dataclasses import dataclass
from typing import Tuple, Any, Union
from sklearn.model_selection import train_test_split
from scipy.sparse import csc_matrix, _csc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.configs import (
    models_inputs
)


_cutoffs = linspace(0.1, 0.9, 17)


_key_rename = {
    "0.0": "active",
    "1.0": "churned",
    "macro avg": "macro avg",
    "weighted avg": "weighted avg"
    }


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


def train_valid_splitting(
        df: DataFrame, frac: float):
    y = df["churn"]
    X = df.drop("churn", axis=1)
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                X,
                                                y,
                                                test_size=1-frac,
                                                stratify=y,
                                                random_state=42)
    return X_train, X_valid, y_train.values, y_valid.values


def dataframe2sparse(df: DataFrame) -> Tuple[_csc.csc_matrix, list]:
    return (
        csc_matrix(
            df.astype(SparseDtype("int32", 0)).sparse.to_coo()),
        df.columns.tolist())


@dataclass
class Modeling_Data:
    X_train_sparse: _csc.csc_matrix
    X_test_sparse: _csc.csc_matrix
    y_train: ndarray
    y_test: ndarray
    target_name: str = "churn"


Modeling_Data.__module__ = __name__


def quantiles_list(x: Series) -> list:
    list_quantile = [x.min(),
                     x.quantile(0.25),
                     x.quantile(0.5),
                     x.quantile(0.75),
                     x.max()]
    return list_quantile


def indicator_ab(x: float, a: float, b: float):
    if a-0.001 < x <= b:
        return 1
    return 0


def df_skewed_feature(df: DataFrame, feature: str) -> DataFrame:
    df_plot = df[[feature]].copy()
    df_plot[f"log_{feature}"] = df_plot[feature].apply(lambda x: log(1+x))
    return df_plot


def _sample_weight(target_array: ndarray) -> ndarray:
    return compute_sample_weight(
                class_weight='balanced',
                y=target_array
            )


def cm_annotation(prefix: str) -> ndarray:
    """Return a list of labels for a given confusion
       matrix axis

    Args:
        prefix (str): the label of the axis: either real
        or pred for respectively real outcome and
        prediction one

    Returns:
        ndarray: list of the labels with the prefix concatenate
        to the target variable vvalues labels
    """
    return array([f"{prefix}_active", f"{prefix}_churned"])


def performance_measure(
        model: Union[RandomForestClassifier, XGBClassifier],
        X: csc_matrix,
        y: ndarray,
        cutoff: float) -> Tuple[Any]:
    proba = model.predict_proba(X)[:, 1]
    yHat = 1.*(proba > cutoff)
    _report = classification_report(
                    y,
                    yHat,
                    # sample_weight=_sample_weight(y),
                    digits=3,
                    output_dict=True
                )
    _report_keys = list(_report.keys())
    _report_keys.remove('accuracy')
    _report = {_key_rename[key]: _report[key] for key in _report_keys}
    _acc = accuracy_score(
                y,
                yHat,
                # sample_weight=_sample_weight(y)
            )
    _cf_matrix = confusion_matrix(
                    y,
                    yHat,
                    # sample_weight=_sample_weight(y)
                )
    return _report, _acc, DataFrame(
                            data=_cf_matrix,
                            columns=cm_annotation("pred"),
                            index=cm_annotation("real")
                            )


def model_report(
        model: Union[RandomForestClassifier, XGBClassifier],
        params: models_inputs,
        cutoff: float) -> Tuple[Any]:
    (
        _report_train,
        _acc_train,
        _cm_train
    ) = performance_measure(
                    model,
                    params.X_train,
                    params.y_train,
                    cutoff
                )
    _report_test, _acc_test, _cm_test = performance_measure(
                                                model,
                                                params.X_test,
                                                params.y_test,
                                                cutoff
                                            )
    return (
        _report_train,
        _acc_train,
        _cm_train,
        _report_test,
        _acc_test,
        _cm_test
    )
