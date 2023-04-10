from dataclasses import dataclass
from numpy import (
    linspace,
    ndarray,
    array
)
from pandas import DataFrame
from scipy.sparse._csc import csc_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight
from typing import Tuple, Any


@dataclass
class Params_rdmf:
    X_train: csc_matrix
    y_train: ndarray
    X_valid: csc_matrix
    y_valid: ndarray
    _n_estimators: int
    _max_depth: int


_cutoffs = linspace(0.1, 0.9, 17)
_key_rename = {
    "0.0": "active",
    "1.0": "churned",
    "macro avg": "macro avg",
    "weighted avg": "weighted avg"
    }


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
        model: RandomForestClassifier,
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
        model: RandomForestClassifier,
        params: Params_rdmf,
        cutoff: float) -> Tuple[Any]:
    _report_train, _acc_train, _cm_train = performance_measure(
                                                model,
                                                params.X_train,
                                                params.y_train,
                                                cutoff
                                            )
    _report_valid, _acc_valid, _cm_valid = performance_measure(
                                                model,
                                                params.X_valid,
                                                params.y_valid,
                                                cutoff
                                            )
    return (
        _report_train,
        _acc_train,
        _cm_train,
        _report_valid,
        _acc_valid,
        _cm_valid
    )
