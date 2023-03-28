from dataclasses import dataclass
from numpy import linspace, ndarray
from scipy.sparse._csc import csc_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
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


def performance_measure(
        model: RandomForestClassifier,
        X,
        y,
        cutoff: float) -> Tuple[Any]:
    proba = model.predict_proba(X)[:, 1]
    yHat = 1.*(proba > cutoff)
    _report = classification_report(
        y,
        yHat,
        target_names=["active", "churned"],
        digits=3,
        output_dict=True)
    _acc = accuracy_score(y, yHat)
    return _report, _acc


def model_report(
        model: RandomForestClassifier,
        params: Params_rdmf,
        cutoff: float) -> Tuple[Any]:
    _report_train, _acc_train = performance_measure(
        model, params.X_train, params.y_train, cutoff
    )
    _report_valid, _acc_valid = performance_measure(
        model, params.X_valid, params.y_valid, cutoff
    )
    return _report_train, _acc_train, _report_valid, _acc_valid
